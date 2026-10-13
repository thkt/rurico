//! Query candidate behind cfg(test, smoke), sharing production bucket forwards.
use super::*;
use crate::embed::{InferenceMetrics, query317};
use std::time::Duration;

// The real candidate and boundary tests share routing, ownership transfer and metrics.
pub(super) fn execute_queries(
    tokens: Vec<Vec<u32>>,
    tokenize: Duration,
    mut forward: impl FnMut(
        &[IndexedChunk],
        usize,
        &mut PhaseMetrics,
    ) -> Result<Vec<Vec<f32>>, EmbedError>,
) -> Result<query317::Queries, EmbedError> {
    let mut phases = PhaseMetrics::new(EmbedKind::Query);
    phases.chunk_plan = tokenize;
    let counts = vec![1; tokens.len()];
    let documents = execute_document_chunks(
        tokens,
        &counts,
        &query317::default_options(),
        &mut phases,
        true,
        |batch, bucket, metrics| forward(batch, bucket, metrics),
    )?;
    let values = documents
        .into_iter()
        .map(|doc| doc.chunks.into_iter().next().unwrap())
        .collect();
    let mut metrics = InferenceMetrics::from(phases);
    metrics.tokenize = Some(tokenize);
    Ok(query317::Queries {
        values,
        metrics: Some(metrics),
    })
}

#[test]
fn query_preparation_preserves_prefix_truncation_and_order() {
    use crate::model_io::EOS_TOKEN_ID;
    use tokenizers::{PaddingParams, PaddingStrategy};
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing,
    };
    let mut tokenizer = tokenizers::Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("[CLS]".into(), 1),
                    ("[SEP]".into(), 2),
                    ("検索クエリ".into(), 3),
                    (":".into(), 4),
                    ("a".into(), 5),
                    ("b".into(), 6),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    ));
    let long = "a ".repeat(MAX_SEQ_LEN);
    let inputs = ["b", "", long.as_str(), "a"];
    let tokens = query317::prepare(&tokenizer, &inputs).unwrap();
    assert_eq!(tokens[0], [1, 3, 4, 6, 2]);
    assert_eq!(tokens[1], [1, 3, 4, 2]);
    assert_eq!(tokens[3], [1, 3, 4, 5, 2]);
    assert_eq!(tokens[2].len(), MAX_SEQ_LEN);
    assert_eq!(tokens[2][0..4], [1, 3, 4, 5]);
    assert_eq!(tokens[2][MAX_SEQ_LEN - 1], EOS_TOKEN_ID);
    let routed = execute_queries(tokens, Duration::ZERO, |batch, _, _| {
        Ok(batch
            .iter()
            .map(|chunk| vec![chunk.tokens[3] as f32, chunk.tokens.len() as f32])
            .collect())
    })
    .unwrap();
    assert_eq!(
        routed.values,
        [
            vec![6., 5.],
            vec![2., 4.],
            vec![5., MAX_SEQ_LEN as f32],
            vec![5., 5.]
        ]
    );

    assert!(query317::prepare(&tokenizer, &[]).unwrap().is_empty());
    let broken = tokenizers::Tokenizer::new(WordLevel::default());
    assert!(matches!(
        query317::prepare(&broken, &["a"]),
        Err(EmbedError::Tokenizer { .. })
    ));
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(10),
        ..PaddingParams::default()
    }));
    assert!(matches!(
        query317::prepare(&tokenizer, &["a"]),
        Err(EmbedError::Inference { .. })
    ));
}

#[test]
fn candidate_routes_identifiable_rows_and_moves_owned_vectors() {
    use std::{collections::BTreeMap, time::Duration};
    // Longer first query executes after the shorter one; values must return in input order.
    let tokens = vec![vec![7; 129], vec![9; 3], vec![5; 130]];
    let mut pointers = BTreeMap::new();
    let mut routed = Vec::new();
    let result = execute_queries(tokens, Duration::from_nanos(17), |batch, bucket, _| {
        routed.push((
            bucket,
            batch.iter().map(|c| c.global_idx).collect::<Vec<_>>(),
        ));
        Ok(batch
            .iter()
            .map(|chunk| {
                let row = vec![chunk.tokens[0] as f32, -(chunk.tokens.len() as f32)];
                pointers.insert(chunk.global_idx, row.as_ptr());
                row
            })
            .collect())
    })
    .unwrap();
    assert_eq!(routed, [(0, vec![1]), (1, vec![0, 2])]);
    assert_eq!(
        result.values,
        [vec![7., -129.], vec![9., -3.], vec![5., -130.]]
    );
    for (index, row) in result.values.iter().enumerate() {
        assert_eq!(row.as_ptr(), pointers[&index]);
    }
    assert_eq!(
        result.metrics.unwrap().tokenize,
        Some(Duration::from_nanos(17))
    );
    assert!(
        execute_queries(vec![], Duration::ZERO, |_, _, _| panic!("empty forward"))
            .unwrap()
            .values
            .is_empty()
    );
    let mut calls = 0;
    let error = execute_queries(vec![vec![7; 3], vec![9; 129]], Duration::ZERO, |_, _, _| {
        calls += 1;
        Err(EmbedError::NonFiniteOutput)
    })
    .err()
    .unwrap();
    assert!(matches!(error, EmbedError::NonFiniteOutput));
    assert_eq!(calls, 1);
    assert!(execute_queries(vec![vec![7]], Duration::ZERO, |_, _, _| Ok(vec![])).is_err());
}
