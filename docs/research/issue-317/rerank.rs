//! Pair candidate changes only token preparation; forwards use the existing path.
use super::*;
use crate::reranker::pairs317;
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn execute_pairs(
    ids: &[Vec<u32>],
    masks: &[Vec<u32>],
    start: Instant,
    mut forward: impl FnMut(&[Vec<u32>], &[Vec<u32>], usize) -> Result<Vec<f32>, RerankerError>,
) -> Result<(Vec<f32>, Value), RerankerError> {
    let tokenize_ns = start.elapsed().as_nanos();
    let mut scores = Vec::with_capacity(ids.len());
    let mut forwards = Vec::new();
    if !ids.is_empty() {
        let (bucket, size) =
            dispatch_sub_batch_size(ids.iter().map(Vec::len).max().unwrap(), ids.len());
        for (rows, masks) in ids.chunks(size).zip(masks.chunks(size)) {
            scores.extend(forward(rows, masks, bucket)?);
            forwards.push(json!({"batch_size":rows.len(),"sequence_length":bucket}));
        }
    }
    let metrics =
        json!({"tokenize_ns":tokenize_ns,"wall_ns":start.elapsed().as_nanos(),"forwards":forwards});
    Ok((scores, metrics))
}

pub(super) fn prepare_reused(
    tokenizer: &tokenizers::Tokenizer,
    pairs: &[(&str, &str)],
) -> Result<EncodedPairs, RerankerError> {
    let prepared = pairs317::prepare_with(tokenizer, pairs, true, |index, encoding| {
        let mut ids = encoding.get_ids().to_vec();
        let mut mask = encoding.get_attention_mask().to_vec();
        truncate_pair(&mut ids, &mut mask, MAX_SEQ_LEN, index);
        Ok((ids, mask))
    })?;
    Ok(prepared.into_iter().unzip())
}

#[test]
fn candidate_pair_preparation_routes_ids_masks_and_stops_after_failure() {
    use crate::model_io::EOS_TOKEN_ID;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
    let mut tokenizer = tokenizers::Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("q".into(), 3),
                    ("d".into(), 4),
                    ("other".into(), 5),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let long = "d ".repeat(MAX_SEQ_LEN + 1);
    let pairs = [("q", "d"), ("q", "other"), ("other", long.as_str())];
    let (ids, masks) = prepare_reused(&tokenizer, &pairs).unwrap();
    assert_eq!(ids[0], [3, 4]);
    assert_eq!(ids[1], [3, 5]);
    assert_eq!(ids[2].len(), MAX_SEQ_LEN);
    assert_eq!(ids[2][MAX_SEQ_LEN - 1], EOS_TOKEN_ID);
    let (scores, metrics) = execute_pairs(
        &ids,
        &masks,
        Instant::now(),
        |rows, actual_masks, bucket| {
            assert_eq!(rows, ids);
            assert_eq!(actual_masks, masks);
            assert!(rows.iter().all(|row| row.len() <= bucket));
            Ok(rows
                .iter()
                .map(|row| row[0] as f32 + row[1] as f32)
                .collect())
        },
    )
    .unwrap();
    assert_eq!(scores, [7., 8., 9.]);
    assert_eq!(metrics["forwards"][0]["batch_size"], 3);
    assert!(
        execute_pairs(&[], &[], Instant::now(), |_, _, _| panic!("empty forward"))
            .unwrap()
            .0
            .is_empty()
    );
    let mut calls = 0;
    // Two 8192-token rows exceed the fixed token budget and require separate forwards.
    let error = execute_pairs(
        &vec![vec![3; MAX_SEQ_LEN]; 2],
        &vec![vec![1; MAX_SEQ_LEN]; 2],
        Instant::now(),
        |_, _, _| {
            calls += 1;
            Err(RerankerError::NonFiniteOutput)
        },
    )
    .unwrap_err();
    assert!(matches!(error, RerankerError::NonFiniteOutput));
    assert_eq!(calls, 1);
    let broken = tokenizers::Tokenizer::new(WordLevel::default());
    assert!(prepare_reused(&broken, &[("q", "d")]).is_err());
}
