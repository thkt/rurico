//! Reuse raw query Encoding; preserve tokenizer pair processing, not concatenation.
use crate::reranker::RerankerError;
use tokenizers::{Encoding, Tokenizer, processors::PostProcessorWrapper};

pub(crate) fn prepare(
    tokenizer: &Tokenizer,
    pairs: &[(&str, &str)],
    reuse: bool,
) -> Result<Vec<Encoding>, RerankerError> {
    prepare_with(tokenizer, pairs, reuse, |_, encoding| Ok(encoding))
}

pub(crate) fn prepare_with<T>(
    tokenizer: &Tokenizer,
    pairs: &[(&str, &str)],
    reuse: bool,
    mut consume: impl FnMut(usize, Encoding) -> Result<T, RerankerError>,
) -> Result<Vec<T>, RerankerError> {
    if !reuse {
        return pairs
            .iter()
            .enumerate()
            .map(|(index, &(q, d))| {
                let encoding = tokenizer
                    .encode((q, d), true)
                    .map_err(RerankerError::tokenizer)?;
                consume(index, encoding)
            })
            .collect();
    }
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    // encode(false) normally still runs post-processing, truncation and padding.
    // Remove these on the raw clone; apply the ORIGINAL processor exactly once.
    let mut raw = tokenizer.clone();
    raw.with_truncation(None)
        .map_err(RerankerError::tokenizer)?;
    raw.with_padding(None);
    raw.with_post_processor(None::<PostProcessorWrapper>);
    let mut cached: Option<(&str, Encoding)> = None;
    pairs
        .iter()
        .enumerate()
        .map(|(index, &(query, doc))| {
            if cached
                .as_ref()
                .is_none_or(|(previous, _)| *previous != query)
            {
                cached = Some((
                    query,
                    raw.encode(query, false).map_err(RerankerError::tokenizer)?,
                ));
            }
            let mut second = raw.encode(doc, false).map_err(RerankerError::tokenizer)?;
            // Match encode_single_sequence(sequence, 1, OffsetType::Byte).
            second.set_type_ids(vec![1; second.len()]);
            let encoding = tokenizer
                .post_process(cached.as_ref().unwrap().1.clone(), Some(second), true)
                .map_err(RerankerError::tokenizer)?;
            consume(index, encoding)
        })
        .collect()
}

pub(crate) fn assert_encoding(actual: &Encoding, expected: &Encoding) {
    assert_eq!(actual.get_ids(), expected.get_ids());
    assert_eq!(actual.get_attention_mask(), expected.get_attention_mask());
    assert_eq!(actual.get_type_ids(), expected.get_type_ids());
    assert_eq!(
        actual.get_special_tokens_mask(),
        expected.get_special_tokens_mask()
    );
    assert_eq!(actual.get_offsets(), expected.get_offsets());
    assert_eq!(actual.get_sequence_ids(), expected.get_sequence_ids());
    assert_eq!(
        actual.get_overflowing().len(),
        expected.get_overflowing().len()
    );
    for (a, b) in actual
        .get_overflowing()
        .iter()
        .zip(expected.get_overflowing())
    {
        assert_encoding(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        AddedToken, PaddingParams, PaddingStrategy, TruncationParams, TruncationStrategy,
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing,
    };
    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("[CLS]".into(), 1),
                    ("[SEP]".into(), 2),
                    ("q".into(), 3),
                    ("d".into(), 4),
                    ("other".into(), 5),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        tokenizer
    }
    #[test]
    fn candidate_pair_entry_preserves_boundaries_settings_and_errors() {
        let mut tokenizer = tokenizer();
        let pairs = [
            ("q", "d"),
            ("q", "other"),
            ("", ""),
            ("other", "q"),
            ("q", "d d d d d d"),
        ];
        let mut outputs = prepare(&tokenizer, &pairs, true).unwrap();
        assert_eq!(outputs[0].get_ids(), [1, 3, 2, 4, 2]);
        assert_eq!(outputs[0].get_type_ids(), [0, 0, 0, 1, 1]);
        assert_eq!(outputs[2].get_ids(), [1, 2, 2]);
        for configured in [false, true] {
            if configured {
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: 7,
                        stride: 1,
                        ..TruncationParams::default()
                    }))
                    .unwrap();
                tokenizer.with_padding(Some(PaddingParams {
                    strategy: PaddingStrategy::Fixed(9),
                    ..PaddingParams::default()
                }));
                outputs = prepare(&tokenizer, &pairs, true).unwrap();
            }
            let baseline = prepare(&tokenizer, &pairs, false).unwrap();
            assert_eq!(outputs.len(), pairs.len());
            assert_eq!(baseline.len(), pairs.len());
            for (actual, expected) in outputs.iter().zip(&baseline) {
                assert_encoding(actual, expected);
            }
        }
        assert!(prepare(&tokenizer, &[], true).unwrap().is_empty());
        tokenizer.with_truncation(None).unwrap();
        tokenizer.with_padding(None);
        tokenizer
            .add_special_tokens([AddedToken::from("[SEP]", true)])
            .unwrap();
        let special = prepare(&tokenizer, &[("q [SEP]", "d")], true).unwrap();
        assert_eq!(special[0].get_ids(), [1, 3, 2, 2, 4, 2]);
        assert_encoding(
            &special[0],
            &tokenizer.encode(("q [SEP]", "d"), true).unwrap(),
        );
        // Without a processor, type/sequence IDs still match pair encode.
        tokenizer.with_post_processor(None::<PostProcessorWrapper>);
        assert_encoding(
            &prepare(&tokenizer, &[("q", "d")], true).unwrap()[0],
            &tokenizer.encode(("q", "d"), true).unwrap(),
        );
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                strategy: TruncationStrategy::OnlySecond,
                ..TruncationParams::default()
            }))
            .unwrap();
        for reuse in [false, true] {
            assert!(matches!(
                prepare(&tokenizer, &[("q q q", "d")], reuse),
                Err(RerankerError::Tokenizer { .. })
            ));
        }

        // Unknown word with no configured UNK token: same operational error.
        let broken = Tokenizer::new(WordLevel::default());
        for reuse in [false, true] {
            assert!(matches!(
                prepare(&broken, &[("missing", "d")], reuse),
                Err(RerankerError::Tokenizer { .. })
            ));
        }
    }
}
