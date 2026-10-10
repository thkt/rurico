use tracing_test::traced_test;

use super::super::EmbedError;
use super::super::metrics::EmbedKind;
use super::{IndexedChunk, build_indexed_chunks, distribute_into_buckets, split_pooled};
use crate::embed::tokenize_with_prefix;
use crate::model_io::{BUCKET_BOUNDS, assign_bucket, pad_sequences};

fn make_chunk(global_idx: usize, token_count: usize) -> IndexedChunk {
    IndexedChunk {
        global_idx,
        tokens: vec![0u32; token_count],
    }
}

#[test]
fn t_bkt_001_assign_bucket_boundary_128_129() {
    assert_eq!(assign_bucket(1), 0, "len=1 is in bucket 0");
    assert_eq!(
        assign_bucket(128),
        0,
        "len=128 is in bucket 0 (upper bound)"
    );
    assert_eq!(assign_bucket(129), 1, "len=129 crosses into bucket 1");
}

#[test]
fn t_bkt_002_assign_bucket_boundary_512_513() {
    assert_eq!(
        assign_bucket(512),
        1,
        "len=512 is in bucket 1 (upper bound)"
    );
    assert_eq!(assign_bucket(513), 2, "len=513 crosses into bucket 2");
}

#[test]
fn t_bkt_003_assign_bucket_boundary_2048_2049() {
    assert_eq!(
        assign_bucket(2048),
        2,
        "len=2048 is in bucket 2 (upper bound)"
    );
    assert_eq!(assign_bucket(2049), 3, "len=2049 crosses into bucket 3");
}

#[test]
fn t_bkt_004_assign_bucket_max_seq_len() {
    assert_eq!(
        assign_bucket(BUCKET_BOUNDS[3]),
        3,
        "len=MAX_SEQ_LEN is in final bucket"
    );
}

#[test]
fn t_bkt_005_uniform_length_single_bucket() {
    let chunks: Vec<IndexedChunk> = (0..10).map(|i| make_chunk(i, 300)).collect();
    let buckets = distribute_into_buckets(chunks);
    assert_eq!(buckets[0].len(), 0, "bucket 0 (<=128) should be empty");
    assert_eq!(buckets[1].len(), 10, "all len=300 chunks in bucket 1");
    assert_eq!(buckets[2].len(), 0, "bucket 2 (<=2048) should be empty");
    assert_eq!(buckets[3].len(), 0, "bucket 3 should be empty");
}

#[test]
fn t_bkt_006_single_chunk_distribution() {
    let buckets = distribute_into_buckets(vec![make_chunk(0, 50)]);
    let total: usize = buckets.iter().map(Vec::len).sum();
    assert_eq!(total, 1, "single chunk routes to exactly one bucket");
    assert_eq!(buckets[0].len(), 1, "len=50 lands in bucket 0");
}

#[test]
fn build_indexed_chunks_preserves_document_chunk_order() {
    let all_chunks = vec![vec![1u32; 10], vec![2u32; 20], vec![3u32; 30]];
    let indexed = build_indexed_chunks(all_chunks, &[2, 1])
        .expect("balanced chunks_per_doc and all_chunk_tokens must build ok");
    let rows: Vec<_> = indexed
        .iter()
        .map(|c| (c.global_idx, c.tokens.clone()))
        .collect();
    assert_eq!(
        rows,
        vec![(0, vec![1; 10]), (1, vec![2; 20]), (2, vec![3; 30])],
        "global_idx anchors the original document/chunk token order"
    );
}

#[test]
fn build_indexed_chunks_rejects_chunks_per_doc_excess() {
    let all_chunks = vec![vec![1u32; 10]];
    let err = build_indexed_chunks(all_chunks, &[2])
        .expect_err("chunks_per_doc total > all_chunk_tokens length must error");
    match err {
        EmbedError::Inference { message, .. } => assert!(
            message.contains("chunks_per_doc total exceeds"),
            "expected chunks_per_doc-side error wording, got: {message}"
        ),
        other => panic!("expected Inference, got {other:?}"),
    }
}

#[test]
fn build_indexed_chunks_rejects_all_chunk_tokens_excess() {
    let all_chunks = vec![vec![1u32; 10], vec![2u32; 20]];
    let err = build_indexed_chunks(all_chunks, &[1])
        .expect_err("all_chunk_tokens length > chunks_per_doc total must error");
    match err {
        EmbedError::Inference { message, .. } => assert!(
            message.contains("all_chunk_tokens has 1 more"),
            "expected extras count in surplus-side error wording, got: {message}"
        ),
        other => panic!("expected Inference, got {other:?}"),
    }
}

#[test]
fn bucket_execution_restores_document_chunk_rows_with_remainder_and_nonfloor_budget() {
    use super::execute_document_chunks;
    use crate::embed::{EmbedOptions, metrics::PhaseMetrics};
    use std::time::Duration;

    let tokens = [
        (11, 300),
        (12, 5),
        (13, 130),
        (21, 6),
        (22, 400),
        (31, 7),
        (32, 10),
        (33, 20),
        (34, 800),
        (35, 3000),
    ]
    .into_iter()
    .map(|(id, len)| vec![id; len])
    .collect();
    let options = EmbedOptions {
        token_budget: Some(383),
        forward_pause: Some(Duration::ZERO),
    };
    let mut metrics = PhaseMetrics::new(EmbedKind::Batch);
    let mut calls = Vec::new();
    let docs = execute_document_chunks(
        tokens,
        &[3, 2, 5],
        &options,
        &mut metrics,
        true,
        |batch, bucket, _| {
            let target = BUCKET_BOUNDS[bucket];
            let (padded, mask, rows, width) = pad_sequences(batch, None, Some(target));
            assert_eq!((rows, width), (batch.len(), target));
            for (i, chunk) in batch.iter().enumerate() {
                let row = &padded[i * width..(i + 1) * width];
                let row_mask = &mask[i * width..(i + 1) * width];
                assert!(
                    row[..chunk.tokens.len()]
                        .iter()
                        .all(|&id| id == chunk.tokens[0])
                );
                assert!(row[chunk.tokens.len()..].iter().all(|&id| id == 0));
                assert!(row_mask[..chunk.tokens.len()].iter().all(|&m| m == 1));
                assert!(row_mask[chunk.tokens.len()..].iter().all(|&m| m == 0));
            }
            let ids: Vec<_> = batch.iter().map(|c| c.tokens[0]).collect();
            calls.push((bucket, ids.clone()));
            let flat: Vec<_> = ids
                .iter()
                .flat_map(|&id| [id as f32, (id + 100) as f32])
                .collect();
            split_pooled(&flat, batch.len(), 2, EmbedKind::Batch)
        },
    )
    .unwrap();
    assert_eq!(
        calls,
        [
            (0, vec![12, 21]),
            (0, vec![31, 32]),
            (0, vec![33]),
            (1, vec![11]),
            (1, vec![13]),
            (1, vec![22]),
            (2, vec![34]),
            (3, vec![35])
        ]
    );
    assert_eq!(
        docs[0].chunks(),
        [vec![11.0, 111.0], vec![12.0, 112.0], vec![13.0, 113.0]]
    );
    assert_eq!(docs[1].chunks(), [vec![21.0, 121.0], vec![22.0, 122.0]]);
    assert_eq!(
        docs[2].chunks(),
        [
            vec![31.0, 131.0],
            vec![32.0, 132.0],
            vec![33.0, 133.0],
            vec![34.0, 134.0],
            vec![35.0, 135.0]
        ]
    );
    assert_eq!(docs[2].chunk_ids(), ["c0", "c1", "c2", "c3", "c4"]);
    assert_eq!(metrics.bucket_hist, [5, 3, 1, 1]);
    assert_eq!(
        metrics.pause_count, 8,
        "including the final remainder forward"
    );

    let mut metrics = PhaseMetrics::new(EmbedKind::Batch);
    assert!(
        execute_document_chunks(vec![], &[], &options, &mut metrics, true, |_, _, _| panic!(
            "empty input must not forward"
        ))
        .unwrap()
        .is_empty()
    );
    assert_eq!(metrics.pause_count, 0);
}

#[test]
fn bucket_execution_rejects_missing_rows_and_preserves_later_forward_error() {
    use super::execute_document_chunks;
    use crate::embed::{EmbedOptions, metrics::PhaseMetrics};
    use std::time::Duration;

    let options = EmbedOptions {
        token_budget: Some(128),
        forward_pause: Some(Duration::ZERO),
    };
    let mut metrics = PhaseMetrics::new(EmbedKind::Batch);
    let mut calls = Vec::new();
    let error = execute_document_chunks(
        vec![vec![11], vec![21]],
        &[1, 1],
        &options,
        &mut metrics,
        false,
        |batch, _, _| {
            let id = batch[0].tokens[0];
            calls.push(id);
            // Simulate a lost readback row after an earlier document succeeded.
            Ok(if id == 11 { vec![vec![11.0]] } else { vec![] })
        },
    )
    .unwrap_err();
    assert_eq!(calls, [11, 21]);
    assert!(matches!(error, EmbedError::Inference { message, .. }
        if message == "chunk slot 1 not filled by any bucket forward (distribution bug)"));
    assert_eq!(
        metrics.pause_count, 0,
        "ordinary API does not record pauses"
    );

    let mut metrics = PhaseMetrics::new(EmbedKind::Batch);
    let mut calls = Vec::new();
    let error = execute_document_chunks(
        vec![vec![11], vec![21], vec![31]],
        &[1, 1, 1],
        &options,
        &mut metrics,
        true,
        |batch, _, _| {
            let id = batch[0].tokens[0];
            calls.push(id);
            if id == 11 {
                Ok(vec![vec![11.0]])
            } else {
                Err(EmbedError::NonFiniteOutput)
            }
        },
    )
    .unwrap_err();
    assert_eq!(calls, [11, 21], "failure must stop before the next forward");
    assert!(matches!(error, EmbedError::NonFiniteOutput));
    assert_eq!(metrics.pause_count, 1, "only the successful forward pauses");
}

#[test]
fn split_pooled_happy_path_preserves_row_major_order() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let split = split_pooled(&flat, 2, 4, EmbedKind::Query).expect("happy path must split ok");

    assert_eq!(split.len(), 2, "[T-012a] outer Vec must have `batch` rows");
    assert_eq!(
        split[0],
        vec![0.0, 1.0, 2.0, 3.0],
        "[T-012a] row 0 preserved"
    );
    assert_eq!(
        split[1],
        vec![4.0, 5.0, 6.0, 7.0],
        "[T-012a] row 1 preserved"
    );
}

#[test]
fn split_pooled_shape_mismatch_short_returns_buffer_shape_mismatch() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    match split_pooled(&flat, 2, 4, EmbedKind::Query) {
        Err(EmbedError::BufferShapeMismatch { expected, actual }) => {
            assert_eq!(expected, 8, "[T-012b] expected = batch * hidden = 8");
            assert_eq!(actual, 7, "[T-012b] actual = flat.len() = 7");
        }
        other => panic!(
            "[T-012b] expected Err(BufferShapeMismatch {{ expected: 8, actual: 7 }}), \
                 got {other:?}"
        ),
    }
}

#[test]
fn split_pooled_shape_mismatch_long_returns_buffer_shape_mismatch() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0];
    match split_pooled(&flat, 2, 4, EmbedKind::Query) {
        Err(EmbedError::BufferShapeMismatch { expected, actual }) => {
            assert_eq!(expected, 8, "[T-012c] expected = batch * hidden = 8");
            assert_eq!(actual, 9, "[T-012c] actual = flat.len() = 9");
        }
        other => panic!(
            "[T-012c] expected Err(BufferShapeMismatch {{ expected: 8, actual: 9 }}), \
                 got {other:?}"
        ),
    }
}

#[test]
fn split_pooled_zero_batch_returns_empty_vec() {
    let flat: Vec<f32> = Vec::new();
    let split =
        split_pooled(&flat, 0, 768, EmbedKind::Query).expect("zero-batch flat must split ok");
    assert!(
        split.is_empty(),
        "[T-012d] batch=0 must yield an empty outer Vec, got {split:?}"
    );
}

#[test]
fn split_pooled_single_batch_for_embed_query_path() {
    let flat: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let split = split_pooled(&flat, 1, 5, EmbedKind::Query).expect("single-batch must split ok");

    assert_eq!(split.len(), 1, "[T-012e] batch=1 yields a single inner row");
    assert_eq!(
        split[0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        "[T-012e] inner row must equal the full flat buffer"
    );
}

// T-015a / FR-002c / AC-1 (sub-case of spec T-015)
//
// [T-015a] `split_pooled` rejects `NaN` with `NonFiniteOutput`. The
// `is_finite` guard catches non-finite outputs that Phase 3b would
// otherwise miss: the all-zero-mask `0/0` source is
// already rejected by `validate_attention_mask` upstream, but other
// sources (corrupt weights, kernel overflow) are not. The check runs
// on the already-readback flat buffer so it does not defeat the
// readback-free hot path (ADR 0002 primary lever).
#[test]
fn split_pooled_rejects_nan_with_non_finite_output() {
    let flat: Vec<f32> = vec![0.0, 1.0, f32::NAN, 3.0];
    match split_pooled(&flat, 1, 4, EmbedKind::Query) {
        Err(EmbedError::NonFiniteOutput) => {}
        other => panic!("[T-015a] expected Err(NonFiniteOutput), got {other:?}"),
    }
}

// T-015b / FR-002c / AC-1 (sub-case of spec T-015)
//
// [T-015b] `split_pooled` rejects `±Inf` with `NonFiniteOutput`. Same
// safety-net contract as the NaN case; covers the kernel-overflow
// failure mode separately so a regression that handles only `NaN` is
// caught.
#[test]
fn split_pooled_rejects_positive_inf_with_non_finite_output() {
    let flat: Vec<f32> = vec![0.0, f32::INFINITY, 2.0, 3.0];
    match split_pooled(&flat, 1, 4, EmbedKind::Query) {
        Err(EmbedError::NonFiniteOutput) => {}
        other => panic!("[T-015b] expected Err(NonFiniteOutput), got {other:?}"),
    }
}

// T-015c / FR-002c / AC-1 (sub-case of spec T-015)
//
// [T-015c] `split_pooled` also rejects `-Inf` (not just positive
// infinity). f32::is_finite returns false for both — guarding the
// assumption explicitly so a future check that uses `> f32::MAX`
// alone would fail.
#[test]
fn split_pooled_rejects_negative_inf_with_non_finite_output() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, f32::NEG_INFINITY];
    match split_pooled(&flat, 1, 4, EmbedKind::Query) {
        Err(EmbedError::NonFiniteOutput) => {}
        other => panic!("[T-015c] expected Err(NonFiniteOutput), got {other:?}"),
    }
}

// T-012f (sub-case of spec T-012): emits warn so operators can diagnose corrupt readback (see ADR 0007).
#[traced_test]
#[test]
fn split_pooled_emits_warn_on_buffer_shape_mismatch() {
    let flat: Vec<f32> = vec![0.0; 7]; // expected 8
    let _ = split_pooled(&flat, 2, 4, EmbedKind::Query);
    assert!(
        logs_contain("split_pooled: buffer shape mismatch"),
        "warn must be emitted on shape mismatch"
    );
    assert!(
        logs_contain("call_site=\"query\""),
        "call_site field must be emitted so query/batch paths are distinguishable",
    );
}

// T-015d (sub-case of spec T-015): emits warn so operators can distinguish kernel overflow / corrupt
// weights from upstream input errors (see ADR 0007).
#[traced_test]
#[test]
fn split_pooled_emits_warn_on_non_finite_output() {
    let flat: Vec<f32> = vec![0.0, 1.0, f32::NAN, 3.0];
    let _ = split_pooled(&flat, 1, 4, EmbedKind::Query);
    assert!(
        logs_contain("split_pooled: non-finite output"),
        "warn must be emitted on non-finite output"
    );
}

fn word_tokenizer(words: &[String]) -> tokenizers::Tokenizer {
    use crate::embed::DOCUMENT_PREFIX;
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
        processors::template::TemplateProcessing,
    };

    let vocab = [
        ("[UNK]".into(), 0),
        ("[BOS]".into(), 1),
        ("[EOS]".into(), 2),
        (DOCUMENT_PREFIX.trim().into(), 3),
    ]
    .into_iter()
    .chain(
        words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), u32::try_from(i + 4).unwrap())),
    )
    .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[BOS] $A [EOS]")
            .unwrap()
            .special_tokens(vec![("[BOS]", 1), ("[EOS]", 2)])
            .build()
            .unwrap(),
    ));
    tokenizer
}

// Distinct word IDs expose dropped tails, wrong overlaps, prefixes and document order.
// WordLevel deliberately has no prefix-boundary merges; real-tokenizer checks remain ignored.
#[test]
fn document_planning_preserves_prefix_overlap_tail_and_document_order() {
    use super::plan_document_chunks;
    use crate::embed::{DOCUMENT_PREFIX, extract_prefix_tokens, max_content};

    let words: Vec<_> = (0..9000).map(|i| format!("語{i}")).collect();
    let tokenizer = word_tokenizer(&words);
    let prefix = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    let long = words.join(" ");
    // 8192 slots minus BOS, prefix and EOS = 8189 content tokens.
    let expected_first: Vec<_> = [1, 3].into_iter().chain(4..8193).chain([2]).collect();
    // Second chunk starts 2048 content tokens before the accepted first end.
    let expected_last: Vec<_> = [1, 3].into_iter().chain(6145..9004).chain([2]).collect();
    let budget = max_content(prefix.len());
    // The extra-token candidate forces adaptive shrink through the production planner.
    // Its next start must use the accepted end, not the original oversized candidate.
    for candidate_budget in [budget, budget + 1] {
        let (chunks, counts) =
            plan_document_chunks(&tokenizer, &["語0", &long, ""], &prefix, candidate_budget)
                .unwrap();
        assert_eq!(counts, [1, 2, 1]);
        assert_eq!(chunks[0], [1, 3, 4, 2]);
        assert_eq!(chunks[1], expected_first);
        assert_eq!(chunks[2], expected_last);
        assert_eq!(chunks[3], [1, 3, 2]);
    }
    let empty = plan_document_chunks(&tokenizer, &[], &prefix, budget).unwrap();
    assert_eq!(empty, (vec![], vec![]));
}

#[test]
fn shrink_chunk_to_fit_preserves_fitting_range_and_rejects_empty_range() {
    use super::shrink_chunk_to_fit;

    let tokenizer = word_tokenizer(&["語0".into()]);
    let text = "語0";
    let encoding = tokenizer.encode(text, false).unwrap();
    let mut end = 1;
    let ids = shrink_chunk_to_fit(&tokenizer, text, encoding.get_offsets(), 0, &mut end).unwrap();
    assert_eq!(ids, [1, 3, 4, 2]);
    assert_eq!(end, 1, "a fitting range must not shrink");

    end = 0;
    let error =
        shrink_chunk_to_fit(&tokenizer, text, encoding.get_offsets(), 0, &mut end).unwrap_err();
    assert!(
        matches!(error, EmbedError::Inference { message, .. } if message.contains("cannot fit"))
    );
}

// An added token spans the prefix/content boundary. Slicing an unprefixed
// encoding and concatenating prefix IDs would incorrectly yield [3, 4].
#[test]
fn long_planning_retokenizes_prefix_boundary_without_losing_utf8_tail() {
    use super::plan_document_chunks;
    use crate::embed::{DOCUMENT_PREFIX, extract_prefix_tokens, max_content};
    use tokenizers::AddedToken;

    let words: Vec<_> = (0..9000).map(|i| format!("語{i}")).collect();
    let mut tokenizer = word_tokenizer(&words);
    tokenizer
        .add_tokens([AddedToken::from(format!("{DOCUMENT_PREFIX}語0"), false)])
        .unwrap();
    let prefix = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    let text = words.join(" ");
    let budget = max_content(prefix.len());
    let (chunks, counts) = plan_document_chunks(&tokenizer, &[&text], &prefix, budget).unwrap();
    assert_eq!(counts, [2]);
    let first: Vec<_> = [1, 9004].into_iter().chain(5..8193).chain([2]).collect();
    let last: Vec<_> = [1, 3].into_iter().chain(6145..9004).chain([2]).collect();
    assert_eq!(chunks, [first, last]);

    // Token count can decrease when a longer candidate completes an added token.
    // A shrink optimization must not infer fitting ends from monotonicity.
    let merged = format!("{DOCUMENT_PREFIX}{}", words[..8190].join(" "));
    tokenizer
        .add_tokens([AddedToken::from(merged, false)])
        .unwrap();
    let short_candidate =
        tokenize_with_prefix(&tokenizer, &words[..8189].join(" "), DOCUMENT_PREFIX).unwrap();
    let longer_candidate =
        tokenize_with_prefix(&tokenizer, &words[..8190].join(" "), DOCUMENT_PREFIX).unwrap();
    assert!(short_candidate.seq_len > longer_candidate.seq_len);
    let encoding = tokenizer.encode(text.as_str(), false).unwrap();
    let mut end = 8191;
    let accepted =
        super::shrink_chunk_to_fit(&tokenizer, &text, encoding.get_offsets(), 0, &mut end).unwrap();
    assert_eq!(end, 8191, "the merged candidate plus the next token fits");
    assert_eq!(accepted, [1, 9005, 8194, 2]);
}
