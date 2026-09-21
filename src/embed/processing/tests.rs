use tracing_test::traced_test;

use super::super::EmbedError;
use super::super::metrics::EmbedKind;
use super::{IndexedChunk, build_indexed_chunks, distribute_into_buckets, split_pooled};
use crate::model_io::{BUCKET_BOUNDS, assign_bucket};

/// Full-position helper used by T-BKT-008 to express a shuffled multi-chunk
/// doc layout. `make_chunk` delegates here with each chunk as its own
/// single-chunk document.
fn make_chunk_at(
    global_idx: usize,
    doc_idx: usize,
    chunk_in_doc: usize,
    token_count: usize,
) -> IndexedChunk {
    IndexedChunk {
        global_idx,
        doc_idx,
        chunk_in_doc,
        tokens: vec![0u32; token_count],
    }
}

fn make_chunk(global_idx: usize, token_count: usize) -> IndexedChunk {
    make_chunk_at(global_idx, global_idx, 0, token_count)
}

// T-BKT-001: assign_bucket boundary at 128 / 129
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

// T-BKT-002: assign_bucket boundary at 512 / 513
#[test]
fn t_bkt_002_assign_bucket_boundary_512_513() {
    assert_eq!(
        assign_bucket(512),
        1,
        "len=512 is in bucket 1 (upper bound)"
    );
    assert_eq!(assign_bucket(513), 2, "len=513 crosses into bucket 2");
}

// T-BKT-003: assign_bucket boundary at 2048 / 2049
#[test]
fn t_bkt_003_assign_bucket_boundary_2048_2049() {
    assert_eq!(
        assign_bucket(2048),
        2,
        "len=2048 is in bucket 2 (upper bound)"
    );
    assert_eq!(assign_bucket(2049), 3, "len=2049 crosses into bucket 3");
}

// T-BKT-004: assign_bucket accepts up to MAX_SEQ_LEN (8192)
#[test]
fn t_bkt_004_assign_bucket_max_seq_len() {
    assert_eq!(
        assign_bucket(BUCKET_BOUNDS[3]),
        3,
        "len=MAX_SEQ_LEN is in final bucket"
    );
}

// T-BKT-005: all chunks at len=300 land in bucket 1; other buckets stay empty
#[test]
fn t_bkt_005_uniform_length_single_bucket() {
    let chunks: Vec<IndexedChunk> = (0..10).map(|i| make_chunk(i, 300)).collect();
    let buckets = distribute_into_buckets(chunks);
    assert_eq!(buckets[0].len(), 0, "bucket 0 (<=128) should be empty");
    assert_eq!(buckets[1].len(), 10, "all len=300 chunks in bucket 1");
    assert_eq!(buckets[2].len(), 0, "bucket 2 (<=2048) should be empty");
    assert_eq!(buckets[3].len(), 0, "bucket 3 should be empty");
}

// T-BKT-006: single-chunk distribution — other buckets empty, one bucket has 1
#[test]
fn t_bkt_006_single_chunk_distribution() {
    let buckets = distribute_into_buckets(vec![make_chunk(0, 50)]);
    let total: usize = buckets.iter().map(Vec::len).sum();
    assert_eq!(total, 1, "single chunk routes to exactly one bucket");
    assert_eq!(buckets[0].len(), 1, "len=50 lands in bucket 0");
}

#[test]
fn build_indexed_chunks_tracks_doc_structure() {
    // doc 0 has 2 chunks, doc 1 has 1 chunk → chunks_per_doc = [2, 1]
    let all_chunks = vec![vec![1u32; 10], vec![2u32; 20], vec![3u32; 30]];
    let indexed = build_indexed_chunks(all_chunks, &[2, 1])
        .expect("balanced chunks_per_doc and all_chunk_tokens must build ok");
    let shape: Vec<(usize, usize, usize, usize)> = indexed
        .iter()
        .map(|c| (c.global_idx, c.doc_idx, c.chunk_in_doc, c.tokens.len()))
        .collect();
    assert_eq!(
        shape,
        vec![(0, 0, 0, 10), (1, 0, 1, 20), (2, 1, 0, 30)],
        "global_idx runs 0..N while doc_idx + chunk_in_doc track doc layout"
    );
}

// Defense-in-depth: a regression that emits more `chunks_per_doc` than
// chunks (e.g., wrong loop bound) previously panicked via
// `expect("chunks_per_doc total must match all_chunk_tokens length")`.
// The Result variant surfaces it as a structured EmbedError instead.
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

// T-BKT-007: 10 chunks spanning all 4 buckets round-trip in original order
//
// Mirrors the restoration that `embed_documents_batch_chunked` performs:
// forward writes `out[chunk.global_idx]`, so flattening every bucket and
// sorting by `global_idx` must recover the original insertion order.
// Testing this on `distribute_into_buckets` keeps the check MLX-free.
#[test]
fn t_bkt_007_cross_bucket_order_preserved() {
    // bucket 0 (≤128): idx 0, 4 | bucket 1 (≤512): idx 3, 6, 8
    // bucket 2 (≤2048): idx 1, 7 | bucket 3 (≤MAX): idx 2, 5, 9
    let lengths = [50usize, 1500, 3000, 200, 100, 5000, 300, 800, 400, 7000];
    let chunks: Vec<IndexedChunk> = lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| make_chunk(i, len))
        .collect();

    let buckets = distribute_into_buckets(chunks);

    let sizes: Vec<usize> = buckets.iter().map(Vec::len).collect();
    assert!(
        buckets.iter().all(|b| !b.is_empty()),
        "test input must span all 4 buckets (got {sizes:?})"
    );
    assert_eq!(
        sizes.iter().sum::<usize>(),
        lengths.len(),
        "every chunk must land in exactly one bucket"
    );

    let mut flat: Vec<&IndexedChunk> = buckets.iter().flatten().collect();
    flat.sort_by_key(|c| c.global_idx);
    let recovered: Vec<usize> = flat.iter().map(|c| c.tokens.len()).collect();
    assert_eq!(
        recovered,
        lengths.to_vec(),
        "flatten + sort-by-global_idx must recover original chunk order"
    );
}

// T-BKT-009: empty input pipeline produces zero work
//
// Guards the building blocks behind `texts.is_empty() → Vec::new()` in
// `embed_documents_batch_chunked`: even if the early return were removed,
// an empty `all_chunk_tokens` would yield all-empty buckets, the forward
// loop would not execute, and the `Vec<Option<Vec<f32>>>` (size 0) would
// collect to `Vec::new()`. MLX-free proxy for the end-to-end contract.
#[test]
fn t_bkt_009_empty_input_zero_subbatches() {
    let indexed = build_indexed_chunks(Vec::new(), &[]).expect("empty inputs must build ok");
    assert!(
        indexed.is_empty(),
        "empty chunk tokens → empty IndexedChunk vec"
    );

    let buckets = distribute_into_buckets(indexed);
    assert!(
        buckets.iter().all(Vec::is_empty),
        "empty input → every bucket stays empty"
    );
}

// T-BKT-008: same-doc chunks cluster contiguously inside each bucket after
// the in-loop sort, with chunk_in_doc order preserved.
//
// Uses `IndexedChunk::doc_order_key` — the same function the forward loop
// in `embed_documents_batch_chunked` calls — so this test exercises the
// production sort contract rather than a hand-rolled mirror.
#[test]
fn t_bkt_008_same_doc_chunks_cluster_after_sort() {
    // doc 0 produces 3 chunks split across bucket 0 (×2) and bucket 2 (×1).
    // Inputs are shuffled so the sort must actively reorder. A no-op sort
    // would leave bucket 0 as [doc1, doc0_c0, doc2, doc0_c1] and fail.
    let input = vec![
        make_chunk_at(3, 1, 0, 60),
        make_chunk_at(0, 0, 0, 50),
        make_chunk_at(2, 0, 2, 1000),
        make_chunk_at(4, 2, 0, 70),
        make_chunk_at(1, 0, 1, 100),
    ];

    let mut buckets = distribute_into_buckets(input);
    for bucket in &mut buckets {
        bucket.sort_by_key(IndexedChunk::doc_order_key);
    }

    let b0: Vec<(usize, usize)> = buckets[0].iter().map(IndexedChunk::doc_order_key).collect();
    assert_eq!(
        b0,
        vec![(0, 0), (0, 1), (1, 0), (2, 0)],
        "bucket 0: doc 0 chunks contiguous in chunk_in_doc order, then doc 1, doc 2"
    );

    let b2: Vec<(usize, usize)> = buckets[2].iter().map(IndexedChunk::doc_order_key).collect();
    assert_eq!(
        b2,
        vec![(0, 2)],
        "bucket 2: only doc 0's third chunk (the one that overflowed the smaller buckets)"
    );
}

// T-012a / FR-002a / AC-1 (sub-case of spec T-012)
//
// [T-012a] Happy path: `flat.len() == batch * hidden`. `split_pooled`
// returns `batch` rows, each `hidden` long, with values preserved in
// row-major order. This is the contract Phase 3b's `forward_sub_batch`
// and `embed_query_truncated` rely on after the GPU pool reduces
// readback to `batch * hidden` floats.
#[test]
fn split_pooled_happy_path_preserves_row_major_order() {
    let flat: Vec<f32> = vec![
        0.0, 1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, 7.0, // row 1
    ];
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

// T-012b / FR-002a / AC-1 (sub-case of spec T-012)
//
// [T-012b] Shape mismatch (short): `flat.len() = 7` with `batch = 2,
// hidden = 4` (expected 8). Spec scenario verbatim — exercises the
// whole point of the helper, which is to fail fast rather than silently
// slice an incomplete final row.
#[test]
fn split_pooled_shape_mismatch_short_returns_buffer_shape_mismatch() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // len 7
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

// T-012c / FR-002a / AC-1 (sub-case of spec T-012)
//
// [T-012c] Shape mismatch (long): `flat.len() = 9` with `batch = 2,
// hidden = 4` (expected 8). Regression guard against a future
// implementer that reads the first `batch * hidden` floats and silently
// drops the tail — short-direction tests alone would not catch that.
#[test]
fn split_pooled_shape_mismatch_long_returns_buffer_shape_mismatch() {
    let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0]; // len 9
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

// T-012d / FR-002a / AC-1 (sub-case of spec T-012)
//
// [T-012d] Zero-length boundary: `batch = 0, hidden = N, flat = &[]`.
// Expected length `0 * hidden = 0` matches `flat.len() = 0`, so the
// contract is satisfied and `split_pooled` returns `Ok(vec![])`.
// Documents the "empty input is not an error" branch so a future
// implementer cannot accidentally treat zero as the mismatch case.
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

// T-012e / FR-002a / AC-1 (sub-case of spec T-012)
//
// [T-012e] Single-batch case used by `embed_query_truncated`. After the
// Phase 3b rewire, the query path calls `split_pooled(flat, 1, hidden)`
// and unwraps the single inner row. Verifies the helper does not require
// `batch >= 2`.
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
