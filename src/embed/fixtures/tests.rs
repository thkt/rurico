use super::*;
use std::io::Cursor;

fn sample_docs() -> Vec<ChunkedEmbedding> {
    vec![
        ChunkedEmbedding::try_new(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]).unwrap(),
        ChunkedEmbedding::try_new(vec![vec![-0.7, 0.8, 0.9]]).unwrap(),
    ]
}

#[test]
fn save_and_load_round_trips_identically() {
    let docs = sample_docs();
    let mut buf = Vec::new();
    save(&mut buf, &docs).unwrap();
    let loaded = load(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(docs.len(), loaded.len());
    for (a, b) in docs.iter().zip(&loaded) {
        assert_eq!(a.chunks, b.chunks);
    }
}

#[test]
fn compare_identical_fixtures_reports_zero_diff() {
    let docs = sample_docs();
    let diff = compare(&docs, &docs).unwrap();
    assert_eq!(diff.max_abs_diff, 0.0);
    assert!(
        (diff.cosine_min - 1.0).abs() < 1e-6,
        "expected cosine=1.0, got {}",
        diff.cosine_min
    );
}

// Regression: identical all-zero fixtures must report cosine=1.0 (not 0.0)
// so the NFR-001 threshold stays passable on zero-norm inputs.
#[test]
fn compare_identical_zero_fixtures_reports_cosine_one() {
    let zeros = vec![ChunkedEmbedding::try_new(vec![vec![0.0f32; 8]]).unwrap()];
    let diff = compare(&zeros, &zeros).unwrap();
    assert_eq!(diff.cosine_min, 1.0);
    assert_eq!(diff.max_abs_diff, 0.0);
}

// Regression: zero vs nonzero must report cosine=0.0 (clear mismatch).
#[test]
fn compare_zero_versus_nonzero_reports_cosine_zero() {
    let a = vec![ChunkedEmbedding::try_new(vec![vec![0.0f32; 3]]).unwrap()];
    let b = vec![ChunkedEmbedding::try_new(vec![vec![1.0, 2.0, 3.0]]).unwrap()];
    let diff = compare(&a, &b).unwrap();
    assert_eq!(diff.cosine_min, 0.0);
}

#[test]
fn compare_divergent_fixtures_reports_max_abs_diff() {
    let a = sample_docs();
    let mut b = sample_docs();
    b[0].chunks[0][0] += 0.01;
    let diff = compare(&a, &b).unwrap();
    assert!((diff.max_abs_diff - 0.01).abs() < 1e-6);
    assert!(diff.cosine_min < 1.0);
}

#[test]
fn compare_doc_count_mismatch_returns_err() {
    let a = sample_docs();
    let b: Vec<_> = a.iter().skip(1).cloned().collect();
    assert_eq!(
        compare(&a, &b),
        Err(ShapeMismatch::DocCount {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn compare_chunk_count_mismatch_returns_err_with_doc_index() {
    let a = sample_docs();
    let mut b = sample_docs();
    b[0].chunks.pop();
    match compare(&a, &b) {
        Err(ShapeMismatch::ChunkCount {
            doc,
            expected,
            actual,
        }) => {
            assert_eq!(doc, 0);
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected ChunkCount mismatch, got {other:?}"),
    }
}

#[test]
fn compare_dim_mismatch_returns_err_with_chunk_index() {
    let a = sample_docs();
    let mut b = sample_docs();
    b[1].chunks[0].push(0.0);
    match compare(&a, &b) {
        Err(ShapeMismatch::Dim {
            doc,
            chunk,
            expected,
            actual,
        }) => {
            assert_eq!(doc, 1);
            assert_eq!(chunk, 0);
            assert_eq!(expected, 3);
            assert_eq!(actual, 4);
        }
        other => panic!("expected Dim mismatch, got {other:?}"),
    }
}

#[test]
fn default_tolerances_match_spec_nfr_001() {
    assert!((DEFAULT_COSINE_MIN - 0.99999).abs() < f32::EPSILON);
    assert!((DEFAULT_MAX_ABS_DIFF - 1e-5).abs() < f32::EPSILON);
}

// Corrupt header: stream ends after `num_docs = 1` so reading the
// first `num_chunks` u32 fails with `UnexpectedEof`. The previous
// `as usize` cast read whatever the partial buffer happened to contain
// and could trigger a multi-GB `Vec::with_capacity` on hostile input.
#[test]
fn load_rejects_truncated_header_after_num_docs() {
    let bytes = (1u32).to_le_bytes().to_vec();
    let err = load(&mut Cursor::new(&bytes)).expect_err("truncated header must error");
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn load_rejects_doc_with_zero_chunks() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_docs = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // num_chunks = 0

    let err = load(&mut Cursor::new(&bytes)).expect_err("zero chunks must error");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("at least one chunk"),
        "unexpected error: {err}"
    );
}

// Corrupt header: a chunk declares `dim` but the f32 payload is shorter
// than `dim * 4` bytes, so `read_exact` fails with `UnexpectedEof` after
// a small bounded allocation. Uses a small `dim` to avoid the 16 GiB
// allocation that `dim = u32::MAX` would force on 64-bit hosts.
#[test]
fn load_rejects_chunk_with_truncated_payload() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_docs = 1
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_chunks = 1
    bytes.extend_from_slice(&64u32.to_le_bytes()); // dim = 64 → expects 256 payload bytes
    bytes.extend_from_slice(&[0u8; 16]); // only 16 bytes of payload
    let err = load(&mut Cursor::new(&bytes)).expect_err("truncated payload must error");
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

// 32-bit-only: `dim.checked_mul(4)` overflows `usize` on 32-bit targets
// when `dim > usize::MAX / 4`. Gated to avoid the 16 GiB allocation that
// would happen on 64-bit hosts before `read_exact` could return.
#[cfg(target_pointer_width = "32")]
#[test]
fn load_rejects_dim_times_4_overflow_on_32bit() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_docs = 1
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_chunks = 1
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // dim = u32::MAX → dim * 4 overflows usize
    let err = load(&mut Cursor::new(&bytes)).expect_err("dim * 4 overflow must error");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

// T-105-004: save_load_round_trip_with_zero_docs_yields_empty_vec
#[test]
fn save_load_round_trip_with_zero_docs_yields_empty_vec() {
    let docs: Vec<ChunkedEmbedding> = Vec::new();
    let mut buf = Vec::new();
    save(&mut buf, &docs).unwrap();
    let loaded = load(&mut Cursor::new(&buf)).unwrap();
    assert!(
        loaded.is_empty(),
        "0-doc fixture must round-trip back to an empty Vec"
    );
}

// T-105-005: compare_returns_zero_diff_when_both_sides_empty
#[test]
fn compare_returns_zero_diff_when_both_sides_empty() {
    let diff = compare(&[], &[]).unwrap();
    assert_eq!(diff.max_abs_diff, 0.0);
    assert_eq!(
        diff.cosine_min, 1.0,
        "empty fixtures must report cosine=1.0 (vacuous match), not 0.0"
    );
}
