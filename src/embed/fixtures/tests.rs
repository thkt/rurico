use super::*;
use crate::embed::{EmbeddingValidationError, EmptyChunksError};
use std::io::Cursor;

fn sample_docs() -> Vec<ChunkedEmbedding> {
    vec![
        ChunkedEmbedding::try_new(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]).unwrap(),
        ChunkedEmbedding::try_new(vec![vec![-0.7, 0.8, 0.9]]).unwrap(),
    ]
}

#[test]
fn save_and_load_round_trips_identically() {
    let mut docs = sample_docs();
    // Validation is per document, accepts zero/non-unit/large finite values,
    // and does not infer a shared model dimension for the whole fixture.
    docs.push(ChunkedEmbedding::try_new(vec![vec![0.0, -0.0], vec![-2.0, f32::MAX]]).unwrap());
    let mut buf = Vec::new();
    save(&mut buf, &docs).unwrap();
    let loaded = load(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(docs.len(), loaded.len());
    for (a, b) in docs.iter().zip(&loaded) {
        assert_eq!(a.chunks, b.chunks);
        assert_eq!(a.chunk_ids(), b.chunk_ids());
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

#[test]
fn compare_identical_zero_fixtures_reports_cosine_one() {
    let zeros = vec![ChunkedEmbedding::try_new(vec![vec![0.0f32; 8]]).unwrap()];
    let diff = compare(&zeros, &zeros).unwrap();
    assert_eq!(diff.cosine_min, 1.0);
    assert_eq!(diff.max_abs_diff, 0.0);
}

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
        Err(CompareError::Shape(ShapeMismatch::DocCount {
            expected: 2,
            actual: 1
        }))
    );
}

#[test]
fn compare_chunk_count_mismatch_returns_err_with_doc_index() {
    let a = sample_docs();
    let mut b = sample_docs();
    b[0].chunks.pop();
    match compare(&a, &b) {
        Err(CompareError::Shape(ShapeMismatch::ChunkCount {
            doc,
            expected,
            actual,
        })) => {
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
        Err(CompareError::Shape(ShapeMismatch::Dim {
            doc,
            chunk,
            expected,
            actual,
        })) => {
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
    assert_eq!(
        err.get_ref()
            .unwrap()
            .downcast_ref::<EmbeddingValidationError>(),
        Some(&EmbeddingValidationError::EmptyChunks(EmptyChunksError))
    );
    assert!(
        err.to_string().contains("at least one chunk"),
        "unexpected error: {err}"
    );
}

#[test]
fn load_rejects_invalid_vector_content() {
    use crate::embed::EmbeddingValidationError as E;
    let cases = [
        (vec![vec![]], E::EmptyVector { chunk: 0 }),
        (
            vec![vec![1.0, 2.0], vec![3.0]],
            E::DimensionMismatch {
                chunk: 1,
                expected: 2,
                actual: 1,
            },
        ),
        (
            vec![vec![f32::NAN]],
            E::NonFiniteValue {
                chunk: 0,
                element: 0,
            },
        ),
        (
            vec![vec![f32::INFINITY]],
            E::NonFiniteValue {
                chunk: 0,
                element: 0,
            },
        ),
        (
            vec![vec![f32::NEG_INFINITY]],
            E::NonFiniteValue {
                chunk: 0,
                element: 0,
            },
        ),
    ];
    for (chunks, expected) in cases {
        // The legacy constructor and writer can represent invalid old fixtures.
        let doc = ChunkedEmbedding::try_new(chunks).unwrap();
        let mut bytes = Vec::new();
        save(&mut bytes, &[doc]).unwrap();
        // Invalid content must win over a later chunk's truncated payload.
        let mut later_truncated = bytes.clone();
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        later_truncated[4..8].copy_from_slice(&(count + 1).to_le_bytes());
        later_truncated.extend_from_slice(&u32::MAX.to_le_bytes());
        for input in [bytes, later_truncated] {
            let err = load(&mut Cursor::new(input)).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
            assert_eq!(err.get_ref().unwrap().downcast_ref::<E>(), Some(&expected));
        }
    }
}

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
// when `dim > usize::MAX / 4`. The architecture-independent hostile-header
// test also rejects this payload on 64-bit without header-derived allocation.
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

#[test]
fn compare_returns_zero_diff_when_both_sides_empty() {
    let diff = compare(&[], &[]).unwrap();
    assert_eq!(diff.max_abs_diff, 0.0);
    assert_eq!(
        diff.cosine_min, 1.0,
        "empty fixtures must report cosine=1.0 (vacuous match), not 0.0"
    );
}

fn conditions() -> GenerationConditions {
    GenerationConditions {
        producer: Producer::Rurico,
        model: "synthetic".into(),
        model_revision: "test-v1".into(),
        tokenizer: "none: synthetic vectors".into(),
        inputs: vec!["public A".into(), "public B".into()],
        generation_code: "src/embed/fixtures/tests.rs".into(),
        settings: "synthetic, no inference".into(),
    }
}

#[test]
fn versioned_save_byte_limit_is_checked_before_writing() {
    let docs = sample_docs();
    let metadata = conditions();
    // Version header + JSON + legacy counts (2 docs, 3 chunks) + 9 f32 values.
    let size = 12 + serde_json::to_vec(&metadata).unwrap().len() + 4 + 2 * 4 + 3 * 4 + 9 * 4;
    let mut public_output = Vec::new();
    save_versioned(&mut public_output, &docs, &metadata).unwrap();
    assert_eq!(public_output.len(), size);

    for limit in [size + 1, size] {
        let mut output = Vec::new();
        save_versioned_with_limit(&mut output, &docs, &metadata, limit).unwrap();
        assert_eq!(output, public_output, "limit {limit}");
    }

    // A pre-existing prefix makes writes observable without assuming an empty writer.
    let prefix = b"existing output";
    let mut output = prefix.to_vec();
    let result = save_versioned_with_limit(&mut output, &docs, &metadata, size - 1);
    assert_eq!(output, prefix, "oversized fixture wrote to external writer");
    let err = result.expect_err("oversized fixture must be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert_eq!(err.to_string(), "fixture exceeds byte limit");
}

#[test]
fn versioned_provenance_and_legacy_unknown_round_trip() {
    let docs = sample_docs();
    for producer in [Producer::Rurico, Producer::OfficialReference] {
        let mut metadata = conditions();
        metadata.producer = producer;
        let mut bytes = Vec::new();
        save_versioned(&mut bytes, &docs, &metadata).unwrap();
        let fixture = load_fixture(&mut Cursor::new(&bytes), bytes.len()).unwrap();
        assert_eq!(fixture.generation, Some(metadata));
        assert_eq!(compare(&docs, &fixture.docs).unwrap().max_abs_diff, 0.0);
        assert!(load_fixture(&mut Cursor::new(&bytes), bytes.len() - 1).is_err());
        for end in 0..bytes.len() {
            assert!(
                load(&mut Cursor::new(&bytes[..end])).is_err(),
                "truncation {end}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(
            load(&mut Cursor::new(trailing))
                .unwrap_err()
                .to_string()
                .contains("trailing")
        );
    }
    let mut bytes = Vec::new();
    save(&mut bytes, &docs).unwrap();
    assert!(
        load_fixture(&mut Cursor::new(&bytes), bytes.len())
            .unwrap()
            .generation
            .is_none()
    );
    bytes.push(0);
    assert!(
        load(&mut Cursor::new(bytes))
            .unwrap_err()
            .to_string()
            .contains("trailing")
    );
}

#[test]
fn tiny_hostile_headers_and_metadata_fail_without_header_allocation() {
    for words in [
        vec![u32::MAX - 1],
        vec![1, u32::MAX],
        vec![1, 1, u32::MAX],
        vec![MAGIC, 2],
        vec![MAGIC, 1, u32::MAX],
        vec![MAGIC, 1, 1, 0],
    ] {
        let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(load(&mut Cursor::new(bytes)).is_err());
    }
    let mut metadata = conditions();
    metadata.model_revision.clear();
    let mut output = Vec::new();
    assert!(save_versioned(&mut output, &sample_docs(), &metadata).is_err());
    assert!(output.is_empty());
    metadata = conditions();
    metadata.inputs.pop();
    assert!(save_versioned(&mut output, &sample_docs(), &metadata).is_err());
    // Also reject invalid metadata from a reader, rather than trusting the writer.
    let json = serde_json::to_vec(&metadata).unwrap();
    let mut bytes = Vec::new();
    for value in [MAGIC, 1, u32::try_from(json.len()).unwrap()] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(json);
    save(&mut bytes, &sample_docs()).unwrap();
    assert!(load(&mut Cursor::new(bytes)).is_err());
}

#[test]
fn compare_rejects_legacy_invalid_content_on_either_side() {
    let mut cases = vec![vec![vec![]], vec![vec![1.0], vec![1.0, 2.0]]];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        cases.push(vec![vec![value]]);
    }
    for chunks in cases {
        let invalid = vec![ChunkedEmbedding::try_new(chunks).unwrap()];
        for (expected, actual, side) in [
            (&invalid, &invalid, FixtureSide::Expected),
            (&sample_docs(), &invalid, FixtureSide::Actual),
        ] {
            assert!(
                matches!(compare(expected, actual), Err(CompareError::InvalidContent { side: got, doc: 0, .. }) if got == side)
            );
        }
        let mut bytes = Vec::new();
        assert!(
            save_versioned(
                &mut bytes,
                &invalid,
                &GenerationConditions {
                    inputs: vec!["public".into()],
                    ..conditions()
                }
            )
            .is_err()
        );
        assert!(bytes.is_empty());
    }
}

#[test]
fn compare_large_finite_vectors_does_not_hide_overflow() {
    let a = vec![ChunkedEmbedding::try_new(vec![vec![f32::MAX, f32::MAX]]).unwrap()];
    let equal = compare(&a, &a).unwrap();
    assert_eq!(equal.cosine_min, 1.0);
    assert_eq!(equal.max_abs_diff, 0.0);
    let orthogonal = vec![ChunkedEmbedding::try_new(vec![vec![f32::MAX, -f32::MAX]]).unwrap()];
    assert!(matches!(
        compare(&a, &orthogonal),
        Err(CompareError::NonFiniteMetric { doc: 0, chunk: 0 })
    ));
    let smaller = vec![ChunkedEmbedding::try_new(vec![vec![f32::MAX, 0.0]]).unwrap()];
    let diff = compare(&a, &smaller).unwrap();
    assert!((diff.cosine_min - 1.0 / 2.0f32.sqrt()).abs() < 1e-6);
    assert_eq!(diff.max_abs_diff, f32::MAX);
}

#[test]
fn committed_legacy_workloads_remain_readable() {
    for bytes in [
        include_bytes!("../../../tests/fixtures/phase2_baseline/w1.bin").as_slice(),
        include_bytes!("../../../tests/fixtures/phase2_baseline/w2.bin").as_slice(),
        include_bytes!("../../../tests/fixtures/phase2_baseline/w3.bin").as_slice(),
    ] {
        let fixture = load_fixture(&mut Cursor::new(bytes), DEFAULT_MAX_BYTES).unwrap();
        assert!(fixture.generation.is_none());
        assert!(!fixture.docs.is_empty());
        assert_eq!(
            compare(&fixture.docs, &fixture.docs).unwrap().max_abs_diff,
            0.0
        );
    }
}

#[test]
fn public_reproduction_fixtures_are_accepted_or_rejected_as_documented() {
    for bytes in [
        include_bytes!("../../../tests/fixtures/embedding_format/legacy.bin").as_slice(),
        include_bytes!("../../../tests/fixtures/embedding_format/v1.bin").as_slice(),
    ] {
        let fixture = load_fixture(&mut Cursor::new(bytes), DEFAULT_MAX_BYTES).unwrap();
        assert_eq!(fixture.docs[0].chunks()[0], [1.0, 0.0]);
    }
    for bytes in [
        include_bytes!("../../../tests/fixtures/embedding_format/hostile-dim.bin").as_slice(),
        include_bytes!("../../../tests/fixtures/embedding_format/nonfinite.bin").as_slice(),
        include_bytes!("../../../tests/fixtures/embedding_format/trailing.bin").as_slice(),
    ] {
        assert!(load(&mut Cursor::new(bytes)).is_err());
    }
}
