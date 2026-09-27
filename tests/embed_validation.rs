//! External-consumer compatibility and validation contract.
use rurico::embed::{ChunkedEmbedding, EmbedError, EmbeddingValidationError, EmptyChunksError};

#[test]
fn legacy_constructor_and_error_remain_source_compatible() {
    let result: Result<ChunkedEmbedding, EmptyChunksError> = ChunkedEmbedding::try_new(vec![]);
    let EmptyChunksError = result.unwrap_err();
    let runtime: EmbedError = EmptyChunksError.into();
    assert!(matches!(runtime, EmbedError::EmptyChunks(EmptyChunksError)));
    let legacy = ChunkedEmbedding::try_new(vec![vec![0.0], vec![1.0]]).unwrap();
    assert_eq!(legacy.chunk_ids(), ["c0", "c1"]);
    // The legacy entry point deliberately does not validate vector content.
    assert!(ChunkedEmbedding::try_new(vec![vec![], vec![f32::NAN]]).is_ok());
}

#[test]
fn validated_constructor_reports_kind_and_position() {
    use EmbeddingValidationError as E;
    let cases = [
        (vec![], E::EmptyChunks(EmptyChunksError)),
        (vec![vec![]], E::EmptyVector { chunk: 0 }),
        (vec![vec![1.0], vec![]], E::EmptyVector { chunk: 1 }),
        (
            vec![vec![1.0, 2.0], vec![3.0]],
            E::DimensionMismatch {
                chunk: 1,
                expected: 2,
                actual: 1,
            },
        ),
        (
            vec![vec![1.0], vec![2.0, 3.0]],
            E::DimensionMismatch {
                chunk: 1,
                expected: 1,
                actual: 2,
            },
        ),
    ];
    for (chunks, expected) in cases {
        assert_eq!(
            ChunkedEmbedding::try_new_validated(chunks).unwrap_err(),
            expected
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            ChunkedEmbedding::try_new_validated(vec![vec![1.0, 2.0], vec![3.0, value]])
                .unwrap_err(),
            E::NonFiniteValue {
                chunk: 1,
                element: 1
            },
        );
    }
}

#[test]
fn validated_constructor_preserves_values_order_and_ids_without_normalizing() {
    let chunks = vec![vec![0.0, -0.0, 0.0], vec![-2.0, 3.0, f32::MAX]];
    let result: Result<ChunkedEmbedding, EmbeddingValidationError> =
        ChunkedEmbedding::try_new_validated(chunks.clone());
    let embedding = result.unwrap();
    assert_eq!(embedding.chunk_ids(), ["c0", "c1"]);
    assert_eq!(embedding.chunks(), chunks);
    for (actual, expected) in embedding
        .chunks()
        .iter()
        .flatten()
        .zip(chunks.iter().flatten())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
