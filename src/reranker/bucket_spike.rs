//! Research candidate: fixed buckets, original indices, unchanged token budget.
//! Not selected by the public API. The injected boundary is a real sub-forward.
use crate::model_io::{BUCKET_BOUNDS, assign_bucket, compute_sub_batch_size};

#[derive(Debug, PartialEq)]
pub(super) enum DispatchError<E> {
    Forward(E),
    Shape { expected: usize, actual: usize },
}

/// Shared by the candidate's real-model probe and model-free order regression.
/// Retains all encoded rows; this is not a streaming or memory-bound planner.
pub(super) fn dispatch<T: Default + Clone, E>(
    lengths: &[usize],
    mut forward: impl FnMut(&[usize], usize) -> Result<Vec<T>, E>,
) -> Result<Vec<T>, DispatchError<E>> {
    let mut buckets: [Vec<usize>; 4] = Default::default();
    for (index, &length) in lengths.iter().enumerate() {
        buckets[assign_bucket(length)].push(index);
    }
    let mut result = vec![T::default(); lengths.len()];
    for (indices, bucket_len) in buckets.iter().zip(BUCKET_BOUNDS) {
        for chunk in indices.chunks(compute_sub_batch_size(bucket_len, None)) {
            let values = forward(chunk, bucket_len).map_err(DispatchError::Forward)?;
            if values.len() != chunk.len() {
                return Err(DispatchError::Shape {
                    expected: chunk.len(),
                    actual: values.len(),
                });
            }
            for (&index, value) in chunk.iter().zip(values) {
                result[index] = value;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reranker::sort_results;

    #[test]
    fn candidate_restores_distinct_rows_across_buckets_chunks_and_tails() {
        // 2001 short rows -> 2000 + 1; 32 long -> 31 + 1. Interleave
        // so concatenating bucket outputs cannot accidentally pass.
        let mut lengths = vec![128; 2033];
        for index in (0..64).step_by(2) {
            lengths[index] = 2049;
        }
        let mut shapes = Vec::new();
        let values = dispatch(&lengths, |indices, bucket| {
            shapes.push((indices.len(), bucket));
            Ok::<_, ()>(
                indices
                    .iter()
                    .map(|&index| (index + 100, if index % 2 == 0 { 0.5 } else { 0.9 }))
                    .collect(),
            )
        })
        .unwrap();
        assert_eq!(
            values.iter().map(|row| row.0).collect::<Vec<_>>(),
            (100..2133).collect::<Vec<_>>()
        );
        let scores: Vec<_> = values.iter().map(|row| row.1).collect();
        let ranks = sort_results(&scores);
        let expected: Vec<_> = (1..2033).step_by(2).chain((0..2033).step_by(2)).collect();
        assert_eq!(
            ranks.iter().map(|row| row.index).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(shapes, [(2000, 128), (1, 128), (31, 8192), (1, 8192)]);
    }

    #[test]
    fn candidate_preserves_equal_lengths_boundaries_and_empty_input() {
        for (lengths, expected) in [
            (vec![512; 501], vec![(500, 512), (1, 512)]),
            (
                vec![129, 127, 513, 128, 512, 2049, 2048, 8192, 2047, 511, 8191],
                vec![(2, 128), (3, 512), (3, 2048), (3, 8192)],
            ),
            (vec![], vec![]),
        ] {
            let mut shapes = Vec::new();
            let actual = dispatch(&lengths, |indices, bucket| {
                shapes.push((indices.len(), bucket));
                Ok::<_, ()>(indices.to_vec())
            })
            .unwrap();
            assert_eq!(actual, (0..lengths.len()).collect::<Vec<_>>());
            assert_eq!(shapes, expected);
        }
    }

    #[test]
    fn candidate_stops_on_forward_failure_or_wrong_row_count() {
        for too_many in [false, true] {
            let mut calls = 0;
            let error = dispatch(&[128, 129], |indices, _| {
                calls += 1;
                Ok::<_, ()>(vec![
                    0;
                    if too_many {
                        indices.len() + 1
                    } else {
                        indices.len() - 1
                    }
                ])
            })
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(
                error,
                DispatchError::Shape {
                    expected: 1,
                    actual: if too_many { 2 } else { 0 }
                }
            );
        }
        let mut calls = 0;
        let result = dispatch::<usize, _>(&[128, 129], |_, _| {
            calls += 1;
            Err("forward failed")
        });
        assert_eq!(result, Err(DispatchError::Forward("forward failed")));
        assert_eq!(calls, 1);
    }
}
