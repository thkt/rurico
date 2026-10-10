//! Pair preparation, batch orchestration, and score conversion.
use super::RerankerError;
use crate::model_io::truncate_with_eos;

/// Truncate pair tokens to `max_len`, setting the last token to EOS.
/// A zero `max_len` leaves the input unchanged.
pub(super) fn truncate_pair(
    ids: &mut Vec<u32>,
    mask: &mut Vec<u32>,
    max_len: usize,
    pair_idx: usize,
) {
    let orig_len = ids.len();
    if truncate_with_eos(ids, mask, max_len) {
        tracing::warn!(
            pair_idx,
            orig_len,
            max_len,
            "pair exceeds max_seq_len, truncating"
        );
    }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Validate the readback buffer before returning scores to callers.
pub(super) fn scores_from_logits(
    flat: &[f32],
    batch_size: usize,
    bucket_len: usize,
) -> Result<Vec<f32>, RerankerError> {
    let flat_len = flat.len();
    if flat_len != batch_size {
        tracing::warn!(
            expected = batch_size,
            actual = flat_len,
            batch_size,
            bucket_len,
            "score_batch: output shape mismatch"
        );
        return Err(RerankerError::inference_message(format!(
            "score_batch: expected {batch_size} scores, got {flat_len}"
        )));
    }
    // Sigmoid maps +/-Inf to finite endpoints, so validate the raw readback.
    if flat.iter().any(|v| !v.is_finite()) {
        tracing::warn!(
            batch_size,
            bucket_len,
            "score_batch: non-finite output detected (NaN or Inf in reranker logits)"
        );
        return Err(RerankerError::NonFiniteOutput);
    }
    Ok(flat.iter().map(|&logit| sigmoid(logit)).collect())
}

// Keep the public empty-input and pair assembly paths testable without model
// allocation. The injected boundary is the lock/inference call, not a copy of it.
pub(super) fn score_batch_with(
    pairs: &[(&str, &str)],
    score: impl FnOnce(&[(&str, &str)]) -> Result<Vec<f32>, RerankerError>,
) -> Result<Vec<f32>, RerankerError> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    score(pairs)
}

pub(super) fn rerank_with(
    query: &str,
    documents: &[&str],
    score: impl FnOnce(&[(&str, &str)]) -> Result<Vec<f32>, RerankerError>,
) -> Result<Vec<super::RankedResult>, RerankerError> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    let pairs: Vec<_> = documents.iter().map(|&doc| (query, doc)).collect();
    Ok(super::sort_results(&score(&pairs)?))
}

/// Plan the production dispatch and emit its per-call telemetry.
pub(super) fn dispatch_sub_batch_size(raw_max: usize, total_pairs: usize) -> (usize, usize) {
    use crate::model_io::{BUCKET_BOUNDS, assign_bucket, compute_sub_batch_size};
    let bucket_len = BUCKET_BOUNDS[assign_bucket(raw_max)];
    let sub_batch_size = compute_sub_batch_size(bucket_len, None);
    let sub_batch_count = total_pairs.div_ceil(sub_batch_size);
    tracing::debug!(
        batch_size = total_pairs,
        sub_batch_count,
        sub_batch_size,
        bucket_len,
        "reranker score_batch dispatch",
    );
    (bucket_len, sub_batch_size)
}
