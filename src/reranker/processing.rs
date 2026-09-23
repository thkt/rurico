//! Pure pair truncation and score conversion used by the MLX backend.
use super::RerankerError;
use crate::model_io::truncate_with_eos;

/// Truncate pair tokens to `max_len`, setting the last token to EOS.
///
/// # Precondition
///
/// `max_len` must be ≥ 1. A zero `max_len` is a no-op (returns immediately).
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
