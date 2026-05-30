use mlx_rs::{Array, Dtype, error::Exception, ops::maximum};

/// GPU-side mask-weighted mean pool + L2 unit-norm. Mask is cast to
/// `Float32` internally (FR-006). `hidden` is consumed by value to preserve
/// the drop-before-clear ordering in
/// `src/mlx_cache.rs::release_inference_output`.
///
/// No post-check on the pooled output: production callers go through
/// `ModernBert::forward::validate_attention_mask`, which rejects all-zero
/// rows — the only NaN source (`0/0` on a fully-masked row). Keeping the
/// function readback-free is the primary lever of ADR 0002 (`O(hidden)` per
/// batch entry instead of `O(seq × hidden)`); an internal `eval() +
/// as_slice() + is_finite scan` would defeat that on every forward. The
/// production caller `pool_output` calls `split_pooled` post-readback,
/// where `is_finite` runs against the already-readback flat buffer — so
/// non-finite outputs (corrupt weights, kernel overflow) still get caught
/// without a second readback.
///
/// # Errors
///
/// Returns an MLX [`Exception`] if any tensor operation fails.
// See ADR 0002 sub-decision 2 / NFR-005: `hidden` by-value enforces
// drop-before-clear at compile time. `clippy` can't see that cross-function
// intent, so suppress the lint here.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn gpu_pool_and_normalize(hidden: Array, mask: &Array) -> Result<Array, Exception> {
    let shape = mask.shape();
    let batch = shape[0];
    let seq = shape[1];

    let mask_f32 = mask.as_dtype(Dtype::Float32)?;
    let mask_expanded = mask_f32.reshape(&[batch, seq, 1])?;

    let masked = hidden.multiply(&mask_expanded)?;
    let sum_hidden = masked.sum_axes(&[1], false)?;
    let mask_sum = mask_f32.sum_axes(&[1], true)?;

    let pooled = sum_hidden.divide(&mask_sum)?;
    let norm_sq = pooled.multiply(&pooled)?.sum_axes(&[-1], true)?;
    let norm = norm_sq.sqrt()?;
    // Clamp the norm to `f32::MIN_POSITIVE` before division to leave a
    // zero pooled row at zero (0 / 1e-38 = 0); any real norm ≥
    // MIN_POSITIVE passes through unchanged. Readback-free.
    let eps = Array::from_slice(&[f32::MIN_POSITIVE], &[1]);
    let safe_norm = maximum(&norm, &eps)?;
    pooled.divide(&safe_norm)
}

#[cfg(test)]
mod tests;
