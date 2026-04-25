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
pub fn gpu_pool_and_normalize(hidden: Array, mask: &Array) -> Result<Array, Exception> {
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
mod tests {
    //! `t_NNN_` prefix maps to Spec test scenarios in
    //! `.claude/workspace/planning/2026-04-24-phase3-gpu-pooling/spec.md`.

    use super::*;
    use mlx_rs::Array;
    use mlx_rs::error::Exception;

    // T-004 / FR-005 / AC-3
    //
    // Pure compile-time contract check: `gpu_pool_and_normalize` must consume
    // `hidden` by value so the drop-before-clear contract in
    // `src/mlx_cache.rs::release_inference_output` keeps its compile-time
    // guarantee. If someone relaxes the signature to `&Array`, the fn-pointer
    // coercion below fails to typecheck and this test refuses to compile —
    // which is the intended regression signal for AC-3.
    #[test]
    fn t_004_gpu_pool_signature_consumes_hidden_by_value() {
        let _coerce: fn(Array, &Array) -> Result<Array, Exception> = gpu_pool_and_normalize;
    }

    /// MLX runtime tests — construct `mlx_rs::Array` and call
    /// `gpu_pool_and_normalize`. Gated behind `test-mlx` so the default
    /// `cargo test` (including CI seatbelt) stays green.
    ///
    /// Run with `cargo test --features test-mlx -- --ignored` outside the
    /// Codex seatbelt. Mirrors the pattern in
    /// `src/modernbert/model.rs::mlx_runtime_tests`.
    #[cfg(feature = "test-mlx")]
    mod mlx_runtime_tests {
        use serial_test::serial;

        use super::*;
        use crate::sandbox::require_unsandboxed_mlx_runtime;

        /// Build a `[batch, seq, hidden]` f32 Array filled deterministically
        /// from a row-major iteration counter so a CPU reference can recompute
        /// the expected values without MLX.
        fn make_hidden(batch: i32, seq: i32, hidden: i32) -> Array {
            let total = (batch * seq * hidden) as usize;
            let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
            Array::from_slice(&data, &[batch, seq, hidden])
        }

        // T-001 / FR-001 / AC-3
        //
        // [T-001] Small known-shape pool: batch=2, seq=4, hidden=3 with mask
        // `[[1,1,0,0],[1,1,1,0]]`. Output must be shape [2, 3] and every row
        // must be unit-norm within `1e-6`.
        #[test]
        #[ignore = "requires unsandboxed MLX runtime"]
        #[serial]
        fn t_001_small_shape_pool_yields_unit_norm_rows() {
            require_unsandboxed_mlx_runtime();
            let hidden = make_hidden(2, 4, 3);
            let mask = Array::from_slice(&[1u32, 1, 0, 0, 1, 1, 1, 0], &[2, 4]);

            let pooled = gpu_pool_and_normalize(hidden, &mask).expect("pool ok");
            pooled.eval().expect("eval pool");

            assert_eq!(
                pooled.shape(),
                &[2, 3],
                "[T-001] output shape must be [2, 3]"
            );

            let data: &[f32] = pooled.as_slice();
            assert_eq!(data.len(), 6, "[T-001] flat length must be batch*hidden");
            for row in 0..2usize {
                let r = &data[row * 3..(row + 1) * 3];
                let norm_sq: f32 = r.iter().map(|v| v * v).sum();
                assert!(
                    (norm_sq - 1.0).abs() <= 1e-6,
                    "[T-001] row {row} L2 norm^2 must be 1.0 ± 1e-6, got {norm_sq} (row={r:?})"
                );
            }
        }

        // T-003 / FR-001 / AC-3
        //
        // [T-003] Large W1-class shape `[1, 8192, 768]`. Output shape must be
        // `[1, 768]` and every value finite.
        #[test]
        #[ignore = "requires unsandboxed MLX runtime"]
        #[serial]
        fn t_003_large_w1_shape_produces_finite_output() {
            require_unsandboxed_mlx_runtime();
            let batch = 1i32;
            let seq = 8192i32;
            let hidden = 768i32;
            let total = (batch * seq * hidden) as usize;
            // Seed-based reproducible synthetic input (LCG). Values in
            // roughly `[-1.0, 1.0]` so mask-weighted mean stays in-range.
            let data: Vec<f32> = {
                let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
                (0..total)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let u = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
                        u * 2.0 - 1.0
                    })
                    .collect()
            };
            let mask: Vec<u32> = vec![1u32; (batch * seq) as usize];

            let hidden_arr = Array::from_slice(&data, &[batch, seq, hidden]);
            let mask_arr = Array::from_slice(&mask, &[batch, seq]);

            let pooled = gpu_pool_and_normalize(hidden_arr, &mask_arr).expect("pool ok");
            pooled.eval().expect("eval pool");

            assert_eq!(
                pooled.shape(),
                &[batch, hidden],
                "[T-003] output shape must be [1, 768]"
            );
            let flat: &[f32] = pooled.as_slice();
            assert_eq!(flat.len(), 768, "[T-003] flat len must be hidden_size");
            assert!(
                flat.iter().all(|v| v.is_finite()),
                "[T-003] all output values must be finite"
            );
            // Spot-check unit norm so the test fails if L2 normalize is skipped.
            let norm_sq: f32 = flat.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq - 1.0).abs() <= 1e-5,
                "[T-003] pooled row must be unit-norm (norm^2={norm_sq})"
            );
        }

        // T-005 / FR-006 / AC-3
        //
        // [T-005] `u32` mask with values in `{0, 1}` must flow through the
        // internal `as_dtype(Float32)` cast without a dtype panic.
        #[test]
        #[ignore = "requires unsandboxed MLX runtime"]
        #[serial]
        fn t_005_u32_mask_cast_to_f32_succeeds() {
            require_unsandboxed_mlx_runtime();
            let hidden = make_hidden(1, 3, 2);
            let mask = Array::from_slice(&[1u32, 0, 1], &[1, 3]);

            let pooled = gpu_pool_and_normalize(hidden, &mask).expect("pool ok");
            pooled.eval().expect("eval pool");

            assert_eq!(pooled.shape(), &[1, 2], "[T-005] shape must be [1, 2]");
            let flat: &[f32] = pooled.as_slice();
            assert_eq!(flat.len(), 2, "[T-005] flat len must be batch*hidden");
            assert!(
                flat.iter().all(|v| v.is_finite()),
                "[T-005] output must be finite (got {flat:?})"
            );
        }

        // T-011 / FR-001a / AC-3
        //
        // [T-011] Upstream-guarantee regression signal. `gpu_pool_and_normalize`
        // intentionally has no internal all-zero-mask guard so the hot path
        // stays readback-free (ADR 0002 primary lever). Production callers go
        // through `ModernBert::forward::validate_attention_mask`, which rejects
        // fully-masked rows; `pool_output` then runs `is_finite` against the
        // already-readback flat buffer in `split_pooled`.
        //
        // This test feeds an all-zero mask row directly and asserts the
        // output contains NaN. If someone re-adds an in-function guard that
        // converts this to `Err` or a zero vector, the test fails, forcing
        // a revisit of ADR 0002 sub-decision 2 before the readback tax
        // silently returns to the production path.
        #[test]
        #[ignore = "requires unsandboxed MLX runtime"]
        #[serial]
        fn t_011_all_zero_mask_propagates_nan_without_internal_guard() {
            require_unsandboxed_mlx_runtime();
            let hidden = make_hidden(2, 3, 4);
            let mask = Array::from_slice(&[1u32, 1, 0, 0, 0, 0], &[2, 3]);

            let pooled = gpu_pool_and_normalize(hidden, &mask).expect("pool ok");
            pooled.eval().expect("eval pool");
            let flat: &[f32] = pooled.as_slice();

            assert!(
                flat.iter().any(|v| v.is_nan()),
                "[T-011] all-zero mask row must propagate NaN (no internal \
                 guard — upstream validate_attention_mask is the contract); \
                 got all-finite: {flat:?}"
            );
        }
    }
}
