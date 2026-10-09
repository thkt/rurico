//! Synthetic pooling boundary checks; official numerical comparison remains #307.

/// Evaluated synthetic MLX boundary tests, gated by `test-mlx` and ignored.
/// Run outside the sandbox; these do not validate official model weights.
#[cfg(feature = "test-mlx")]
mod mlx_runtime_tests {
    use serial_test::serial;

    use super::super::gpu_pool_and_normalize;
    use crate::sandbox::require_unsandboxed_mlx_runtime;
    use mlx_rs::Array;

    /// Build a `[batch, seq, hidden]` f32 Array filled deterministically
    /// from a row-major iteration counter so a CPU reference can recompute
    /// the expected values without MLX.
    fn make_hidden(batch: i32, seq: i32, hidden: i32) -> Array {
        let total = (batch * seq * hidden) as usize;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        Array::from_slice(&data, &[batch, seq, hidden])
    }

    // Hand-calculated masked means: [-3,4,0], [2,-1,2], [0,0,0].
    // Asymmetric rows and masked sentinels expose ignored masks, wrong axes,
    // constant unit vectors and an unguarded zero norm. The mask is u32.
    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn masked_mean_matches_hand_calculated_vectors_and_zero_norm() {
        require_unsandboxed_mlx_runtime();
        let hidden = Array::from_slice(
            &[
                -4.0_f32, 2.0, 1.0, -2.0, 6.0, -1.0, 90.0, -70.0, 40.0, 20.0, 30.0, 80.0, 1.0,
                -3.0, 2.0, 50.0, 60.0, -90.0, 2.0, 0.0, 1.0, 3.0, 0.0, 3.0, 1.0, -2.0, 3.0, -1.0,
                2.0, -3.0, 80.0, 70.0, 60.0, 40.0, 30.0, 20.0,
            ],
            &[3, 4, 3],
        );
        let mask = Array::from_slice(&[1u32, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0], &[3, 4]);
        let pooled = gpu_pool_and_normalize(hidden, &mask).unwrap();
        pooled.eval().unwrap();
        assert_eq!(pooled.shape(), &[3, 3]);
        let expected = [
            -0.6_f32,
            0.8,
            0.0,
            2.0 / 3.0,
            -1.0 / 3.0,
            2.0 / 3.0,
            0.0,
            0.0,
            0.0,
        ];
        let actual: &[f32] = pooled.as_slice();
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && (a - e).abs() <= 1e-6,
                "element {i}: {a} != {e}"
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
    fn large_w1_shape_produces_finite_output() {
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
    fn all_zero_mask_propagates_nan_without_internal_guard() {
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
