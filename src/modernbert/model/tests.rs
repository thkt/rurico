use super::*;
use crate::modernbert::config::tests::test_config;

#[test]
fn load_nonexistent_path_errors() {
    let config = test_config();
    let result = ModernBert::load("/nonexistent/model.safetensors", &config);
    assert!(result.is_err());
}

fn assert_new_rejects_without_panicking(config: &Config, expected_field: &str) {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let outcome = catch_unwind(AssertUnwindSafe(|| ModernBert::new(config)));
    assert!(
        outcome.is_ok(),
        "oversized config should return Err, not panic"
    );

    let err = outcome
        .unwrap()
        .expect_err("oversized config should be rejected");
    let expected = format!("invalid config: {expected_field}");
    assert!(
        err.what().contains(&expected),
        "expected error to contain {expected:?}, got: {}",
        err.what()
    );
}

#[test]
fn new_rejects_hidden_size_above_bound_without_panicking() {
    let mut config = test_config();
    // Just above `i32::MAX / 3`; would overflow `h * 3` in Wqkv if not rejected.
    config.hidden_size = (i32::MAX / 3) as usize + 1;
    assert_new_rejects_without_panicking(&config, "hidden_size");
}

#[test]
fn new_rejects_intermediate_size_above_bound_without_panicking() {
    let mut config = test_config();
    // Just above `i32::MAX / 2`; would overflow `inter * 2` in Wi if not rejected.
    config.intermediate_size = (i32::MAX / 2) as usize + 1;
    assert_new_rejects_without_panicking(&config, "intermediate_size");
}

#[test]
fn validate_mask_rejects_fully_masked_row() {
    let result = validate_attention_mask(&[0, 0, 0], 3);
    assert!(result.is_err(), "fully masked row should return Err");
    assert!(
        result.unwrap_err().what().contains("fully masked"),
        "error message should mention 'fully masked'"
    );
}

#[test]
fn validate_mask_rejects_fully_masked_row_in_batch() {
    // batch of 2: row 0 valid, row 1 fully masked
    let result = validate_attention_mask(&[1, 1, 1, 0, 0, 0], 3);
    assert!(
        result.is_err(),
        "batch with a fully masked row should return Err"
    );
}

#[test]
fn validate_mask_rejects_invalid_value() {
    let result = validate_attention_mask(&[1, 2, 1], 3);
    assert!(result.is_err(), "mask value > 1 should return Err");
    assert!(
        result.unwrap_err().what().contains("other than 0 or 1"),
        "error message should mention 'other than 0 or 1'"
    );
}

#[test]
fn validate_mask_accepts_valid_input() {
    assert!(validate_attention_mask(&[1, 1, 0], 3).is_ok());
    assert!(validate_attention_mask(&[], 0).is_ok());
    // all-ones single row
    assert!(validate_attention_mask(&[1, 1, 1], 3).is_ok());
    // multi-row valid batch: row 0 = [1,1,0], row 1 = [1,0,1]
    assert!(validate_attention_mask(&[1, 1, 0, 1, 0, 1], 3).is_ok());
}

// Defense-in-depth: a negative `seq_len` previously produced a giant
// `usize` via `as` cast, allocating gigabytes before `Array::from_slice`
// ever ran. `usize::try_from` now rejects it as a structured Exception.
#[test]
fn get_local_attention_mask_rejects_negative_seq_len() {
    let err = get_local_attention_mask(-1, 1).expect_err("negative seq_len must error");
    assert!(
        err.what().contains("seq_len"),
        "error message should reference seq_len, got: {}",
        err.what()
    );
}

#[test]
fn get_local_attention_mask_rejects_negative_half_window() {
    let err = get_local_attention_mask(8, -1).expect_err("negative half_window must error");
    assert!(
        err.what().contains("half_window"),
        "error message should reference half_window, got: {}",
        err.what()
    );
}

// Defense-in-depth: `config_dim_to_i32` Err path is unreachable after a
// valid `Config::validate()` since `validate` already enforces the i32
// bound. The test exercises the helper directly so a regression that
// skips `validate` cannot panic in a hot forward.
#[test]
fn config_dim_to_i32_rejects_value_above_i32_max() {
    let err = config_dim_to_i32("hidden_size", (i32::MAX as usize) + 1)
        .expect_err("value above i32::MAX must error");
    assert!(
        err.what().contains("hidden_size"),
        "error message should reference field name, got: {}",
        err.what()
    );
}

#[test]
fn config_dim_to_i32_accepts_value_at_i32_max() {
    assert_eq!(
        config_dim_to_i32("hidden_size", i32::MAX as usize).unwrap(),
        i32::MAX
    );
}

/// MLX runtime tests — may abort due to foreign exceptions from mlx-rs FFI.
/// Run with `cargo test --features test-mlx -- --ignored` outside Codex seatbelt.
#[cfg(feature = "test-mlx")]
mod mlx_runtime_tests {
    use serial_test::serial;

    use super::*;
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_produces_correct_shape() {
        require_unsandboxed_mlx_runtime();
        let config = test_config();
        let mut model = ModernBert::new(&config).expect("create model");

        let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
        let mask: Vec<u32> = vec![1, 1, 1, 1, 1];

        let output = model.forward(&input_ids, &mask, 1, 5).expect("forward");
        assert_eq!(output.shape(), &[1, 5, 768]);
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_different_seq_lengths() {
        require_unsandboxed_mlx_runtime();
        let config = test_config();
        let mut model = ModernBert::new(&config).expect("create model");

        let output = model
            .forward(&[1, 2, 3], &[1, 1, 1], 1, 3)
            .expect("forward");
        assert_eq!(output.shape(), &[1, 3, 768]);
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_with_padding_mask() {
        require_unsandboxed_mlx_runtime();
        let config = test_config();
        let mut model = ModernBert::new(&config).expect("create model");

        let input_ids: Vec<u32> = vec![1, 2, 3, 0, 0];
        let mask: Vec<u32> = vec![1, 1, 1, 0, 0];

        let output = model.forward(&input_ids, &mask, 1, 5).expect("forward");
        assert_eq!(output.shape(), &[1, 5, 768]);
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn global_mask_values() {
        require_unsandboxed_mlx_runtime();
        let mask = Array::from_slice(&[1u32, 1, 0], &[1, 3]);
        let result = prepare_4d_attention_mask(&mask, 1, 3).expect("mask");
        result.eval().unwrap();

        assert_eq!(result.shape(), &[1, 1, 1, 3]);
        let data: &[f32] = result.as_slice();
        assert_eq!(data[0], 0.0, "unmasked should be 0.0");
        assert_eq!(data[1], 0.0, "unmasked should be 0.0");
        assert!(
            data[2] < 0.0 && data[2].is_finite(),
            "masked should be finite negative (not NaN or -inf), got {}",
            data[2]
        );
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn global_mask_all_ones() {
        require_unsandboxed_mlx_runtime();
        let mask = Array::from_slice(&[1u32, 1, 1, 1], &[1, 4]);
        let result = prepare_4d_attention_mask(&mask, 1, 4).expect("mask");
        result.eval().unwrap();

        let data: &[f32] = result.as_slice();
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, 0.0, "all-ones mask should produce 0.0 at index {i}");
        }
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn local_mask_window() {
        require_unsandboxed_mlx_runtime();
        let result = get_local_attention_mask(5, 1).expect("local mask");
        result.eval().unwrap();

        assert_eq!(result.shape(), &[5, 5]);
        let data: &[f32] = result.as_slice();

        for i in 0..5usize {
            for j in 0..5usize {
                let val = data[i * 5 + j];
                let dist = (i as isize - j as isize).unsigned_abs();
                if dist <= 1 {
                    assert_eq!(val, 0.0, "({i},{j}) within window should be 0.0");
                } else {
                    assert!(
                        val.is_infinite() && val.is_sign_negative(),
                        "({i},{j}) outside window should be -inf, got {val}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn local_mask_zero_window() {
        require_unsandboxed_mlx_runtime();
        let result = get_local_attention_mask(3, 0).expect("local mask");
        result.eval().unwrap();

        let data: &[f32] = result.as_slice();
        for i in 0..3usize {
            for j in 0..3usize {
                let val = data[i * 3 + j];
                if i == j {
                    assert_eq!(val, 0.0, "diagonal ({i},{j}) should be 0.0");
                } else {
                    assert!(val.is_infinite() && val.is_sign_negative());
                }
            }
        }
    }

    // T-010: forward_truncates_oversize_input
    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_truncates_oversize_input() {
        require_unsandboxed_mlx_runtime();
        // [T-010] seq_len > max_seq_len → truncate + warn, not error
        let config = test_config(); // max_position_embeddings = 512
        let mut model = ModernBert::new(&config).expect("create model");

        let oversize = config.max_position_embeddings + 100; // 612
        let input_ids = vec![1u32; oversize];
        let mask = vec![1u32; oversize];

        let result = model.forward(
            &input_ids,
            &mask,
            1,
            i32::try_from(oversize).expect("bounded by model config"),
        );
        assert!(
            result.is_ok(),
            "forward should truncate oversize input, not error: {result:?}"
        );
        let output = result.unwrap();
        assert_eq!(
            output.shape(),
            &[
                1,
                i32::try_from(config.max_position_embeddings).expect("bounded by model config"),
                i32::try_from(config.hidden_size).expect("bounded by model config"),
            ]
        );
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_rejects_oversize_seq_with_short_buffer() {
        require_unsandboxed_mlx_runtime();
        let config = test_config(); // max_position_embeddings = 512
        let mut model = ModernBert::new(&config).expect("create model");

        let oversize_seq =
            i32::try_from(config.max_position_embeddings + 100).expect("bounded by model config");
        let short_buf = vec![1u32; 5]; // much shorter than batch_size * seq_len
        let short_mask = vec![1u32; 5];

        let result = model.forward(&short_buf, &short_mask, 1, oversize_seq);
        assert!(
            result.is_err(),
            "should return Err for short buffer, not panic"
        );
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_rejects_zero_batch_size() {
        require_unsandboxed_mlx_runtime();
        let config = test_config();
        let mut model = ModernBert::new(&config).expect("create model");

        let result = model.forward(&[], &[], 0, 5);
        assert!(result.is_err(), "batch_size=0 should return Err");
        assert!(
            result
                .unwrap_err()
                .what()
                .contains("batch_size must be positive"),
            "error message should mention 'batch_size must be positive'"
        );
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn forward_rejects_negative_batch_size() {
        require_unsandboxed_mlx_runtime();
        let config = test_config();
        let mut model = ModernBert::new(&config).expect("create model");

        let result = model.forward(&[1, 2, 3], &[1, 1, 1], -1, 3);
        assert!(result.is_err(), "negative batch_size should return Err");
    }
}
