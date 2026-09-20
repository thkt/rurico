use super::*;

pub fn test_config() -> Config {
    Config {
        vocab_size: 1000,
        hidden_size: 768,
        num_hidden_layers: 2,
        num_attention_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 512,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.0,
        local_attention: 128,
        local_rope_theta: 10000.0,
    }
}

fn assert_rejects<F: FnOnce(&mut Config)>(mutate: F, expected_substring: &str) {
    let mut c = test_config();
    mutate(&mut c);
    let err = c.validate().unwrap_err();
    assert!(
        err.contains(expected_substring),
        "expected error to contain {expected_substring:?}, got: {err}"
    );
}

#[test]
fn config_validate_valid() {
    assert!(test_config().validate().is_ok());
}

#[test]
fn config_validate_rope_bases_after_f32_conversion() {
    // Round-to-nearest ties: half the smallest subnormal becomes zero;
    // halfway from f32::MAX to 2^128 becomes infinity.
    let underflow = f64::from(f32::from_bits(1)) / 2.0;
    let overflow = f64::from(f32::MAX) + 2.0_f64.powi(103);
    let cases = [
        (0.0, false),
        (-0.0, false),
        (-10000.0, false),
        (f64::NAN, false),
        (f64::INFINITY, false),
        (f64::NEG_INFINITY, false),
        (underflow, false),
        (overflow, false),
        (underflow.next_up(), true),
        (f64::from(f32::from_bits(1)), true),
        (f64::from(f32::MIN_POSITIVE), true),
        (0.5, true),
        (1.0, true),
        (10000.0, true),
        (160000.0, true),
        (f64::from(f32::MAX), true),
        (overflow.next_down(), true),
    ];
    for field in ["global_rope_theta", "local_rope_theta"] {
        for (theta, accepted) in cases {
            let mut config = test_config();
            if field == "global_rope_theta" {
                config.global_rope_theta = theta;
            } else {
                config.local_rope_theta = theta;
            }
            let result = config.validate();
            assert_eq!(result.is_ok(), accepted, "{field}={theta:?}: {result:?}");
            if let Err(err) = result {
                assert!(err.contains(field), "{field}={theta:?}: {err}");
            }
        }
    }
}

#[test]
fn config_validate_rope_head_dimension() {
    // Small even dimensions remain valid; neither a multiple of four nor the
    // official models' dimension of 64 is required by MLX.
    for (head_dim, accepted) in [(1, false), (3, false), (2, true), (6, true), (64, true)] {
        let mut config = test_config();
        config.num_attention_heads = 3;
        config.hidden_size = head_dim * config.num_attention_heads;
        let result = config.validate();
        assert_eq!(result.is_ok(), accepted, "head_dim={head_dim}: {result:?}");
        if let Err(err) = result {
            assert!(err.contains("head dimension"), "{err}");
        }
    }
}

#[test]
fn config_validate_official_model_configs() {
    // Byte-for-byte config snapshots and pinned source revisions are documented
    // in tests/fixtures/modernbert_configs/README.md. No model or MLX execution.
    for (model, json) in [
        (
            "ruri-v3-30m",
            include_str!("../../../tests/fixtures/modernbert_configs/ruri-v3-30m.json"),
        ),
        (
            "ruri-v3-70m",
            include_str!("../../../tests/fixtures/modernbert_configs/ruri-v3-70m.json"),
        ),
        (
            "ruri-v3-130m",
            include_str!("../../../tests/fixtures/modernbert_configs/ruri-v3-130m.json"),
        ),
        (
            "ruri-v3-310m",
            include_str!("../../../tests/fixtures/modernbert_configs/ruri-v3-310m.json"),
        ),
        (
            "ruri-v3-reranker-310m",
            include_str!("../../../tests/fixtures/modernbert_configs/ruri-v3-reranker-310m.json"),
        ),
    ] {
        let config: Config =
            serde_json::from_str(json).unwrap_or_else(|err| panic!("{model}: {err}"));
        config
            .validate()
            .unwrap_or_else(|err| panic!("{model}: {err}"));
    }
}

#[test]
fn config_validate_zero_hidden_size() {
    assert_rejects(|c| c.hidden_size = 0, "hidden_size");
}

#[test]
fn config_validate_zero_attention_heads() {
    assert_rejects(|c| c.num_attention_heads = 0, "num_attention_heads");
}

#[test]
fn config_validate_indivisible_hidden_size() {
    assert_rejects(
        |c| {
            c.hidden_size = 100;
            c.num_attention_heads = 3;
        },
        "divisible",
    );
}

#[test]
fn config_validate_zero_global_attn() {
    assert_rejects(
        |c| c.global_attn_every_n_layers = 0,
        "global_attn_every_n_layers",
    );
}

#[test]
fn config_validate_zero_vocab_size() {
    assert_rejects(|c| c.vocab_size = 0, "vocab_size");
}

#[test]
fn config_validate_zero_max_position_embeddings() {
    assert_rejects(|c| c.max_position_embeddings = 0, "max_position_embeddings");
}

#[test]
fn config_validate_zero_num_hidden_layers() {
    assert_rejects(|c| c.num_hidden_layers = 0, "num_hidden_layers");
}

#[test]
fn config_validate_zero_intermediate_size() {
    assert_rejects(|c| c.intermediate_size = 0, "intermediate_size");
}

#[test]
fn config_validate_zero_local_attention() {
    assert_rejects(|c| c.local_attention = 0, "local_attention");
}

#[test]
fn config_validate_rejects_nan_layer_norm_eps() {
    assert_rejects(|c| c.layer_norm_eps = f64::NAN, "layer_norm_eps");
}

#[test]
fn config_validate_rejects_infinite_layer_norm_eps() {
    assert_rejects(|c| c.layer_norm_eps = f64::INFINITY, "layer_norm_eps");
}

#[test]
fn config_validate_rejects_zero_layer_norm_eps() {
    assert_rejects(|c| c.layer_norm_eps = 0.0, "layer_norm_eps");
}

#[test]
fn config_validate_rejects_negative_layer_norm_eps() {
    assert_rejects(|c| c.layer_norm_eps = -1e-5, "layer_norm_eps");
}

// 1e39 is finite in f64 but overflows to inf after the runtime f32 cast.
#[test]
fn config_validate_rejects_layer_norm_eps_overflowing_f32() {
    assert_rejects(|c| c.layer_norm_eps = 1e39, "layer_norm_eps");
}

// 1e-50 is positive in f64 but flushes to 0.0 after the runtime f32 cast.
#[test]
fn config_validate_rejects_layer_norm_eps_underflowing_f32() {
    assert_rejects(|c| c.layer_norm_eps = 1e-50, "layer_norm_eps");
}

#[test]
fn config_validate_rejects_hidden_size_above_i32_max_div_3() {
    assert_rejects(
        |c| c.hidden_size = (i32::MAX / 3) as usize + 1,
        "hidden_size",
    );
}

#[test]
fn config_validate_rejects_attention_heads_above_i32_max() {
    assert_rejects(
        |c| c.num_attention_heads = i32::MAX as usize + 1,
        "num_attention_heads",
    );
}

#[test]
fn config_validate_rejects_vocab_size_above_i32_max() {
    assert_rejects(|c| c.vocab_size = i32::MAX as usize + 1, "vocab_size");
}

#[test]
fn config_validate_rejects_intermediate_size_above_i32_max_div_2() {
    assert_rejects(
        |c| c.intermediate_size = (i32::MAX / 2) as usize + 1,
        "intermediate_size",
    );
}

#[test]
fn config_validate_rejects_max_position_embeddings_above_i32_max() {
    assert_rejects(
        |c| c.max_position_embeddings = i32::MAX as usize + 1,
        "max_position_embeddings",
    );
}

// `(i32::MAX as usize) * 2 + 2` overflows on 32-bit `usize` targets; guard accordingly.
#[cfg(target_pointer_width = "64")]
#[test]
fn config_validate_rejects_local_attention_above_bound() {
    assert_rejects(
        |c| c.local_attention = (i32::MAX as usize) * 2 + 2,
        "local_attention",
    );
}
