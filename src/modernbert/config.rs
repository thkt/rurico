use serde::Deserialize;

// Ensures `value * scale` fits in i32. Pass `scale = 1` for a plain `value <= i32::MAX` check.
fn validate_i32_bound(field: &str, value: usize, scale: i32) -> Result<(), String> {
    let max = (i32::MAX / scale) as usize;
    if value > max {
        return Err(format!("{field} must be <= {max}"));
    }
    Ok(())
}

/// ModernBERT config (config.json).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Vocabulary size (number of token embeddings).
    pub vocab_size: usize,
    /// Hidden dimension of each transformer layer.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// FFN intermediate dimension (before GLU split).
    pub intermediate_size: usize,
    /// Maximum input sequence length.
    pub max_position_embeddings: usize,
    /// Epsilon for layer normalization.
    pub layer_norm_eps: f64,
    /// Padding token ID.
    #[allow(dead_code)]
    pub pad_token_id: u32,
    /// Apply global (full-sequence) attention every N layers.
    pub global_attn_every_n_layers: usize,
    /// RoPE theta for global attention layers.
    pub global_rope_theta: f64,
    /// Sliding window size for local attention layers.
    pub local_attention: usize,
    /// RoPE theta for local attention layers.
    pub local_rope_theta: f64,
}

impl Config {
    /// Validate invariants (non-zero sizes, divisibility).
    ///
    /// # Errors
    ///
    /// Returns an opaque validation message if any required size is zero or
    /// `hidden_size` is not divisible by `num_attention_heads`. The exact
    /// message text is not part of the stable API contract.
    pub fn validate(&self) -> Result<(), String> {
        // `hidden_size * 3` is computed during Wqkv construction.
        validate_i32_bound("hidden_size", self.hidden_size, 3)?;
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".into());
        }
        validate_i32_bound("num_attention_heads", self.num_attention_heads, 1)?;
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".into());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.global_attn_every_n_layers == 0 {
            return Err("global_attn_every_n_layers must be > 0".into());
        }
        validate_i32_bound("vocab_size", self.vocab_size, 1)?;
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        // `intermediate_size * 2` is computed during Wi construction.
        validate_i32_bound("intermediate_size", self.intermediate_size, 2)?;
        validate_i32_bound("max_position_embeddings", self.max_position_embeddings, 1)?;
        if self.max_position_embeddings == 0 {
            return Err("max_position_embeddings must be > 0".into());
        }
        // `local_attention / 2` is cast to i32 when building the local-attention mask.
        let max_local = (i32::MAX as usize).saturating_mul(2).saturating_add(1);
        if self.local_attention > max_local {
            return Err(format!(
                "local_attention must be <= {max_local} (local_attention / 2 must fit in i32)"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
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
}
