use serde::Deserialize;

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
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".into());
        }
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
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.max_position_embeddings == 0 {
            return Err("max_position_embeddings must be > 0".into());
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

    #[test]
    fn config_validate_valid() {
        assert!(test_config().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_hidden_size() {
        let mut c = test_config();
        c.hidden_size = 0;
        assert!(c.validate().unwrap_err().contains("hidden_size"));
    }

    #[test]
    fn config_validate_zero_attention_heads() {
        let mut c = test_config();
        c.num_attention_heads = 0;
        assert!(c.validate().unwrap_err().contains("num_attention_heads"));
    }

    #[test]
    fn config_validate_indivisible_hidden_size() {
        let mut c = test_config();
        c.hidden_size = 100;
        c.num_attention_heads = 3;
        assert!(c.validate().unwrap_err().contains("divisible"));
    }

    #[test]
    fn config_validate_zero_global_attn() {
        let mut c = test_config();
        c.global_attn_every_n_layers = 0;
        assert!(
            c.validate()
                .unwrap_err()
                .contains("global_attn_every_n_layers")
        );
    }

    #[test]
    fn config_validate_zero_vocab_size() {
        let mut c = test_config();
        c.vocab_size = 0;
        assert!(c.validate().unwrap_err().contains("vocab_size"));
    }

    #[test]
    fn config_validate_zero_max_position_embeddings() {
        let mut c = test_config();
        c.max_position_embeddings = 0;
        assert!(
            c.validate()
                .unwrap_err()
                .contains("max_position_embeddings")
        );
    }
}
