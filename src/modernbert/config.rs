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
    /// Validate invariants (non-zero sizes, divisibility, finite positive epsilon).
    ///
    /// # Errors
    ///
    /// Returns an opaque validation message if any required size is zero,
    /// `hidden_size` is not divisible by `num_attention_heads`, or
    /// `layer_norm_eps` is not finite and positive. The exact message text
    /// is not part of the stable API contract.
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
        if self.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be > 0".into());
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
        if self.intermediate_size == 0 {
            return Err("intermediate_size must be > 0".into());
        }
        validate_i32_bound("max_position_embeddings", self.max_position_embeddings, 1)?;
        if self.max_position_embeddings == 0 {
            return Err("max_position_embeddings must be > 0".into());
        }
        // `local_attention / 2` is cast to i32 when building the local-attention mask.
        // Compute the bound in u64 so the check stays correct on 32-bit `usize`
        // targets, where `(i32::MAX as usize).saturating_mul(2)` would clamp to
        // `usize::MAX` and silently widen the accepted range.
        let max_local: u64 = (i32::MAX as u64) * 2 + 1;
        if self.local_attention as u64 > max_local {
            return Err(format!(
                "local_attention must be <= {max_local} (local_attention / 2 must fit in i32)"
            ));
        }
        if self.local_attention == 0 {
            return Err("local_attention must be > 0".into());
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err("layer_norm_eps must be finite and > 0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests;
