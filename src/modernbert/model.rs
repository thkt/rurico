use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast::scaled_dot_product_attention,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::IndexOp,
};

use super::Config;

#[derive(Debug, Clone, ModuleParameters)]
#[allow(non_snake_case)]
struct Attention {
    #[param]
    Wqkv: nn::Linear,
    #[param]
    Wo: nn::Linear,
    #[param]
    rope: nn::Rope,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    uses_local_attention: bool,
}

impl Attention {
    fn new(
        config: &Config,
        rope_theta: f32,
        uses_local_attention: bool,
    ) -> Result<Self, Exception> {
        let h = config_dim_to_i32("hidden_size", config.hidden_size)?;
        let num_heads = config_dim_to_i32("num_attention_heads", config.num_attention_heads)?;
        let head_dim = h / num_heads;

        #[allow(non_snake_case)]
        let Wqkv = nn::LinearBuilder::new(h, h * 3).bias(false).build()?;
        #[allow(non_snake_case)]
        let Wo = nn::LinearBuilder::new(h, h).bias(false).build()?;
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(rope_theta)
            .build()?;

        Ok(Self {
            Wqkv,
            Wo,
            rope,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            uses_local_attention,
        })
    }

    fn forward(
        &mut self,
        xs: &Array,
        global_mask: &Array,
        local_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = xs.shape();
        let (b, seq_len) = (shape[0], shape[1]);

        let qkv = self.Wqkv.forward(xs)?;
        let qkv = qkv.reshape(&[b, seq_len, 3, self.num_heads, self.head_dim])?;
        let qkv = qkv.transpose_axes(&[2, 0, 3, 1, 4])?;

        let q = qkv.index(0);
        let k = qkv.index(1);
        let v = qkv.index(2);

        let rope_input_q = nn::RopeInputBuilder::new(&q).build()?;
        let rope_input_k = nn::RopeInputBuilder::new(&k).build()?;
        let q = self.rope.forward(rope_input_q)?;
        let k = self.rope.forward(rope_input_k)?;

        let mask = if self.uses_local_attention {
            if let Some(local) = local_mask {
                global_mask.add(local)?
            } else {
                global_mask.clone()
            }
        } else {
            global_mask.clone()
        };

        let output = scaled_dot_product_attention(q, &k, &v, self.scale, &mask)?;

        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[b, seq_len, -1])?;
        self.Wo.forward(&output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
#[allow(non_snake_case)]
struct Mlp {
    #[param]
    Wi: nn::Linear,
    #[param]
    Wo: nn::Linear,
}

impl Mlp {
    fn new(config: &Config) -> Result<Self, Exception> {
        let h = config_dim_to_i32("hidden_size", config.hidden_size)?;
        let inter = config_dim_to_i32("intermediate_size", config.intermediate_size)?;
        #[allow(non_snake_case)]
        let Wi = nn::LinearBuilder::new(h, inter * 2).bias(false).build()?;
        #[allow(non_snake_case)]
        let Wo = nn::LinearBuilder::new(inter, h).bias(false).build()?;
        Ok(Self { Wi, Wo })
    }

    fn forward(&mut self, xs: &Array) -> Result<Array, Exception> {
        let xs = self.Wi.forward(xs)?;
        let chunks = xs.split(2, Some(-1))?;
        let gate = nn::gelu(&chunks[0])?;
        let xs = gate.multiply(&chunks[1])?;
        self.Wo.forward(&xs)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct TransformerLayer {
    #[param]
    attn_norm: nn::LayerNorm,
    #[param]
    attn: Attention,
    #[param]
    mlp_norm: nn::LayerNorm,
    #[param]
    mlp: Mlp,
    uses_attn_norm: bool,
}

impl TransformerLayer {
    fn new(config: &Config, layer_id: usize) -> Result<Self, Exception> {
        let uses_local = !layer_id.is_multiple_of(config.global_attn_every_n_layers);
        let rope_theta = rope_theta_f32(config, uses_local);
        let h = config_dim_to_i32("hidden_size", config.hidden_size)?;
        let eps = layer_norm_eps_f32(config);

        let attn_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let attn = Attention::new(config, rope_theta, uses_local)?;
        let mlp_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let mlp = Mlp::new(config)?;

        Ok(Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            uses_attn_norm: layer_id != 0,
        })
    }

    fn forward(
        &mut self,
        xs: &Array,
        global_mask: &Array,
        local_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = xs.clone();

        let normed = if self.uses_attn_norm {
            self.attn_norm.forward(xs)?
        } else {
            xs.clone()
        };

        let attn_out = self.attn.forward(&normed, global_mask, local_mask)?;
        let xs = attn_out.add(&residual)?;

        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&xs)?)?;
        xs.add(&mlp_out)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct Embeddings {
    #[param]
    tok_embeddings: nn::Embedding,
    #[param]
    norm: nn::LayerNorm,
}

/// ModernBERT backbone (mlx-rs). Ported from cl-nagoya/ruri-v3-310m.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ModernBert {
    #[param]
    embeddings: Embeddings,
    #[param]
    layers: Vec<TransformerLayer>,
    #[param]
    final_norm: nn::LayerNorm,
    local_attention_half: usize,
    max_seq_len: usize,
    /// Per-seq-len cache keyed on the four [`BUCKET_BOUNDS`] ceilings. Every
    /// `forward()` caller (chunk encoder, query encoder, reranker pair scorer)
    /// rounds its actual `seq_len` up to a bucket bound before calling, so this
    /// map holds at most four entries for the lifetime of the model.
    ///
    /// [`BUCKET_BOUNDS`]: crate::model_io::BUCKET_BOUNDS
    local_mask_cache: HashMap<i32, Array>,
}

impl ModernBert {
    /// Construct a model with randomly initialized weights from `config`.
    ///
    /// Use [`ModernBert::load`] to load pre-trained weights from a SafeTensors file.
    ///
    /// # Errors
    ///
    /// Returns an MLX [`Exception`] if `config` is invalid or any layer
    /// allocation fails. Exception messages are backend-generated and should be
    /// treated as opaque.
    pub fn new(config: &Config) -> Result<Self, Exception> {
        config
            .validate()
            .map_err(|e| Exception::custom(format!("invalid config: {e}")))?;
        let h = config_dim_to_i32("hidden_size", config.hidden_size)?;
        let eps = layer_norm_eps_f32(config);
        let vocab_size = config_dim_to_i32("vocab_size", config.vocab_size)?;

        let tok_embeddings = nn::Embedding::new(vocab_size, h)?;
        let emb_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| TransformerLayer::new(config, i))
            .collect::<Result<Vec<_>, _>>()?;

        let final_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;

        Ok(Self {
            embeddings: Embeddings {
                tok_embeddings,
                norm: emb_norm,
            },
            layers,
            final_norm,
            local_attention_half: config.local_attention / 2,
            max_seq_len: config.max_position_embeddings,
            local_mask_cache: HashMap::new(),
        })
    }

    /// Load a pre-trained model from a SafeTensors file.
    ///
    /// # Errors
    ///
    /// Returns an MLX [`Exception`] if the model file does not exist, `config`
    /// is invalid, or SafeTensors loading fails. Exception messages are
    /// backend-generated and should be treated as opaque.
    pub fn load(path: impl AsRef<Path>, config: &Config) -> Result<Self, Exception> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Exception::custom(format!(
                "model file not found: {}",
                path.display()
            )));
        }
        let mut model = Self::new(config)?;
        model
            .load_safetensors(path)
            .map_err(|e| Exception::custom(format!("SafeTensors load error: {e}")))?;
        Ok(model)
    }

    /// Run a forward pass, returning final hidden states `[batch_size, seq_len, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Returns an MLX [`Exception`] if:
    /// - `batch_size <= 0`
    /// - `seq_len < 0`
    /// - `input_ids.len()` or `attention_mask.len()` does not equal `batch_size * seq_len`
    /// - any `attention_mask` row contains values other than `0` or `1`
    /// - any `attention_mask` row is fully masked
    /// - MLX tensor construction or attention execution fails
    ///
    /// When `seq_len` exceeds `config.max_position_embeddings`, the input is
    /// truncated as a defense-in-depth fallback; `Ok` is returned with a shorter
    /// output tensor and a `tracing::warn!` warning is emitted.
    ///
    /// Exception messages are backend-generated and should be treated as opaque.
    pub fn forward(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[u32],
        batch_size: i32,
        seq_len: i32,
    ) -> Result<Array, Exception> {
        if batch_size <= 0 {
            return Err(Exception::custom("batch_size must be positive"));
        }
        if seq_len < 0 {
            return Err(Exception::custom("seq_len must be non-negative"));
        }
        let expected_len = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| Exception::custom("batch_size * seq_len overflows i32"))?
            as usize;
        let assert_len = |name: &str, slice: &[u32]| {
            if slice.len() != expected_len {
                return Err(Exception::custom(format!(
                    "{name} length {} != batch_size ({batch_size}) * seq_len ({seq_len})",
                    slice.len()
                )));
            }
            Ok(())
        };
        assert_len("input_ids", input_ids)?;
        assert_len("attention_mask", attention_mask)?;

        let max_seq = config_dim_to_i32("max_seq_len", self.max_seq_len)?;
        if seq_len > max_seq {
            // Defense-in-depth: truncate oversize inputs, validating only the effective prefix.
            tracing::warn!(
                seq_len,
                max_seq,
                batch_size,
                "model.forward: seq_len exceeds max_seq_len, truncating as defense-in-depth fallback"
            );
            let bs = batch_size as usize;
            let old_len = seq_len as usize;
            let new_len = max_seq as usize;
            let mut ids = Vec::with_capacity(bs * new_len);
            let mut mask = Vec::with_capacity(bs * new_len);
            for row_idx in 0..bs {
                let id_start = row_idx * old_len;
                ids.extend_from_slice(&input_ids[id_start..id_start + new_len]);
                let prefix = &attention_mask[row_idx * old_len..row_idx * old_len + new_len];
                mask.extend_from_slice(prefix);
            }
            validate_attention_mask(&mask, new_len)?;
            return self.run_forward(&ids, &mask, batch_size, max_seq);
        }

        validate_attention_mask(attention_mask, seq_len as usize)?;
        self.run_forward(input_ids, attention_mask, batch_size, seq_len)
    }

    fn run_forward(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[u32],
        batch_size: i32,
        seq_len: i32,
    ) -> Result<Array, Exception> {
        let ids = Array::from_slice(input_ids, &[batch_size, seq_len]);
        let mask = Array::from_slice(attention_mask, &[batch_size, seq_len]);

        let mut xs = self.embeddings.tok_embeddings.forward(&ids)?;
        xs = self.embeddings.norm.forward(&xs)?;

        let global_mask = prepare_4d_attention_mask(&mask, batch_size, seq_len)?;
        let local_mask = if let Some(cached) = self.local_mask_cache.get(&seq_len) {
            cached.clone()
        } else {
            let half = config_dim_to_i32("local_attention_half", self.local_attention_half)?;
            let m = get_local_attention_mask(seq_len, half)?;
            self.local_mask_cache.insert(seq_len, m.clone());
            m
        };

        for layer in &mut self.layers {
            xs = layer.forward(&xs, &global_mask, Some(&local_mask))?;
        }

        self.final_norm.forward(&xs)
    }
}

// Use a large negative instead of -inf to avoid 0.0 * (-inf) = NaN on Metal kernels
// for certain sequence lengths (threshold ~9 tokens on Apple Silicon).
const MASK_FILL: f32 = -1e9;

/// Cast a `usize` config-derived dimension to `i32`, returning a structured
/// error when it exceeds `i32::MAX`. [`Config::validate`] already enforces the
/// bound for every field passed here; the `Err` arm is defense-in-depth against
/// a regression that constructs a `Config` without going through `validate`.
fn config_dim_to_i32(name: &str, value: usize) -> Result<i32, Exception> {
    i32::try_from(value).map_err(|_| {
        Exception::custom(format!(
            "{name} ({value}) exceeds i32::MAX (config invariant violation)"
        ))
    })
}

// Model config constant; layer norm epsilon fits in f32.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn layer_norm_eps_f32(config: &Config) -> f32 {
    config.layer_norm_eps as f32
}

// Model config constants from config.json; RoPE base values fit in f32.
#[allow(clippy::cast_possible_truncation)]
fn rope_theta_f32(config: &Config, uses_local: bool) -> f32 {
    if uses_local {
        config.local_rope_theta as f32
    } else {
        config.global_rope_theta as f32
    }
}

fn prepare_4d_attention_mask(
    mask: &Array,
    batch_size: i32,
    seq_len: i32,
) -> Result<Array, Exception> {
    let expanded = mask
        .reshape(&[batch_size, 1, 1, seq_len])?
        .as_dtype(mlx_rs::Dtype::Float32)?;
    let ones = Array::from_f32(1.0);
    let fill = Array::from_f32(MASK_FILL);
    let inverted = ones.subtract(&expanded)?;
    inverted.multiply(fill)
}

fn validate_attention_mask(mask: &[u32], seq_len: usize) -> Result<(), Exception> {
    if seq_len == 0 {
        return Ok(());
    }
    for (row_idx, row) in mask.chunks(seq_len).enumerate() {
        let mut has_any_one = false;
        for &m in row {
            if m > 1 {
                return Err(Exception::custom(format!(
                    "attention_mask row {row_idx} contains value other than 0 or 1"
                )));
            }
            if m == 1 {
                has_any_one = true;
            }
        }
        if !has_any_one {
            return Err(Exception::custom(format!(
                "attention_mask row {row_idx} is fully masked"
            )));
        }
    }
    Ok(())
}

fn get_local_attention_mask(seq_len: i32, half_window: i32) -> Result<Array, Exception> {
    let n = usize::try_from(seq_len)
        .map_err(|_| Exception::custom(format!("seq_len ({seq_len}) must be non-negative")))?;
    let hw = usize::try_from(half_window).map_err(|_| {
        Exception::custom(format!("half_window ({half_window}) must be non-negative"))
    })?;
    // `n * n` allocation is bounded by callers (`run_forward` passes `seq_len <=
    // max_position_embeddings`); `checked_mul` is defense-in-depth against a
    // regression that bypasses that gate.
    let n_squared = n
        .checked_mul(n)
        .ok_or_else(|| Exception::custom(format!("seq_len ({n}) squared overflows usize")))?;
    // NEG_INFINITY is safe here: values are written directly, never multiplied by a mask array,
    // so the 0.0 * (-inf) = NaN Metal hazard (see MASK_FILL) does not apply.
    let mut data = vec![f32::NEG_INFINITY; n_squared];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw + 1).min(n);
        for j in lo..hi {
            data[i * n + j] = 0.0;
        }
    }
    Ok(Array::from_slice(&data, &[seq_len, seq_len]))
}

#[cfg(test)]
mod tests;
