use super::{Artifacts, RerankerError, RerankerInitError};
use crate::model_io::{EOS_TOKEN_ID, MAX_SEQ_LEN};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::IndexOp,
};

pub(super) struct RerankerInner {
    model: RerankerModel,
    tokenizer: tokenizers::Tokenizer,
}

impl RerankerInner {
    pub(super) fn new(artifacts: &Artifacts) -> Result<Self, RerankerInitError> {
        let config = &artifacts.config;
        let tokenizer = artifacts.tokenizer.clone();

        let model = RerankerModel::load(&artifacts.paths.model, config)
            .map_err(RerankerInitError::backend)?;

        Ok(Self { model, tokenizer })
    }

    pub(super) fn score_batch(
        &mut self,
        pairs: &[(&str, &str)],
    ) -> Result<Vec<f32>, RerankerError> {
        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());
        let mut all_masks: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());

        for (pair_idx, &(query, doc)) in pairs.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode((query, doc), true)
                .map_err(|e| RerankerError::Tokenizer(e.to_string()))?;
            let mut ids = encoding.get_ids().to_vec();
            let mut mask = encoding.get_attention_mask().to_vec();
            truncate_pair(&mut ids, &mut mask, MAX_SEQ_LEN, pair_idx);
            all_ids.push(ids);
            all_masks.push(mask);
        }

        let (flat_ids, flat_mask, batch_size, max_len) =
            crate::model_io::pad_sequences(&all_ids, Some(&all_masks));

        let output = self
            .model
            .forward(&flat_ids, &flat_mask, batch_size as i32, max_len as i32)
            .map_err(RerankerError::inference)?;

        let result = (|| -> Result<Vec<f32>, RerankerError> {
            output.eval().map_err(RerankerError::inference)?;
            let flat: &[f32] = output.as_slice();
            let flat_len = flat.len();
            if flat_len != batch_size {
                return Err(RerankerError::inference(format!(
                    "score_batch: expected {batch_size} scores, got {flat_len}"
                )));
            }
            let scores: Vec<f32> = flat.iter().map(|&logit| sigmoid(logit)).collect();
            if scores.iter().any(|v| !v.is_finite()) {
                return Err(RerankerError::NonFiniteOutput);
            }
            Ok(scores)
        })();
        crate::mlx_cache::release_inference_output(output);
        result
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct PredictionHead {
    #[param]
    dense: nn::Linear,
    #[param]
    norm: nn::LayerNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
struct RerankerModel {
    #[param]
    model: crate::modernbert::ModernBert,
    #[param]
    head: PredictionHead,
    #[param]
    classifier: nn::Linear,
}

impl RerankerModel {
    fn new(config: &crate::modernbert::Config) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let eps = config.layer_norm_eps as f32;

        let model = crate::modernbert::ModernBert::new(config)?;
        let dense = nn::LinearBuilder::new(h, h).build()?;
        let norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let classifier = nn::LinearBuilder::new(h, 1).build()?;

        Ok(Self {
            model,
            head: PredictionHead { dense, norm },
            classifier,
        })
    }

    fn load(path: &std::path::Path, config: &crate::modernbert::Config) -> Result<Self, Exception> {
        let mut model = Self::new(config)?;
        model
            .load_safetensors(path)
            .map_err(|e| Exception::custom(format!("SafeTensors load error: {e}")))?;
        Ok(model)
    }

    fn forward(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[u32],
        batch_size: i32,
        seq_len: i32,
    ) -> Result<mlx_rs::Array, Exception> {
        let hidden = self
            .model
            .forward(input_ids, attention_mask, batch_size, seq_len)?;

        // CLS pooling: [batch, seq, hidden] -> [seq, batch, hidden] -> index(0) -> [batch, hidden]
        let cls = hidden.transpose_axes(&[1, 0, 2])?.index(0);

        let x = self.head.dense.forward(&cls)?;
        let x = nn::gelu(&x)?;
        let x = self.head.norm.forward(&x)?;

        self.classifier.forward(&x)
    }
}

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
    if max_len == 0 || ids.len() <= max_len {
        return;
    }
    log::warn!(
        "pair {pair_idx} exceeds max_seq_len ({} > {max_len}), truncating",
        ids.len()
    );
    ids.truncate(max_len);
    ids[max_len - 1] = EOS_TOKEN_ID;
    mask.truncate(max_len);
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
