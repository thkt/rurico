use super::{Artifacts, ModelInitError, RerankerError};
use crate::mlx_cache::release_inference_output;
use crate::model_io::{
    BUCKET_BOUNDS, MAX_SEQ_LEN, assign_bucket, compute_sub_batch_size, pad_sequences,
    truncate_with_eos,
};
use crate::modernbert::{Config, ModernBert, layer_norm_eps_f32};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::IndexOp,
};

use std::path::Path;

pub(super) struct RerankerInner {
    model: RerankerModel,
    tokenizer: tokenizers::Tokenizer,
}

impl RerankerInner {
    pub(super) fn new(artifacts: &Artifacts) -> Result<Self, ModelInitError> {
        let config = &artifacts.config;
        let tokenizer = artifacts.tokenizer.clone();

        let model =
            RerankerModel::load(&artifacts.paths.model, config).map_err(ModelInitError::backend)?;

        tracing::info!(
            hidden_size = config.hidden_size,
            model_path = ?artifacts.paths.model,
            "reranker: model loaded"
        );
        Ok(Self { model, tokenizer })
    }

    /// Score every `(query, document)` pair in `pairs`.
    ///
    /// Pairs are tokenised and bucketed by maximum length, then forwarded in
    /// sub-batches sized so each forward pass stays under the shared
    /// [`crate::model_io::TOKEN_BUDGET`] ceiling. Sub-batching mirrors the
    /// embed path so a large `pairs.len()` cannot allocate an unbounded
    /// `[N × bucket_len]` flat tensor.
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

        let raw_max = all_ids.iter().map(Vec::len).max().unwrap_or(0);
        let bucket_idx = assign_bucket(raw_max);
        let bucket_len = BUCKET_BOUNDS[bucket_idx];
        let sub_batch_size = compute_sub_batch_size(bucket_len);

        let total_pairs = pairs.len();
        let sub_batch_count = total_pairs.div_ceil(sub_batch_size);
        tracing::debug!(
            batch_size = total_pairs,
            sub_batch_count,
            sub_batch_size,
            bucket_len,
            "reranker score_batch dispatch",
        );

        let mut all_scores = Vec::with_capacity(total_pairs);
        for (ids_chunk, masks_chunk) in all_ids
            .chunks(sub_batch_size)
            .zip(all_masks.chunks(sub_batch_size))
        {
            let sub_scores = self.forward_sub_batch(ids_chunk, masks_chunk, bucket_len)?;
            all_scores.extend(sub_scores);
        }
        Ok(all_scores)
    }

    /// Forward one sub-batch and map logits to `[0, 1]` via sigmoid.
    fn forward_sub_batch(
        &mut self,
        ids_chunk: &[Vec<u32>],
        masks_chunk: &[Vec<u32>],
        bucket_len: usize,
    ) -> Result<Vec<f32>, RerankerError> {
        let (flat_ids, flat_mask, batch_size, max_len) =
            pad_sequences(ids_chunk, Some(masks_chunk), Some(bucket_len));

        let batch_size_i32 = i32::try_from(batch_size).expect("batch_size fits in i32");
        let max_len_i32 = i32::try_from(max_len).expect("max_len fits in i32");
        let output = self
            .model
            .forward(&flat_ids, &flat_mask, batch_size_i32, max_len_i32)
            .map_err(RerankerError::inference)?;

        let result = (|| -> Result<Vec<f32>, RerankerError> {
            output.eval().map_err(RerankerError::inference)?;
            let flat: &[f32] = output.as_slice();
            let flat_len = flat.len();
            if flat_len != batch_size {
                tracing::warn!(
                    expected = batch_size,
                    actual = flat_len,
                    batch_size,
                    bucket_len,
                    "score_batch: output shape mismatch"
                );
                return Err(RerankerError::inference(format!(
                    "score_batch: expected {batch_size} scores, got {flat_len}"
                )));
            }
            let scores: Vec<f32> = flat.iter().map(|&logit| sigmoid(logit)).collect();
            if scores.iter().any(|v| !v.is_finite()) {
                tracing::warn!(
                    batch_size,
                    bucket_len,
                    "score_batch: non-finite output detected (NaN or Inf in reranker scores)"
                );
                return Err(RerankerError::NonFiniteOutput);
            }
            Ok(scores)
        })();
        release_inference_output(output, "reranker");
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
    model: ModernBert,
    #[param]
    head: PredictionHead,
    #[param]
    classifier: nn::Linear,
}

impl RerankerModel {
    fn new(config: &Config) -> Result<Self, Exception> {
        let h = i32::try_from(config.hidden_size).expect("hidden_size fits in i32");
        let eps = layer_norm_eps_f32(config);

        let model = ModernBert::new(config)?;
        let dense = nn::LinearBuilder::new(h, h).build()?;
        let norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let classifier = nn::LinearBuilder::new(h, 1).build()?;

        Ok(Self {
            model,
            head: PredictionHead { dense, norm },
            classifier,
        })
    }

    fn load(path: &Path, config: &Config) -> Result<Self, Exception> {
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
    let orig_len = ids.len();
    if truncate_with_eos(ids, mask, max_len) {
        tracing::warn!(
            pair_idx,
            orig_len,
            max_len,
            "pair exceeds max_seq_len, truncating"
        );
    }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
