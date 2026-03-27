use super::{
    DOCUMENT_PREFIX, EmbedError, ModelPaths, load_tokenizer, postprocess_embedding, read_config,
    sort_indices_by_len, tokenize_with_prefix,
};

struct PaddedBatch {
    sorted_indices: Vec<usize>,
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    max_seq_len: usize,
}

pub(super) struct EmbedderInner {
    model: crate::modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
}

impl EmbedderInner {
    pub(super) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;

        let config: crate::modernbert::Config = read_config(&paths.config)?;

        let model = crate::modernbert::ModernBert::load(&paths.model, &config)
            .map_err(EmbedError::inference)?;

        let tokenizer = load_tokenizer(&paths.tokenizer)?;

        Ok(Self { model, tokenizer })
    }

    pub(super) fn embed_with_prefix(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;

        let output = self
            .model
            .forward(&tok.input_ids, &tok.attention_mask, 1, tok.seq_len as i32)
            .map_err(EmbedError::inference)?;

        output.eval().map_err(EmbedError::inference)?;
        let flat: &[f32] = output.as_slice();

        postprocess_embedding(flat, tok.seq_len, &tok.attention_mask)
    }

    fn prepare_batch(&self, texts: &[&str], prefix: &str) -> Result<PaddedBatch, EmbedError> {
        let sorted_indices = sort_indices_by_len(texts);

        let prefixed: Vec<String> = sorted_indices
            .iter()
            .map(|&i| format!("{prefix}{}", texts[i]))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(EmbedError::tokenizer)?;

        let max_seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .ok_or_else(|| EmbedError::inference("tokenizer returned no encodings"))?;

        let batch_size = encodings.len();
        let mut input_ids = vec![0u32; batch_size * max_seq_len];
        let mut attention_mask = vec![0u32; batch_size * max_seq_len];
        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let offset = i * max_seq_len;
            input_ids[offset..offset + ids.len()].copy_from_slice(ids);
            attention_mask[offset..offset + mask.len()].copy_from_slice(mask);
        }

        Ok(PaddedBatch {
            sorted_indices,
            input_ids,
            attention_mask,
            max_seq_len,
        })
    }

    pub(super) fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch = self.prepare_batch(texts, DOCUMENT_PREFIX)?;

        let output = self
            .model
            .forward(
                &batch.input_ids,
                &batch.attention_mask,
                batch.sorted_indices.len() as i32,
                batch.max_seq_len as i32,
            )
            .map_err(EmbedError::inference)?;

        output.eval().map_err(EmbedError::inference)?;
        let flat: &[f32] = output.as_slice();

        unpack_batch_output(
            flat,
            &batch.sorted_indices,
            batch.max_seq_len,
            &batch.attention_mask,
        )
    }
}

/// Validate output shape and unpack batched model output into per-input embeddings.
pub(super) fn unpack_batch_output(
    flat: &[f32],
    sorted_indices: &[usize],
    max_seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let total = sorted_indices.len() * max_seq_len;
    if total == 0 || !flat.len().is_multiple_of(total) {
        return Err(EmbedError::DimensionMismatch {
            expected: total,
            actual: flat.len(),
        });
    }
    let hidden_size = flat.len() / total;
    let stride = max_seq_len * hidden_size;
    let mut results = vec![Vec::new(); sorted_indices.len()];
    for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        let seq_data = &flat[sorted_pos * stride..(sorted_pos + 1) * stride];
        let mask_slice =
            &attention_mask[sorted_pos * max_seq_len..(sorted_pos + 1) * max_seq_len];
        results[orig_idx] = postprocess_embedding(seq_data, max_seq_len, mask_slice)?;
    }
    Ok(results)
}
