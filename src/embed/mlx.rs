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
        let result = postprocess_embedding(flat, tok.seq_len, &tok.attention_mask)?;
        release_inference_output(output);
        Ok(result)
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
        let result = unpack_batch_output(
            flat,
            &batch.sorted_indices,
            batch.max_seq_len,
            &batch.attention_mask,
        )?;
        release_inference_output(output);
        Ok(result)
    }
}

static MLX_CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn clear_cache() -> Result<(), EmbedError> {
    // Recover from poison: cache-clear is stateless (guards `()`), safe to proceed
    // after a panic in another thread. Contrast with `Embedder::lock_inner` (mod.rs)
    // which propagates poison because it guards mutable model state.
    let _guard = MLX_CACHE_LOCK.lock().unwrap_or_else(|e| {
        log::warn!("MLX cache lock was poisoned; recovering");
        e.into_inner()
    });
    // SAFETY:
    // 1. Callers use `release_inference_output` which takes the Array by value,
    //    ensuring no borrows/references to inference output remain
    // 2. model weights remain live on `&mut self` — only unused cache buffers are freed
    // 3. MLX_CACHE_LOCK guarantees exclusive access — no concurrent mlx_clear_cache calls
    let code = unsafe { mlx_sys::mlx_clear_cache() };
    check_clear_cache_result(code)
}

fn check_clear_cache_result(code: std::ffi::c_int) -> Result<(), EmbedError> {
    if code != 0 {
        return Err(EmbedError::inference(format!(
            "mlx_clear_cache failed (code: {code})"
        )));
    }
    Ok(())
}

/// Consume the MLX output array and attempt best-effort cache clearing.
///
/// Takes `output` by value to enforce the drop-before-clear ordering at
/// compile time. Cache-clear failure is non-fatal: the embedding result
/// is already in CPU memory as an owned `Vec`.
fn release_inference_output(output: mlx_rs::Array) {
    drop(output);
    if let Err(e) = clear_cache() {
        log::warn!("{e}");
    }
}

/// Validate output shape and unpack batched model output into per-input embeddings.
pub(super) fn unpack_batch_output(
    flat: &[f32],
    sorted_indices: &[usize],
    max_seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let total = sorted_indices
        .len()
        .checked_mul(max_seq_len)
        .filter(|&t| t > 0 && flat.len().is_multiple_of(t))
        .ok_or(EmbedError::DimensionMismatch {
            expected: sorted_indices.len().saturating_mul(max_seq_len),
            actual: flat.len(),
        })?;
    let hidden_size = flat.len() / total;
    let stride = max_seq_len.checked_mul(hidden_size).ok_or(
        EmbedError::DimensionMismatch {
            expected: total,
            actual: flat.len(),
        },
    )?;
    let mut results = vec![Vec::new(); sorted_indices.len()];
    for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        let seq_data = &flat[sorted_pos * stride..(sorted_pos + 1) * stride];
        let mask_slice = &attention_mask[sorted_pos * max_seq_len..(sorted_pos + 1) * max_seq_len];
        results[orig_idx] = postprocess_embedding(seq_data, max_seq_len, mask_slice)?;
    }
    Ok(results)
}

/// `t_NNN_` prefix maps to Spec test scenario IDs (T-001, T-002, …).
/// Tests without spec references omit the prefix.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_001_check_clear_cache_result_nonzero_returns_inference_error() {
        let err = check_clear_cache_result(1).unwrap_err();
        assert!(
            matches!(err, EmbedError::Inference(ref msg) if msg.contains("1")),
            "expected Inference error containing code '1', got: {err}"
        );
    }

    #[test]
    fn check_clear_cache_result_zero_returns_ok() {
        assert!(check_clear_cache_result(0).is_ok());
    }

    #[test]
    fn mlx_cache_lock_is_acquirable() {
        let guard = MLX_CACHE_LOCK.lock().unwrap();
        drop(guard);
    }

    #[test]
    fn poison_recovery_pattern_works() {
        let lock = std::sync::Mutex::new(());
        let _ = std::panic::catch_unwind(|| {
            let _guard = lock.lock().unwrap();
            panic!("intentional panic to poison lock");
        });
        assert!(lock.is_poisoned());
        // Same pattern as clear_cache — unwrap_or_else recovers the guard
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        drop(guard);
    }
}
