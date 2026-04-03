use super::{
    CHUNK_OVERLAP_TOKENS, ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, MAX_SEQ_LEN, ModelPaths,
    extract_prefix_tokens, load_tokenizer, max_content, postprocess_embedding, read_config,
    tokenize_with_prefix, truncate_for_query,
};

pub(super) struct EmbedderInner {
    model: crate::modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
    doc_prefix_tokens: Vec<u32>,
    embedding_dims: usize,
}

impl EmbedderInner {
    pub(super) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;

        let config: crate::modernbert::Config = read_config(&paths.config)?;

        let model = crate::modernbert::ModernBert::load(&paths.model, &config)
            .map_err(EmbedError::inference)?;

        let tokenizer = load_tokenizer(&paths.tokenizer)?;
        let doc_prefix_tokens = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX)?;
        let embedding_dims = config.hidden_size;

        Ok(Self {
            model,
            tokenizer,
            doc_prefix_tokens,
            embedding_dims,
        })
    }

    pub(super) fn embedding_dims(&self) -> usize {
        self.embedding_dims
    }

    /// Embed a query with truncation (no chunking). FR-009, FR-010.
    pub(super) fn embed_query_truncated(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;
        let (input_ids, attention_mask, seq_len) =
            truncate_for_query(tok.input_ids, tok.attention_mask, MAX_SEQ_LEN);

        let output = self
            .model
            .forward(&input_ids, &attention_mask, 1, seq_len as i32)
            .map_err(EmbedError::inference)?;

        output.eval().map_err(EmbedError::inference)?;
        let flat: &[f32] = output.as_slice();
        let result = postprocess_embedding(flat, seq_len, &attention_mask)?;
        release_inference_output(output);
        Ok(result)
    }

    /// Embed a single document with chunking support. FR-003, FR-006, FR-012.
    pub(super) fn embed_document_chunked(
        &mut self,
        text: &str,
    ) -> Result<ChunkedEmbedding, EmbedError> {
        let mut results = self.embed_documents_batch_chunked(&[text])?;
        Ok(results.remove(0))
    }

    /// Batch-embed documents with chunking support. FR-003, FR-006, FR-007, FR-012.
    ///
    /// Short documents produce a single chunk identical to pre-chunking output.
    /// Long documents are split into overlapping chunks. All chunks from all
    /// documents are flattened into a single forward pass (NFR-002).
    pub(super) fn embed_documents_batch_chunked(
        &mut self,
        texts: &[&str],
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mc = max_content(self.doc_prefix_tokens.len());
        let (all_chunk_tokens, chunks_per_doc) =
            plan_document_chunks(&self.tokenizer, texts, &self.doc_prefix_tokens, mc)?;

        // Pad and flatten into batch (sorted by length for padding efficiency)
        let max_len = all_chunk_tokens.iter().map(|c| c.len()).max().unwrap_or(0);
        let batch_size = all_chunk_tokens.len();

        let mut sorted_indices: Vec<usize> = (0..batch_size).collect();
        sorted_indices.sort_unstable_by_key(|&i| all_chunk_tokens[i].len());

        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];
        for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
            let chunk = &all_chunk_tokens[orig_idx];
            let offset = sorted_pos * max_len;
            input_ids[offset..offset + chunk.len()].copy_from_slice(chunk);
            attention_mask[offset..offset + chunk.len()].fill(1);
        }

        // Single forward pass (NFR-002)
        let output = self
            .model
            .forward(
                &input_ids,
                &attention_mask,
                batch_size as i32,
                max_len as i32,
            )
            .map_err(EmbedError::inference)?;

        output.eval().map_err(EmbedError::inference)?;
        let flat: &[f32] = output.as_slice();

        // Unpack (sorted → original chunk order)
        let all_embeddings = unpack_batch_output(flat, &sorted_indices, max_len, &attention_mask)?;
        release_inference_output(output);

        // Regroup by document (FR-007: preserving input order)
        let mut results = Vec::with_capacity(texts.len());
        let mut offset = 0;
        for &count in &chunks_per_doc {
            let chunks = all_embeddings[offset..offset + count].to_vec();
            results.push(ChunkedEmbedding { chunks });
            offset += count;
        }

        Ok(results)
    }
}

/// Plan token chunks for a batch of documents.
///
/// For each document:
/// - Short documents (text tokens ≤ max_content): single chunk from full tokenization
/// - Long documents: sequential planner with prefix-aware re-tokenization
///
/// Returns (all_chunk_tokens, chunks_per_doc).
fn plan_document_chunks(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
    prefix_tokens: &[u32],
    mc: usize,
) -> Result<(Vec<Vec<u32>>, Vec<usize>), EmbedError> {
    let mut all_chunk_tokens: Vec<Vec<u32>> = Vec::new();
    let mut chunks_per_doc: Vec<usize> = Vec::new();

    for &text in texts {
        let tok = tokenize_with_prefix(tokenizer, text, DOCUMENT_PREFIX)?;
        // Estimate text token count from the full tokenization to decide short/long path.
        // The estimate may be off by 1 due to prefix boundary merging, but that only
        // affects the path selection for texts near the boundary — both paths are correct.
        let text_token_count = tok.seq_len.saturating_sub(2 + prefix_tokens.len());

        if text_token_count <= mc {
            // Short document: use full tokenization as-is (FR-012)
            all_chunk_tokens.push(tok.input_ids);
            chunks_per_doc.push(1);
        } else {
            let chunks = plan_long_document(tokenizer, text, mc)?;
            chunks_per_doc.push(chunks.len());
            all_chunk_tokens.extend(chunks);
        }
    }

    Ok((all_chunk_tokens, chunks_per_doc))
}

/// Plan chunks for a single long document using sequential re-tokenization.
///
/// Each chunk is re-tokenized with the document prefix to handle prefix boundary
/// merging correctly (Approach A / IG-001). The adaptive shrink loop reduces
/// chunk size until the re-tokenized result fits within [`MAX_SEQ_LEN`].
fn plan_long_document(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    mc: usize,
) -> Result<Vec<Vec<u32>>, EmbedError> {
    let text_enc = tokenizer
        .encode(text, false)
        .map_err(EmbedError::tokenizer)?;
    let offsets = text_enc.get_offsets();
    let n = text_enc.get_ids().len();

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < n {
        let mut end = (start + mc).min(n);
        let ids = shrink_chunk_to_fit(tokenizer, text, offsets, start, &mut end)?;
        chunks.push(ids);

        if end >= n {
            break;
        }
        let next_start = end.saturating_sub(CHUNK_OVERLAP_TOKENS);
        if next_start <= start {
            break;
        }
        start = next_start;
    }

    Ok(chunks)
}

/// Re-tokenize a candidate chunk, shrinking until it fits within [`MAX_SEQ_LEN`].
///
/// Encodes `DOCUMENT_PREFIX + text[offsets[start].0..byte_end]` with special tokens.
/// Decreases `end` by one token at a time until the result fits.
pub(super) fn shrink_chunk_to_fit(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    offsets: &[(usize, usize)],
    start: usize,
    end: &mut usize,
) -> Result<Vec<u32>, EmbedError> {
    let byte_start = offsets[start].0;
    loop {
        if *end <= start {
            return Err(EmbedError::inference(format!(
                "chunk at token {start} cannot fit within \
                 MAX_SEQ_LEN after adaptive shrink"
            )));
        }
        let byte_end = offsets[*end - 1].1;
        let tok = tokenize_with_prefix(tokenizer, &text[byte_start..byte_end], DOCUMENT_PREFIX)?;
        if tok.seq_len <= MAX_SEQ_LEN {
            return Ok(tok.input_ids);
        }
        *end -= 1;
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
///
/// `sorted_indices` maps sorted position → original index (as produced by
/// sorting chunk indices by token length). Returns a `Vec` indexed by
/// original input order, not sorted order.
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
        .ok_or(EmbedError::BufferShapeMismatch {
            expected: sorted_indices.len().saturating_mul(max_seq_len),
            actual: flat.len(),
        })?;
    let hidden_size = flat.len() / total;
    let stride = max_seq_len
        .checked_mul(hidden_size)
        .ok_or(EmbedError::BufferShapeMismatch {
            expected: total,
            actual: flat.len(),
        })?;
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
