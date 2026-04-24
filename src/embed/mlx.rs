use std::time::Instant;

use super::Artifacts;
use super::metrics::PhaseMetrics;
use super::{
    CHUNK_OVERLAP_TOKENS, ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, EmbedInitError,
    MAX_SEQ_LEN, extract_prefix_tokens, max_content, postprocess_embedding, tokenize_with_prefix,
    truncate_for_query,
};
use crate::mlx_cache::release_inference_output;
use crate::model_io::pad_sequences;
use crate::modernbert::ModernBert;

pub(super) struct EmbedderInner {
    model: ModernBert,
    tokenizer: tokenizers::Tokenizer,
    doc_prefix_tokens: Vec<u32>,
    embedding_dims: usize,
}

impl EmbedderInner {
    pub(super) fn new(artifacts: &Artifacts) -> Result<Self, EmbedInitError> {
        let config = &artifacts.config;
        let tokenizer = artifacts.tokenizer.clone();

        let model =
            ModernBert::load(&artifacts.paths.model, config).map_err(EmbedInitError::backend)?;

        let doc_prefix_tokens =
            extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).map_err(EmbedInitError::backend)?;
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

    pub(super) fn embed_query_truncated(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let mut metrics = PhaseMetrics::new("query");

        let t_tok = Instant::now();
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;
        let (input_ids, attention_mask, seq_len) =
            truncate_for_query(tok.input_ids, tok.attention_mask, MAX_SEQ_LEN);
        metrics.tokenize = t_tok.elapsed();

        let seq_len_i32 = i32::try_from(seq_len).expect("seq_len fits in i32");
        let t_forward = Instant::now();
        let output = self
            .model
            .forward(&input_ids, &attention_mask, 1, seq_len_i32)
            .map_err(EmbedError::inference)?;

        let result = (|| {
            output.eval().map_err(EmbedError::inference)?;
            metrics.forward_eval = t_forward.elapsed();

            let t_readback = Instant::now();
            let flat: &[f32] = output.as_slice();
            let pooled = postprocess_embedding(flat, seq_len, &attention_mask)?;
            metrics.readback_pool = t_readback.elapsed();
            Ok(pooled)
        })();
        let t_clear = Instant::now();
        release_inference_output(output);
        metrics.cache_clear = t_clear.elapsed();

        // Query tokenization produces no padding (truncate, not pad), so every
        // mask entry is non-zero — real_tokens equals seq_len.
        metrics.real_tokens = seq_len;
        metrics.padded_tokens = seq_len;
        metrics.num_chunks = 1;
        metrics.batch_size = 1;
        metrics.max_seq_len = seq_len;
        metrics.log();

        result
    }

    pub(super) fn embed_document_chunked(
        &mut self,
        text: &str,
    ) -> Result<ChunkedEmbedding, EmbedError> {
        let mut results = self.embed_documents_batch_chunked(&[text])?;
        Ok(results.remove(0))
    }

    /// Batch-embed documents with chunking support.
    ///
    /// Short documents produce a single chunk identical to pre-chunking output.
    /// Long documents are split into overlapping chunks. All chunks from all
    /// documents are flattened into a single forward pass.
    pub(super) fn embed_documents_batch_chunked(
        &mut self,
        texts: &[&str],
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut metrics = PhaseMetrics::new("batch");

        let t_plan = Instant::now();
        let max_content_tokens = max_content(self.doc_prefix_tokens.len());
        let (all_chunk_tokens, chunks_per_doc) = plan_document_chunks(
            &self.tokenizer,
            texts,
            &self.doc_prefix_tokens,
            max_content_tokens,
        )?;
        metrics.chunk_plan = t_plan.elapsed();
        metrics.num_chunks = all_chunk_tokens.len();

        // Split into sub-batches bounded by TOKEN_BUDGET total positions to avoid
        // Metal OOM when long sequences produce large padded tensors.
        // TOKEN_BUDGET = 256K positions ≈ 128 chunks × 2048 tokens, which matches
        // the empirically confirmed safe range for ruri-v3-310m on Apple Silicon.
        const TOKEN_BUDGET: usize = 256_000;
        let max_len_overall = all_chunk_tokens.iter().map(Vec::len).max().unwrap_or(1);
        let sub_batch_size = (TOKEN_BUDGET / max_len_overall).max(1);

        let mut all_embeddings = Vec::with_capacity(all_chunk_tokens.len());
        for sub_batch in all_chunk_tokens.chunks(sub_batch_size) {
            let (input_ids, attention_mask, batch_size, max_len) = pad_sequences(sub_batch, None);
            // Pre-padding lengths sum to the real token count (pad_sequences pads with 0).
            metrics.real_tokens += sub_batch.iter().map(Vec::len).sum::<usize>();
            metrics.padded_tokens += batch_size * max_len;
            metrics.batch_size = metrics.batch_size.max(batch_size);
            metrics.max_seq_len = metrics.max_seq_len.max(max_len);

            let batch_size_i32 = i32::try_from(batch_size).expect("batch_size fits in i32");
            let max_len_i32 = i32::try_from(max_len).expect("max_len fits in i32");
            let t_forward = Instant::now();
            let output = self
                .model
                .forward(&input_ids, &attention_mask, batch_size_i32, max_len_i32)
                .map_err(EmbedError::inference)?;

            let sub_result = (|| {
                output.eval().map_err(EmbedError::inference)?;
                metrics.forward_eval += t_forward.elapsed();

                let t_readback = Instant::now();
                let flat: &[f32] = output.as_slice();
                let unpacked = unpack_batch_output(flat, batch_size, max_len, &attention_mask)?;
                metrics.readback_pool += t_readback.elapsed();
                Ok(unpacked)
            })();
            let t_clear = Instant::now();
            release_inference_output(output);
            metrics.cache_clear += t_clear.elapsed();
            all_embeddings.extend(sub_result?);
        }

        metrics.log();

        // Regroup by document, preserving input order
        let mut results = Vec::with_capacity(texts.len());
        let mut iter = all_embeddings.into_iter();
        for &count in &chunks_per_doc {
            let chunks: Vec<_> = iter.by_ref().take(count).collect();
            results.push(ChunkedEmbedding { chunks });
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
    max_content_tokens: usize,
) -> Result<(Vec<Vec<u32>>, Vec<usize>), EmbedError> {
    let mut all_chunk_tokens: Vec<Vec<u32>> = Vec::new();
    let mut chunks_per_doc: Vec<usize> = Vec::new();

    for &text in texts {
        let tok = tokenize_with_prefix(tokenizer, text, DOCUMENT_PREFIX)?;
        // Estimate text token count from the full tokenization to decide short/long path.
        // The estimate may be off by 1 due to prefix boundary merging, but that only
        // affects the path selection for texts near the boundary — both paths are correct.
        let text_token_count = tok.seq_len.saturating_sub(2 + prefix_tokens.len());

        if text_token_count <= max_content_tokens {
            // Short document: use full tokenization as-is (FR-012)
            all_chunk_tokens.push(tok.input_ids);
            chunks_per_doc.push(1);
        } else {
            let chunks = plan_long_document(tokenizer, text, max_content_tokens)?;
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
    max_content_tokens: usize,
) -> Result<Vec<Vec<u32>>, EmbedError> {
    let text_enc = tokenizer
        .encode(text, false)
        .map_err(EmbedError::tokenizer)?;
    let offsets = text_enc.get_offsets();
    let n = text_enc.get_ids().len();

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < n {
        let mut end = (start + max_content_tokens).min(n);
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

/// Validate output shape and unpack batched model output into per-chunk embeddings.
///
/// Returns a `Vec` in the same order as the input batch.
pub(super) fn unpack_batch_output(
    flat: &[f32],
    batch_size: usize,
    max_seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let total = batch_size
        .checked_mul(max_seq_len)
        .filter(|&t| t > 0 && flat.len().is_multiple_of(t))
        .ok_or(EmbedError::BufferShapeMismatch {
            expected: batch_size.saturating_mul(max_seq_len),
            actual: flat.len(),
        })?;
    let hidden_size = flat.len() / total;
    let stride = max_seq_len
        .checked_mul(hidden_size)
        .ok_or(EmbedError::BufferShapeMismatch {
            expected: total,
            actual: flat.len(),
        })?;
    let mut results = vec![Vec::new(); batch_size];
    for i in 0..batch_size {
        let seq_data = &flat[i * stride..(i + 1) * stride];
        let mask_slice = &attention_mask[i * max_seq_len..(i + 1) * max_seq_len];
        results[i] = postprocess_embedding(seq_data, max_seq_len, mask_slice)?;
    }
    Ok(results)
}

/// `t_NNN_` prefix maps to Spec test scenario IDs (T-001, T-002, …).
/// Tests without spec references omit the prefix.
#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;
    use std::sync::{Mutex, PoisonError};

    #[test]
    fn poison_recovery_pattern_works() {
        let lock = Mutex::new(());
        let _ = catch_unwind(|| {
            let _guard = lock.lock().unwrap();
            panic!("intentional panic to poison lock");
        });
        assert!(lock.is_poisoned());
        // Same pattern as mlx_cache::release_inference_output — unwrap_or_else recovers the guard
        let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        drop(guard);
    }
}
