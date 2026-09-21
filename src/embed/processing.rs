//! Token planning, bucket distribution and readback validation shared by MLX and CPU tests.
use super::metrics::EmbedKind;
use super::{CHUNK_OVERLAP_TOKENS, DOCUMENT_PREFIX, EmbedError, MAX_SEQ_LEN, tokenize_with_prefix};
use crate::model_io::assign_bucket;

/// Per-chunk metadata carried through bucket forward so the flat output can be
/// restored to the original position after bucket passes reorder by length.
#[derive(Debug, Clone)]
pub(super) struct IndexedChunk {
    /// Position in the flat `all_chunk_tokens` ordering emitted by planning.
    pub(super) global_idx: usize,
    /// Index of the originating document in the input `texts` slice.
    doc_idx: usize,
    /// 0-based chunk position inside the originating document.
    chunk_in_doc: usize,
    /// Tokenized chunk payload (includes prefix + BOS/EOS).
    pub(super) tokens: Vec<u32>,
}

impl IndexedChunk {
    /// Sort key that clusters same-doc chunks together inside a bucket and
    /// keeps the chunk-in-doc reading order (R-M02). Shared so production and
    /// T-BKT-008 test against the same contract.
    pub(super) fn doc_order_key(&self) -> (usize, usize) {
        (self.doc_idx, self.chunk_in_doc)
    }
}

/// Partition indexed chunks into the four length buckets.
///
/// Kept as a standalone helper so the pure distribution logic can be tested
/// without spinning up MLX (T-BKT-005, T-BKT-006).
pub(super) fn distribute_into_buckets(chunks: Vec<IndexedChunk>) -> [Vec<IndexedChunk>; 4] {
    let mut buckets: [Vec<IndexedChunk>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for chunk in chunks {
        let b = assign_bucket(chunk.tokens.len());
        buckets[b].push(chunk);
    }
    buckets
}

/// Wrap flat chunk tokens into indexed chunks.
///
/// `global_idx` anchors each chunk to its pre-bucketing position so bucket
/// forward can reorder by length yet still restore output order via the
/// `global_idx` lookup. `doc_idx` + `chunk_in_doc` carry the document layout
/// through the bucket pass so same-doc chunks can be clustered inside each
/// bucket (R-M02) and the chunk-in-doc order can be preserved.
///
/// `chunks_per_doc[i]` must equal the number of chunks the i-th document
/// contributed to `all_chunk_tokens`; the sum across all docs must equal
/// `all_chunk_tokens.len()` (guaranteed by `plan_document_chunks`).
pub(super) fn build_indexed_chunks(
    all_chunk_tokens: Vec<Vec<u32>>,
    chunks_per_doc: &[usize],
) -> Result<Vec<IndexedChunk>, EmbedError> {
    let mut result = Vec::with_capacity(all_chunk_tokens.len());
    let mut tokens_iter = all_chunk_tokens.into_iter();
    for (doc_idx, &count) in chunks_per_doc.iter().enumerate() {
        for chunk_in_doc in 0..count {
            let tokens = tokens_iter.next().ok_or_else(|| {
                EmbedError::inference_message(format!(
                    "chunks_per_doc total exceeds all_chunk_tokens length \
                     (doc_idx={doc_idx}, chunk_in_doc={chunk_in_doc})"
                ))
            })?;
            result.push(IndexedChunk {
                global_idx: result.len(),
                doc_idx,
                chunk_in_doc,
                tokens,
            });
        }
    }
    if tokens_iter.next().is_some() {
        let extras = tokens_iter.count() + 1;
        return Err(EmbedError::inference_message(format!(
            "all_chunk_tokens has {extras} more entries than chunks_per_doc total"
        )));
    }
    Ok(result)
}

/// Plan token chunks for a batch of documents.
///
/// For each document:
/// - Short documents (text tokens ≤ max_content): single chunk from full tokenization
/// - Long documents: sequential planner with prefix-aware re-tokenization
///
/// Returns (all_chunk_tokens, chunks_per_doc).
pub(super) fn plan_document_chunks(
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
            tracing::warn!(
                start_token = start,
                end_token = *end,
                text_byte_start = byte_start,
                text_total_bytes = text.len(),
                total_offsets = offsets.len(),
                "chunk cannot fit within MAX_SEQ_LEN after adaptive shrink"
            );
            return Err(EmbedError::inference_message(format!(
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

/// Split a flat pooled buffer of length `batch_size * hidden_size` into
/// `batch_size` owned `hidden_size`-long row vectors in row-major order.
///
/// The caller (typically `forward_sub_batch` after `pool_output`) performs
/// `pooled.as_slice()` and feeds the resulting slice here. Keeping
/// `flat: &[f32]` (instead of `&mlx_rs::Array`) makes this function
/// MLX-free and testable under the Codex seatbelt — the chief reason it is
/// split out from `pool_output`. ADR 0002 sub-decision 2 / NFR-005: the
/// GPU pool reduces readback to `batch_size * hidden_size` floats, so the
/// validation here mirrors the `O(hidden)` invariant.
///
/// The `is_finite` check guards against non-finite outputs (corrupt
/// weights, kernel overflow); this catches
/// sources beyond the all-zero-mask `0/0` case that
/// `validate_attention_mask` already rejects upstream. It runs against
/// the already-readback flat buffer so it does not defeat the ADR 0002
/// readback-free hot path.
///
/// # Errors
///
/// - [`EmbedError::BufferShapeMismatch`] when `flat.len() != batch_size *
///   hidden_size`. Both directions (short and long) error out
///   symmetrically; silently slicing an incomplete or surplus tail is a
///   regression bug.
/// - [`EmbedError::NonFiniteOutput`] when any element of `flat` is
///   `NaN` or `±Inf`.
pub(super) fn split_pooled(
    flat: &[f32],
    batch_size: usize,
    hidden_size: usize,
    call_site: EmbedKind,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let call_site = call_site.as_str();
    let expected = batch_size.saturating_mul(hidden_size);
    if flat.len() != expected {
        tracing::warn!(
            call_site,
            expected,
            actual = flat.len(),
            batch_size,
            hidden_size,
            "split_pooled: buffer shape mismatch"
        );
        return Err(EmbedError::BufferShapeMismatch {
            expected,
            actual: flat.len(),
        });
    }
    if !flat.iter().all(|v| v.is_finite()) {
        tracing::warn!(
            call_site,
            batch_size,
            hidden_size,
            "split_pooled: non-finite output detected (NaN or Inf in pooled buffer)"
        );
        return Err(EmbedError::NonFiniteOutput);
    }
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    Ok(flat
        .chunks_exact(hidden_size)
        .map(<[f32]>::to_vec)
        .collect())
}

/// `t_NNN_` prefix maps to Spec test scenario IDs (T-001, T-002, …).
/// Tests without spec references omit the prefix.
#[cfg(test)]
mod tests;
