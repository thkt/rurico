use std::time::Instant;

use mlx_rs::Array;

use super::Artifacts;
use super::metrics::{BatchMetrics, EmbedKind, PhaseMetrics};
use super::{
    CHUNK_OVERLAP_TOKENS, ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, MAX_SEQ_LEN,
    ModelInitError, extract_prefix_tokens, gpu_pool_and_normalize, max_content,
    tokenize_with_prefix, truncate_for_query,
};
use crate::mlx_cache::{Component, clear_inference_cache, release_inference_output};
use crate::model_io::{BUCKET_BOUNDS, assign_bucket, compute_sub_batch_size, pad_sequences};
use crate::modernbert::ModernBert;

/// Per-chunk metadata carried through bucket forward so the flat output can be
/// restored to the original position after bucket passes reorder by length.
#[derive(Debug, Clone)]
pub(super) struct IndexedChunk {
    /// Position in the flat `all_chunk_tokens` ordering emitted by planning.
    global_idx: usize,
    /// Index of the originating document in the input `texts` slice.
    doc_idx: usize,
    /// 0-based chunk position inside the originating document.
    chunk_in_doc: usize,
    /// Tokenized chunk payload (includes prefix + BOS/EOS).
    tokens: Vec<u32>,
}

impl IndexedChunk {
    /// Sort key that clusters same-doc chunks together inside a bucket and
    /// keeps the chunk-in-doc reading order (R-M02). Shared so production and
    /// T-BKT-008 test against the same contract.
    fn doc_order_key(&self) -> (usize, usize) {
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
fn build_indexed_chunks(
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

pub(super) struct EmbedderInner {
    model: ModernBert,
    tokenizer: tokenizers::Tokenizer,
    doc_prefix_tokens: Vec<u32>,
    embedding_dims: usize,
}

impl EmbedderInner {
    pub(super) fn new(artifacts: &Artifacts) -> Result<Self, ModelInitError> {
        let config = &artifacts.config;
        let tokenizer = artifacts.tokenizer.clone();

        let model =
            ModernBert::load(&artifacts.paths.model, config).map_err(ModelInitError::backend)?;

        let doc_prefix_tokens =
            extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).map_err(ModelInitError::backend)?;
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

    /// Embed a single query string, truncating to [`MAX_SEQ_LEN`] tokens.
    ///
    /// Phase 3b GPU-pool path mirrors `forward_sub_batch` with `batch = 1`:
    /// `pool_output` runs the GPU pool + `eval()` so the readback through
    /// `pooled.as_slice()` reads only `hidden_size` f32s (NFR-002), then
    /// `split_pooled` validates that shape and rejects non-finite values
    /// before yielding the single row.
    /// `release_inference_output` consumes the **pooled** Array; the
    /// original `output` was consumed by `pool_output` (NFR-005
    /// drop-before-clear). Error path: when post-forward ops fail,
    /// `clear_inference_cache` keeps the MLX compile cache bounded
    /// (Codex CX-001 regression guard).
    pub(super) fn embed_query_truncated(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let mut metrics = PhaseMetrics::new(EmbedKind::Query);

        let t_tok = Instant::now();
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;
        let (mut input_ids, mut attention_mask, seq_len) =
            truncate_for_query(tok.input_ids, tok.attention_mask, MAX_SEQ_LEN);
        metrics.tokenize = t_tok.elapsed();

        let bucket_idx = assign_bucket(seq_len);
        let bucket_len = BUCKET_BOUNDS[bucket_idx];
        input_ids.resize(bucket_len, 0);
        attention_mask.resize(bucket_len, 0);
        let bucket_len_i32 = i32::try_from(bucket_len).expect("BUCKET_BOUNDS fits in i32");
        let hidden_size = self.embedding_dims;

        let t_forward = Instant::now();
        let output = self
            .model
            .forward(&input_ids, &attention_mask, 1, bucket_len_i32)
            .map_err(EmbedError::inference)?;

        let outcome: Result<(mlx_rs::Array, Vec<f32>), EmbedError> = (|| {
            let pooled = pool_output(output, &attention_mask, 1, bucket_len_i32)?;
            metrics.forward_eval = t_forward.elapsed();

            let t_readback = Instant::now();
            let flat: &[f32] = pooled.as_slice();
            let pooled_vec = split_pooled(flat, 1, hidden_size, EmbedKind::Query)?
                .into_iter()
                .next()
                .expect("split_pooled(_, 1, _) yields one row");
            metrics.readback_pool = t_readback.elapsed();
            Ok((pooled, pooled_vec))
        })();

        let t_clear = Instant::now();
        let result = match outcome {
            Ok((pooled, pooled_vec)) => {
                release_inference_output(pooled, Component::Embed);
                metrics.cache_clear = t_clear.elapsed();
                Ok(pooled_vec)
            }
            Err(e) => {
                clear_inference_cache(Component::Embed);
                metrics.cache_clear = t_clear.elapsed();
                Err(e)
            }
        };

        // Query has `seq_len` real tokens followed by `bucket_len - seq_len`
        // zero-padding tokens added for bucket alignment.
        metrics.real_tokens = seq_len;
        metrics.padded_tokens = bucket_len;
        metrics.num_chunks = 1;
        metrics.batch_size = 1;
        metrics.max_seq_len = bucket_len;
        metrics.bucket_hist[bucket_idx] = 1;
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
    /// Long documents are split into overlapping chunks. Chunks are routed into
    /// four length buckets (`[128, 512, 2048, MAX_SEQ_LEN]`) and forwarded per
    /// bucket, keeping padding waste bounded by the bucket ceiling.
    pub(super) fn embed_documents_batch_chunked(
        &mut self,
        texts: &[&str],
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch_chunked_with_metrics(texts)
            .map(|(results, _metrics)| results)
    }

    /// Same as [`embed_documents_batch_chunked`](Self::embed_documents_batch_chunked)
    /// but also returns a [`BatchMetrics`] snapshot of the call. Used by the
    /// smoke harness (PR #6) to assert SLA + padding + R² bounds without
    /// parsing a debug-log line. Empty `texts` yields
    /// `(Vec::new(), BatchMetrics::default())`.
    pub(super) fn embed_documents_batch_chunked_with_metrics(
        &mut self,
        texts: &[&str],
    ) -> Result<(Vec<ChunkedEmbedding>, BatchMetrics), EmbedError> {
        if texts.is_empty() {
            return Ok((Vec::new(), BatchMetrics::default()));
        }

        let mut metrics = PhaseMetrics::new(EmbedKind::Batch);

        let t_plan = Instant::now();
        let max_content_tokens = max_content(self.doc_prefix_tokens.len());
        let (all_chunk_tokens, chunks_per_doc) = plan_document_chunks(
            &self.tokenizer,
            texts,
            &self.doc_prefix_tokens,
            max_content_tokens,
        )?;
        metrics.chunk_plan = t_plan.elapsed();
        let total_chunks = all_chunk_tokens.len();
        metrics.num_chunks = total_chunks;

        let buckets =
            distribute_into_buckets(build_indexed_chunks(all_chunk_tokens, &chunks_per_doc)?);

        let mut out: Vec<Option<Vec<f32>>> = (0..total_chunks).map(|_| None).collect();

        for (bucket_idx, mut bucket) in buckets.into_iter().enumerate() {
            metrics.bucket_hist[bucket_idx] = bucket.len();
            if bucket.is_empty() {
                continue;
            }
            // R-M02: cluster same-doc chunks inside each bucket so a sub_batch
            // prefers to carry chunks from the same document. chunk_in_doc is
            // the tie-breaker to preserve reading order within a doc.
            bucket.sort_by_key(IndexedChunk::doc_order_key);
            // sub_batch_size against the bucket ceiling keeps every possible
            // sub-batch under TOKEN_BUDGET even when every chunk is at the
            // bucket_max boundary, matching the pre-bucketing OOM guarantee.
            let sub_batch_size = compute_sub_batch_size(BUCKET_BOUNDS[bucket_idx], None);
            for sub_batch in bucket.chunks(sub_batch_size) {
                self.forward_sub_batch(sub_batch, bucket_idx, &mut out, &mut metrics)?;
            }
        }

        metrics.log();
        let batch_metrics = BatchMetrics::from(&metrics);

        // Invariant: each global_idx was written exactly once across all bucket
        // forwards. None here signals a distribution or unpack bug, not input —
        // surfaced as `Inference` so a regression cannot panic in production.
        let all_embeddings: Vec<Vec<f32>> = out
            .into_iter()
            .enumerate()
            .map(|(idx, slot)| {
                slot.ok_or_else(|| {
                    EmbedError::inference_message(format!(
                        "chunk slot {idx} not filled by any bucket forward (distribution bug)"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut results = Vec::with_capacity(texts.len());
        let mut iter = all_embeddings.into_iter();
        for &count in &chunks_per_doc {
            let chunks: Vec<_> = iter.by_ref().take(count).collect();
            results.push(ChunkedEmbedding::try_new(chunks)?);
        }

        Ok((results, batch_metrics))
    }

    /// Forward one sub-batch of indexed chunks, write pooled embeddings into
    /// `out` at each chunk's `global_idx`, and accumulate metrics.
    ///
    /// Phase 3b GPU-pool path: `pool_output` runs the GPU mask-weighted
    /// mean + L2 normalize and `eval()` materialises the lazy graph; the
    /// readback then reads only `batch_size * hidden_size` f32 elements
    /// (NFR-002, ADR 0002 primary lever) instead of `batch * seq * hidden`.
    /// `split_pooled` validates the readback shape per sub-batch (FR-002a)
    /// and rejects non-finite values before splitting into per-chunk
    /// vectors. The `release_inference_output` argument is the **pooled**
    /// Array; the original `output` was consumed by `pool_output`
    /// (NFR-005 drop-before-clear).
    ///
    /// Error path: when `pool_output` / `split_pooled` errors after the
    /// model forward succeeded, `output` was consumed and there is no
    /// Array to release. `clear_inference_cache` runs unconditionally so
    /// the global MLX compile cache does not accumulate kernel entries
    /// across failed forwards (Codex CX-001 regression guard).
    fn forward_sub_batch(
        &mut self,
        sub_batch: &[IndexedChunk],
        bucket_idx: usize,
        out: &mut [Option<Vec<f32>>],
        metrics: &mut PhaseMetrics,
    ) -> Result<(), EmbedError> {
        let sub_tokens: Vec<Vec<u32>> = sub_batch.iter().map(|c| c.tokens.clone()).collect();
        let (input_ids, attention_mask, batch_size, max_len) =
            pad_sequences(&sub_tokens, None, Some(BUCKET_BOUNDS[bucket_idx]));
        metrics.real_tokens += sub_tokens.iter().map(Vec::len).sum::<usize>();
        metrics.padded_tokens += batch_size * max_len;
        metrics.batch_size = metrics.batch_size.max(batch_size);
        metrics.max_seq_len = metrics.max_seq_len.max(max_len);

        let batch_size_i32 = i32::try_from(batch_size).expect("batch_size fits in i32");
        let max_len_i32 = i32::try_from(max_len).expect("max_len fits in i32");
        let hidden_size = self.embedding_dims;

        let t_forward = Instant::now();
        let output = self
            .model
            .forward(&input_ids, &attention_mask, batch_size_i32, max_len_i32)
            .map_err(EmbedError::inference)?;

        // Closure carries the pooled Array out on success so the caller
        // releases it (drop-before-clear); on Err the closure has already
        // dropped any partial Array via `?` and the caller falls through
        // to `clear_inference_cache`.
        let outcome: Result<(mlx_rs::Array, Vec<Vec<f32>>), EmbedError> = (|| {
            let pooled = pool_output(output, &attention_mask, batch_size_i32, max_len_i32)?;
            metrics.forward_eval += t_forward.elapsed();

            let t_readback = Instant::now();
            let flat: &[f32] = pooled.as_slice();
            let unpacked = split_pooled(flat, batch_size, hidden_size, EmbedKind::Batch)?;
            metrics.readback_pool += t_readback.elapsed();
            Ok((pooled, unpacked))
        })();

        let t_clear = Instant::now();
        let unpacked = match outcome {
            Ok((pooled, unpacked)) => {
                release_inference_output(pooled, Component::Embed);
                metrics.cache_clear += t_clear.elapsed();
                unpacked
            }
            Err(e) => {
                clear_inference_cache(Component::Embed);
                metrics.cache_clear += t_clear.elapsed();
                return Err(e);
            }
        };

        for (chunk, emb) in sub_batch.iter().zip(unpacked) {
            out[chunk.global_idx] = Some(emb);
        }
        Ok(())
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

/// Build the attention-mask `Array`, run `gpu_pool_and_normalize`, and
/// evaluate the lazy graph so the resulting `Array` is materialised on
/// the GPU before the caller reads it back.
///
/// The `output: Array` consume-by-value signature carries the
/// drop-before-clear contract of [`release_inference_output`] from this
/// layer up to the caller (compile-time guard via T-014). The returned
/// pooled `Array` is the **only** Array the caller now owns from this
/// forward pass — `output` was consumed by `gpu_pool_and_normalize`.
///
/// `attention_mask` is constructed with shape `[batch_size, seq_len]` by
/// `Array::from_slice`; production callers (`pad_sequences`) guarantee
/// `attention_mask.len() == batch_size * seq_len` and that mask values
/// are `0` or `1` (validated upstream by
/// `ModernBert::forward::validate_attention_mask`).
///
/// # Errors
///
/// Returns [`EmbedError::Inference`] if a pool op or `eval` fails.
pub(super) fn pool_output(
    output: Array,
    attention_mask: &[u32],
    batch_size: i32,
    seq_len: i32,
) -> Result<Array, EmbedError> {
    let mask = Array::from_slice(attention_mask, &[batch_size, seq_len]);
    let pooled = gpu_pool_and_normalize(output, &mask).map_err(EmbedError::inference)?;
    pooled.eval().map_err(EmbedError::inference)?;
    Ok(pooled)
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
