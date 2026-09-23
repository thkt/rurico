#[cfg(test)]
use crate::mlx_cache::testing::{Stage, checkpoint};

use std::thread;
use std::time::Instant;

use mlx_rs::Array;

use super::Artifacts;
use super::metrics::{BatchMetrics, EmbedKind, PhaseMetrics};
use super::processing::{
    IndexedChunk, build_indexed_chunks, distribute_into_buckets, plan_document_chunks, split_pooled,
};
use super::{
    ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, EmbedOptions, MAX_SEQ_LEN, ModelInitError,
    extract_prefix_tokens, gpu_pool_and_normalize, max_content, tokenize_with_prefix,
    truncate_for_query,
};
use crate::mlx_cache::{Component, clear_inference_cache, run_inference};
use crate::model_io::{BUCKET_BOUNDS, assign_bucket, compute_sub_batch_size, pad_sequences};
use crate::modernbert::ModernBert;

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
    /// `run_inference` drops every temporary Array before cleanup, including
    /// partial forward, pool, eval and readback failures.
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
        let result = run_inference(
            || {
                self.model
                    .forward(&input_ids, &attention_mask, 1, bucket_len_i32)
                    .map_err(EmbedError::inference)
            },
            |output| {
                let pooled = pool_output(output, &attention_mask, 1, bucket_len_i32)?;
                metrics.forward_eval = t_forward.elapsed();

                let t_readback = Instant::now();
                let flat: &[f32] = pooled.as_slice();
                #[cfg(test)]
                checkpoint(Stage::Readback).map_err(EmbedError::inference)?;
                let pooled_vec = split_pooled(flat, 1, hidden_size, EmbedKind::Query)?
                    .into_iter()
                    .next()
                    .expect("split_pooled(_, 1, _) yields one row");
                metrics.readback_pool = t_readback.elapsed();
                Ok(pooled_vec)
            },
            || {
                let t_clear = Instant::now();
                clear_inference_cache(Component::Embed);
                metrics.cache_clear = t_clear.elapsed();
            },
        );

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
        self.embed_documents_batch_chunked_with_options(texts, &EmbedOptions::default())
    }

    /// Same as [`embed_documents_batch_chunked`](Self::embed_documents_batch_chunked)
    /// but honors [`EmbedOptions`]: `token_budget` overrides the sub-batch
    /// sizing budget and `forward_pause` sleeps after each forward pass.
    pub(super) fn embed_documents_batch_chunked_with_options(
        &mut self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch_chunked_with_metrics(texts, options)
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
        options: &EmbedOptions,
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
            let sub_batch_size =
                compute_sub_batch_size(BUCKET_BOUNDS[bucket_idx], options.token_budget);
            for sub_batch in bucket.chunks(sub_batch_size) {
                self.forward_sub_batch(sub_batch, bucket_idx, &mut out, &mut metrics)?;
                // Yield the GPU between forwards so interactive processes
                // (WindowServer) regain responsiveness during long batches.
                if let Some(pause) = options.forward_pause {
                    thread::sleep(pause);
                }
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
    /// vectors. `run_inference` drops all temporary Arrays before cleanup on
    /// success and on forward, pool, eval or readback failure.
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
        let unpacked = run_inference(
            || {
                self.model
                    .forward(&input_ids, &attention_mask, batch_size_i32, max_len_i32)
                    .map_err(EmbedError::inference)
            },
            |output| {
                let pooled = pool_output(output, &attention_mask, batch_size_i32, max_len_i32)?;
                metrics.forward_eval += t_forward.elapsed();

                let t_readback = Instant::now();
                let flat: &[f32] = pooled.as_slice();
                #[cfg(test)]
                checkpoint(Stage::Readback).map_err(EmbedError::inference)?;
                let unpacked = split_pooled(flat, batch_size, hidden_size, EmbedKind::Batch)?;
                metrics.readback_pool += t_readback.elapsed();
                Ok(unpacked)
            },
            || {
                let t_clear = Instant::now();
                clear_inference_cache(Component::Embed);
                metrics.cache_clear += t_clear.elapsed();
            },
        )?;

        for (chunk, emb) in sub_batch.iter().zip(unpacked) {
            out[chunk.global_idx] = Some(emb);
        }
        Ok(())
    }
}

/// Build the attention-mask `Array`, run `gpu_pool_and_normalize`, and
/// evaluate the lazy graph so the resulting `Array` is materialised on
/// the GPU before the caller reads it back.
///
/// The `output: Array` consume-by-value signature carries the
/// drop-before-clear contract of [`run_inference`] from this
/// layer up to the caller. The returned
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
    #[cfg(test)]
    checkpoint(Stage::Pool).map_err(EmbedError::inference)?;
    let pooled = gpu_pool_and_normalize(output, &mask).map_err(EmbedError::inference)?;
    pooled.eval().map_err(EmbedError::inference)?;
    #[cfg(test)]
    checkpoint(Stage::Eval).map_err(EmbedError::inference)?;
    Ok(pooled)
}
