#[cfg(test)]
use crate::mlx_cache::testing::{Stage, checkpoint};

use std::time::Instant;

use mlx_rs::Array;

use super::Artifacts;
use super::metrics::{EmbedKind, ForwardShape, PhaseMetrics};
use super::processing::{
    IndexedChunk, execute_document_chunks, plan_document_chunks, split_pooled,
};
use super::{
    ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, EmbedOptions, MAX_SEQ_LEN, ModelInitError,
    extract_prefix_tokens, gpu_pool_and_normalize, max_content, tokenize_with_prefix,
    truncate_for_query,
};
use crate::mlx_cache::{Component, clear_inference_cache, run_inference};
use crate::model_io::{BUCKET_BOUNDS, assign_bucket, pad_sequences};
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
    /// GPU pooling limits host readback to one `hidden_size` row, validated
    /// for shape and finite values. `run_inference` drops temporary Arrays
    /// before cleanup on success and failure.
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
                let flat = readback(&pooled, &mut metrics.readback_elements, false);
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
        self.embed_documents_batch_chunked_with_metrics(texts, options, false)
            .map(|(results, _metrics)| results)
    }

    /// Shared single-inference pipeline; detailed collection is opt-in.
    pub(super) fn embed_documents_batch_chunked_with_metrics(
        &mut self,
        texts: &[&str],
        options: &EmbedOptions,
        detailed: bool,
    ) -> Result<(Vec<ChunkedEmbedding>, PhaseMetrics), EmbedError> {
        if texts.is_empty() {
            return Ok((Vec::new(), PhaseMetrics::new(EmbedKind::Batch)));
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
        let results = execute_document_chunks(
            all_chunk_tokens,
            &chunks_per_doc,
            options,
            &mut metrics,
            detailed,
            |sub_batch, bucket_idx, metrics| {
                self.forward_sub_batch(sub_batch, bucket_idx, metrics, detailed)
            },
        )?;

        Ok((results, metrics))
    }

    /// Forward one sub-batch and return validated rows in sub-batch order.
    ///
    /// Read back only `batch_size * hidden_size` pooled elements and reject
    /// invalid shape or non-finite values. Temporary Arrays drop before cleanup
    /// on success and failure through `run_inference`.
    fn forward_sub_batch(
        &mut self,
        sub_batch: &[IndexedChunk],
        bucket_idx: usize,
        metrics: &mut PhaseMetrics,
        detailed: bool,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let t_pad = detailed.then(Instant::now);
        let (input_ids, attention_mask, batch_size, max_len) =
            pad_sequences(sub_batch, None, Some(BUCKET_BOUNDS[bucket_idx]));
        metrics.real_tokens += sub_batch.iter().map(|c| c.tokens.len()).sum::<usize>();
        metrics.padded_tokens += batch_size * max_len;
        metrics.batch_size = metrics.batch_size.max(batch_size);
        metrics.max_seq_len = metrics.max_seq_len.max(max_len);

        if let Some(t) = t_pad {
            metrics.preprocessing += t.elapsed();
        }

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
                let flat = readback(&pooled, &mut metrics.readback_elements, detailed);
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

        if detailed {
            metrics.forwards.push(ForwardShape {
                batch_size,
                sequence_length: max_len,
            });
        }

        Ok(unpacked)
    }
}

/// Pool and normalize on the GPU, then evaluate before host readback.
/// Consuming `output` leaves only the pooled Array for the caller to drop
/// inside `run_inference` before cleanup.
///
/// Callers supply a `[batch_size, seq_len]` binary mask; `ModernBert::forward`
/// validates its values upstream. Pool or eval failures return `EmbedError::Inference`.
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

// All embedding host accesses pass this boundary. Count the returned slice,
// including extra accesses; inference shape is only the independent expectation.
fn readback<'a>(array: &'a Array, elements: &mut Vec<usize>, observe: bool) -> &'a [f32] {
    let flat = array.as_slice();
    if observe {
        elements.push(flat.len());
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    #[test]
    fn readback_metrics_observe_host_accesses_and_slice_lengths() {
        require_unsandboxed_mlx_runtime();
        let array = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let expanded = Array::from_slice(&[1.0_f32; 6], &[2, 3]);
        let mut metrics = PhaseMetrics::new(EmbedKind::Batch);
        assert_eq!(
            readback(&array, &mut metrics.readback_elements, true),
            &[1.0, 2.0, 3.0]
        );
        readback(&array, &mut metrics.readback_elements, true);
        readback(&expanded, &mut metrics.readback_elements, true);
        let snapshot = super::super::metrics::InferenceMetrics::from(metrics);
        assert_eq!(snapshot.readback_elements, Some(vec![3, 3, 6]));
    }
}
