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

/// Length-bucket upper bounds. A chunk with `len` tokens lands in the first
/// bucket whose `BUCKET_BOUNDS[i] >= len`. Final bucket equals [`MAX_SEQ_LEN`]
/// so every valid chunk (post-shrink) resolves to some bucket.
///
/// Bucketing keeps padding waste bounded by the bucket ceiling rather than the
/// global max chunk length, so a short chunk batched with a long one pays
/// at most the bucket's padding cost (Spec R-M01, AC-4, NFR-003).
pub(super) const BUCKET_BOUNDS: [usize; 4] = [128, 512, 2048, MAX_SEQ_LEN];

/// Per-chunk metadata carried through bucket forward so the flat output can be
/// restored to the original position after bucket passes reorder by length.
#[derive(Debug, Clone)]
pub(super) struct IndexedChunk {
    /// Position in the flat `all_chunk_tokens` ordering emitted by planning.
    global_idx: usize,
    /// Tokenized chunk payload (includes prefix + BOS/EOS).
    tokens: Vec<u32>,
}

/// Assign `len` to a bucket via the first `BUCKET_BOUNDS[i] >= len`.
///
/// # Panics
///
/// Panics if `len > MAX_SEQ_LEN`. Callers must ensure chunks have already
/// been shrunk to fit (via `shrink_chunk_to_fit`).
pub(super) fn assign_bucket(len: usize) -> usize {
    BUCKET_BOUNDS
        .iter()
        .position(|&max| len <= max)
        .expect("chunk len exceeds MAX_SEQ_LEN; shrink_chunk_to_fit must run first")
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

/// Wrap flat chunk tokens into indexed chunks. `global_idx` anchors each
/// chunk to its pre-bucketing position so bucket forward can reorder by
/// length yet still restore output order via the `global_idx` lookup.
fn build_indexed_chunks(all_chunk_tokens: Vec<Vec<u32>>) -> Vec<IndexedChunk> {
    all_chunk_tokens
        .into_iter()
        .enumerate()
        .map(|(global_idx, tokens)| IndexedChunk { global_idx, tokens })
        .collect()
}

/// Token-count ceiling for a single forward pass. Matches the pre-bucketing
/// value so the memory budget is unchanged by bucketing.
const TOKEN_BUDGET: usize = 256_000;

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
    /// Long documents are split into overlapping chunks. Chunks are routed into
    /// four length buckets (`[128, 512, 2048, MAX_SEQ_LEN]`) and forwarded per
    /// bucket, keeping padding waste bounded by the bucket ceiling.
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
        let total_chunks = all_chunk_tokens.len();
        metrics.num_chunks = total_chunks;

        let buckets = distribute_into_buckets(build_indexed_chunks(all_chunk_tokens));

        let mut out: Vec<Option<Vec<f32>>> = (0..total_chunks).map(|_| None).collect();

        for (bucket_idx, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            // sub_batch_size against the bucket ceiling keeps every possible
            // sub-batch under TOKEN_BUDGET even when every chunk is at the
            // bucket_max boundary — same OOM guarantee as pre-bucketing.
            let sub_batch_size = (TOKEN_BUDGET / BUCKET_BOUNDS[bucket_idx]).max(1);
            for sub_batch in bucket.chunks(sub_batch_size) {
                self.forward_sub_batch(sub_batch, &mut out, &mut metrics)?;
            }
        }

        metrics.log();

        // Invariant: each global_idx was written exactly once across all bucket
        // forwards. None here signals a distribution or unpack bug, not input.
        let all_embeddings: Vec<Vec<f32>> = out
            .into_iter()
            .map(|slot| slot.expect("every chunk slot must be filled by bucket forward"))
            .collect();

        let mut results = Vec::with_capacity(texts.len());
        let mut iter = all_embeddings.into_iter();
        for &count in &chunks_per_doc {
            let chunks: Vec<_> = iter.by_ref().take(count).collect();
            results.push(ChunkedEmbedding { chunks });
        }

        Ok(results)
    }

    /// Forward one sub-batch of indexed chunks, write pooled embeddings into
    /// `out` at each chunk's `global_idx`, and accumulate metrics.
    fn forward_sub_batch(
        &mut self,
        sub_batch: &[IndexedChunk],
        out: &mut [Option<Vec<f32>>],
        metrics: &mut PhaseMetrics,
    ) -> Result<(), EmbedError> {
        let sub_tokens: Vec<Vec<u32>> = sub_batch.iter().map(|c| c.tokens.clone()).collect();
        let (input_ids, attention_mask, batch_size, max_len) = pad_sequences(&sub_tokens, None);
        metrics.real_tokens += sub_tokens.iter().map(Vec::len).sum::<usize>();
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

        let sub_result: Result<Vec<Vec<f32>>, EmbedError> = (|| {
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

        for (chunk, emb) in sub_batch.iter().zip(sub_result?) {
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

    use super::{
        BUCKET_BOUNDS, IndexedChunk, assign_bucket, build_indexed_chunks, distribute_into_buckets,
    };

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

    fn make_chunk(global_idx: usize, token_count: usize) -> IndexedChunk {
        IndexedChunk {
            global_idx,
            tokens: vec![0u32; token_count],
        }
    }

    // T-BKT-001: assign_bucket boundary at 128 / 129
    #[test]
    fn t_bkt_001_assign_bucket_boundary_128_129() {
        assert_eq!(assign_bucket(1), 0, "len=1 is in bucket 0");
        assert_eq!(
            assign_bucket(128),
            0,
            "len=128 is in bucket 0 (upper bound)"
        );
        assert_eq!(assign_bucket(129), 1, "len=129 crosses into bucket 1");
    }

    // T-BKT-002: assign_bucket boundary at 512 / 513
    #[test]
    fn t_bkt_002_assign_bucket_boundary_512_513() {
        assert_eq!(
            assign_bucket(512),
            1,
            "len=512 is in bucket 1 (upper bound)"
        );
        assert_eq!(assign_bucket(513), 2, "len=513 crosses into bucket 2");
    }

    // T-BKT-003: assign_bucket boundary at 2048 / 2049
    #[test]
    fn t_bkt_003_assign_bucket_boundary_2048_2049() {
        assert_eq!(
            assign_bucket(2048),
            2,
            "len=2048 is in bucket 2 (upper bound)"
        );
        assert_eq!(assign_bucket(2049), 3, "len=2049 crosses into bucket 3");
    }

    // T-BKT-004: assign_bucket accepts up to MAX_SEQ_LEN (8192)
    #[test]
    fn t_bkt_004_assign_bucket_max_seq_len() {
        assert_eq!(
            assign_bucket(BUCKET_BOUNDS[3]),
            3,
            "len=MAX_SEQ_LEN is in final bucket"
        );
    }

    // T-BKT-005: all chunks at len=300 land in bucket 1; other buckets stay empty
    #[test]
    fn t_bkt_005_uniform_length_single_bucket() {
        let chunks: Vec<IndexedChunk> = (0..10).map(|i| make_chunk(i, 300)).collect();
        let buckets = distribute_into_buckets(chunks);
        assert_eq!(buckets[0].len(), 0, "bucket 0 (<=128) should be empty");
        assert_eq!(buckets[1].len(), 10, "all len=300 chunks in bucket 1");
        assert_eq!(buckets[2].len(), 0, "bucket 2 (<=2048) should be empty");
        assert_eq!(buckets[3].len(), 0, "bucket 3 should be empty");
    }

    // T-BKT-006: single-chunk distribution — other buckets empty, one bucket has 1
    #[test]
    fn t_bkt_006_single_chunk_distribution() {
        let buckets = distribute_into_buckets(vec![make_chunk(0, 50)]);
        let total: usize = buckets.iter().map(Vec::len).sum();
        assert_eq!(total, 1, "single chunk routes to exactly one bucket");
        assert_eq!(buckets[0].len(), 1, "len=50 lands in bucket 0");
    }

    #[test]
    fn build_indexed_chunks_assigns_sequential_global_idx() {
        let all_chunks = vec![vec![1u32; 10], vec![2u32; 20], vec![3u32; 30]];
        let indexed = build_indexed_chunks(all_chunks);
        let shape: Vec<(usize, usize)> = indexed
            .iter()
            .map(|c| (c.global_idx, c.tokens.len()))
            .collect();
        assert_eq!(shape, vec![(0, 10), (1, 20), (2, 30)]);
    }
}
