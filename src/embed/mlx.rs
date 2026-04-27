use std::time::Instant;

use mlx_rs::Array;

use super::Artifacts;
use super::metrics::{BatchMetrics, PhaseMetrics};
use super::{
    CHUNK_OVERLAP_TOKENS, ChunkedEmbedding, DOCUMENT_PREFIX, EmbedError, EmbedInitError,
    MAX_SEQ_LEN, extract_prefix_tokens, gpu_pool_and_normalize, max_content, tokenize_with_prefix,
    truncate_for_query,
};
use crate::mlx_cache::{clear_inference_cache, release_inference_output};
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
) -> Vec<IndexedChunk> {
    let mut result = Vec::with_capacity(all_chunk_tokens.len());
    let mut tokens_iter = all_chunk_tokens.into_iter();
    for (doc_idx, &count) in chunks_per_doc.iter().enumerate() {
        for chunk_in_doc in 0..count {
            let tokens = tokens_iter
                .next()
                .expect("chunks_per_doc total must match all_chunk_tokens length");
            result.push(IndexedChunk {
                global_idx: result.len(),
                doc_idx,
                chunk_in_doc,
                tokens,
            });
        }
    }
    debug_assert!(
        tokens_iter.next().is_none(),
        "chunks_per_doc total must match all_chunk_tokens length"
    );
    result
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

    /// Embed a single query string, truncating to [`MAX_SEQ_LEN`] tokens.
    ///
    /// Phase 3b GPU-pool path mirrors `forward_sub_batch` with `batch = 1`:
    /// `pool_output` runs the GPU pool + `eval()` so the readback through
    /// `pooled.as_slice()` reads only `hidden_size` f32s (NFR-002), then
    /// `split_pooled(flat, 1, hidden_size)` validates that shape and
    /// rejects non-finite values before yielding the single row.
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
        let mut metrics = PhaseMetrics::new("query");

        let t_tok = Instant::now();
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;
        let (input_ids, attention_mask, seq_len) =
            truncate_for_query(tok.input_ids, tok.attention_mask, MAX_SEQ_LEN);
        metrics.tokenize = t_tok.elapsed();

        let seq_len_i32 = i32::try_from(seq_len).expect("seq_len fits in i32");
        let hidden_size = self.embedding_dims;

        let t_forward = Instant::now();
        let output = self
            .model
            .forward(&input_ids, &attention_mask, 1, seq_len_i32)
            .map_err(EmbedError::inference)?;

        let outcome: Result<(mlx_rs::Array, Vec<f32>), EmbedError> = (|| {
            let pooled = pool_output(output, &attention_mask, 1, seq_len_i32)?;
            metrics.forward_eval = t_forward.elapsed();

            let t_readback = Instant::now();
            let flat: &[f32] = pooled.as_slice();
            let pooled_vec = split_pooled(flat, 1, hidden_size)?
                .into_iter()
                .next()
                .expect("split_pooled(_, 1, _) yields one row");
            metrics.readback_pool = t_readback.elapsed();
            Ok((pooled, pooled_vec))
        })();

        let t_clear = Instant::now();
        let result = match outcome {
            Ok((pooled, pooled_vec)) => {
                release_inference_output(pooled);
                metrics.cache_clear = t_clear.elapsed();
                Ok(pooled_vec)
            }
            Err(e) => {
                clear_inference_cache();
                metrics.cache_clear = t_clear.elapsed();
                Err(e)
            }
        };

        // Query tokenization produces no padding (truncate, not pad), so every
        // mask entry is non-zero — real_tokens equals seq_len.
        metrics.real_tokens = seq_len;
        metrics.padded_tokens = seq_len;
        metrics.num_chunks = 1;
        metrics.batch_size = 1;
        metrics.max_seq_len = seq_len;
        metrics.bucket_hist[assign_bucket(seq_len)] = 1;
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

        let buckets =
            distribute_into_buckets(build_indexed_chunks(all_chunk_tokens, &chunks_per_doc));

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
            let sub_batch_size = (TOKEN_BUDGET / BUCKET_BOUNDS[bucket_idx]).max(1);
            for sub_batch in bucket.chunks(sub_batch_size) {
                self.forward_sub_batch(sub_batch, &mut out, &mut metrics)?;
            }
        }

        metrics.log();
        let batch_metrics = BatchMetrics::from(&metrics);

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
            results.push(ChunkedEmbedding::new(chunks));
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
            let unpacked = split_pooled(flat, batch_size, hidden_size)?;
            metrics.readback_pool += t_readback.elapsed();
            Ok((pooled, unpacked))
        })();

        let t_clear = Instant::now();
        let unpacked = match outcome {
            Ok((pooled, unpacked)) => {
                release_inference_output(pooled);
                metrics.cache_clear += t_clear.elapsed();
                unpacked
            }
            Err(e) => {
                clear_inference_cache();
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
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let expected = batch_size.saturating_mul(hidden_size);
    if flat.len() != expected {
        return Err(EmbedError::BufferShapeMismatch {
            expected,
            actual: flat.len(),
        });
    }
    if !flat.iter().all(|v| v.is_finite()) {
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
mod tests {
    use std::panic::catch_unwind;
    use std::sync::{Mutex, PoisonError};

    use mlx_rs::Array;

    use super::super::EmbedError;
    use super::{
        BUCKET_BOUNDS, IndexedChunk, assign_bucket, build_indexed_chunks, distribute_into_buckets,
        pool_output, split_pooled,
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

    /// Full-position helper used by T-BKT-008 to express a shuffled multi-chunk
    /// doc layout. `make_chunk` delegates here with each chunk as its own
    /// single-chunk document.
    fn make_chunk_at(
        global_idx: usize,
        doc_idx: usize,
        chunk_in_doc: usize,
        token_count: usize,
    ) -> IndexedChunk {
        IndexedChunk {
            global_idx,
            doc_idx,
            chunk_in_doc,
            tokens: vec![0u32; token_count],
        }
    }

    fn make_chunk(global_idx: usize, token_count: usize) -> IndexedChunk {
        make_chunk_at(global_idx, global_idx, 0, token_count)
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
    fn build_indexed_chunks_tracks_doc_structure() {
        // doc 0 has 2 chunks, doc 1 has 1 chunk → chunks_per_doc = [2, 1]
        let all_chunks = vec![vec![1u32; 10], vec![2u32; 20], vec![3u32; 30]];
        let indexed = build_indexed_chunks(all_chunks, &[2, 1]);
        let shape: Vec<(usize, usize, usize, usize)> = indexed
            .iter()
            .map(|c| (c.global_idx, c.doc_idx, c.chunk_in_doc, c.tokens.len()))
            .collect();
        assert_eq!(
            shape,
            vec![(0, 0, 0, 10), (1, 0, 1, 20), (2, 1, 0, 30)],
            "global_idx runs 0..N while doc_idx + chunk_in_doc track doc layout"
        );
    }

    // T-BKT-007: 10 chunks spanning all 4 buckets round-trip in original order
    //
    // Mirrors the restoration that `embed_documents_batch_chunked` performs:
    // forward writes `out[chunk.global_idx]`, so flattening every bucket and
    // sorting by `global_idx` must recover the original insertion order.
    // Testing this on `distribute_into_buckets` keeps the check MLX-free.
    #[test]
    fn t_bkt_007_cross_bucket_order_preserved() {
        // bucket 0 (≤128): idx 0, 4 | bucket 1 (≤512): idx 3, 6, 8
        // bucket 2 (≤2048): idx 1, 7 | bucket 3 (≤MAX): idx 2, 5, 9
        let lengths = [50usize, 1500, 3000, 200, 100, 5000, 300, 800, 400, 7000];
        let chunks: Vec<IndexedChunk> = lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| make_chunk(i, len))
            .collect();

        let buckets = distribute_into_buckets(chunks);

        let sizes: Vec<usize> = buckets.iter().map(Vec::len).collect();
        assert!(
            buckets.iter().all(|b| !b.is_empty()),
            "test input must span all 4 buckets (got {sizes:?})"
        );
        assert_eq!(
            sizes.iter().sum::<usize>(),
            lengths.len(),
            "every chunk must land in exactly one bucket"
        );

        let mut flat: Vec<&IndexedChunk> = buckets.iter().flatten().collect();
        flat.sort_by_key(|c| c.global_idx);
        let recovered: Vec<usize> = flat.iter().map(|c| c.tokens.len()).collect();
        assert_eq!(
            recovered,
            lengths.to_vec(),
            "flatten + sort-by-global_idx must recover original chunk order"
        );
    }

    // T-BKT-009: empty input pipeline produces zero work
    //
    // Guards the building blocks behind `texts.is_empty() → Vec::new()` in
    // `embed_documents_batch_chunked`: even if the early return were removed,
    // an empty `all_chunk_tokens` would yield all-empty buckets, the forward
    // loop would not execute, and the `Vec<Option<Vec<f32>>>` (size 0) would
    // collect to `Vec::new()`. MLX-free proxy for the end-to-end contract.
    #[test]
    fn t_bkt_009_empty_input_zero_subbatches() {
        let indexed = build_indexed_chunks(Vec::new(), &[]);
        assert!(
            indexed.is_empty(),
            "empty chunk tokens → empty IndexedChunk vec"
        );

        let buckets = distribute_into_buckets(indexed);
        assert!(
            buckets.iter().all(Vec::is_empty),
            "empty input → every bucket stays empty"
        );
    }

    // T-BKT-008: same-doc chunks cluster contiguously inside each bucket after
    // the in-loop sort, with chunk_in_doc order preserved.
    //
    // Uses `IndexedChunk::doc_order_key` — the same function the forward loop
    // in `embed_documents_batch_chunked` calls — so this test exercises the
    // production sort contract rather than a hand-rolled mirror.
    #[test]
    fn t_bkt_008_same_doc_chunks_cluster_after_sort() {
        // doc 0 produces 3 chunks split across bucket 0 (×2) and bucket 2 (×1).
        // Inputs are shuffled so the sort must actively reorder. A no-op sort
        // would leave bucket 0 as [doc1, doc0_c0, doc2, doc0_c1] and fail.
        let input = vec![
            make_chunk_at(3, 1, 0, 60),
            make_chunk_at(0, 0, 0, 50),
            make_chunk_at(2, 0, 2, 1000),
            make_chunk_at(4, 2, 0, 70),
            make_chunk_at(1, 0, 1, 100),
        ];

        let mut buckets = distribute_into_buckets(input);
        for bucket in &mut buckets {
            bucket.sort_by_key(IndexedChunk::doc_order_key);
        }

        let b0: Vec<(usize, usize)> = buckets[0].iter().map(IndexedChunk::doc_order_key).collect();
        assert_eq!(
            b0,
            vec![(0, 0), (0, 1), (1, 0), (2, 0)],
            "bucket 0: doc 0 chunks contiguous in chunk_in_doc order, then doc 1, doc 2"
        );

        let b2: Vec<(usize, usize)> = buckets[2].iter().map(IndexedChunk::doc_order_key).collect();
        assert_eq!(
            b2,
            vec![(0, 2)],
            "bucket 2: only doc 0's third chunk (the one that overflowed the smaller buckets)"
        );
    }

    // T-012 / FR-002a / AC-1
    //
    // [T-012] Happy path: `flat.len() == batch * hidden`. `split_pooled`
    // returns `batch` rows, each `hidden` long, with values preserved in
    // row-major order. This is the contract Phase 3b's `forward_sub_batch`
    // and `embed_query_truncated` rely on after the GPU pool reduces
    // readback to `batch * hidden` floats.
    #[test]
    fn t_012_split_pooled_happy_path_preserves_row_major_order() {
        let flat: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, 7.0, // row 1
        ];
        let split = split_pooled(&flat, 2, 4).expect("happy path must split ok");

        assert_eq!(split.len(), 2, "[T-012] outer Vec must have `batch` rows");
        assert_eq!(
            split[0],
            vec![0.0, 1.0, 2.0, 3.0],
            "[T-012] row 0 preserved"
        );
        assert_eq!(
            split[1],
            vec![4.0, 5.0, 6.0, 7.0],
            "[T-012] row 1 preserved"
        );
    }

    // T-012 / FR-002a / AC-1
    //
    // [T-012] Shape mismatch (short): `flat.len() = 7` with `batch = 2,
    // hidden = 4` (expected 8). Spec scenario verbatim — exercises the
    // whole point of the helper, which is to fail fast rather than silently
    // slice an incomplete final row.
    #[test]
    fn t_012_split_pooled_shape_mismatch_short_returns_buffer_shape_mismatch() {
        let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // len 7
        match split_pooled(&flat, 2, 4) {
            Err(EmbedError::BufferShapeMismatch { expected, actual }) => {
                assert_eq!(expected, 8, "[T-012] expected = batch * hidden = 8");
                assert_eq!(actual, 7, "[T-012] actual = flat.len() = 7");
            }
            other => panic!(
                "[T-012] expected Err(BufferShapeMismatch {{ expected: 8, actual: 7 }}), \
                 got {other:?}"
            ),
        }
    }

    // T-012 / FR-002a / AC-1
    //
    // [T-012] Shape mismatch (long): `flat.len() = 9` with `batch = 2,
    // hidden = 4` (expected 8). Regression guard against a future
    // implementer that reads the first `batch * hidden` floats and silently
    // drops the tail — short-direction tests alone would not catch that.
    #[test]
    fn t_012_split_pooled_shape_mismatch_long_returns_buffer_shape_mismatch() {
        let flat: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0]; // len 9
        match split_pooled(&flat, 2, 4) {
            Err(EmbedError::BufferShapeMismatch { expected, actual }) => {
                assert_eq!(expected, 8, "[T-012] expected = batch * hidden = 8");
                assert_eq!(actual, 9, "[T-012] actual = flat.len() = 9");
            }
            other => panic!(
                "[T-012] expected Err(BufferShapeMismatch {{ expected: 8, actual: 9 }}), \
                 got {other:?}"
            ),
        }
    }

    // T-012 / FR-002a / AC-1
    //
    // [T-012] Zero-length boundary: `batch = 0, hidden = N, flat = &[]`.
    // Expected length `0 * hidden = 0` matches `flat.len() = 0`, so the
    // contract is satisfied and `split_pooled` returns `Ok(vec![])`.
    // Documents the "empty input is not an error" branch so a future
    // implementer cannot accidentally treat zero as the mismatch case.
    #[test]
    fn t_012_split_pooled_zero_batch_returns_empty_vec() {
        let flat: Vec<f32> = Vec::new();
        let split = split_pooled(&flat, 0, 768).expect("zero-batch flat must split ok");
        assert!(
            split.is_empty(),
            "[T-012] batch=0 must yield an empty outer Vec, got {split:?}"
        );
    }

    // T-012 / FR-002a / AC-1
    //
    // [T-012] Single-batch case used by `embed_query_truncated`. After the
    // Phase 3b rewire, the query path calls `split_pooled(flat, 1, hidden)`
    // and unwraps the single inner row. Verifies the helper does not require
    // `batch >= 2`.
    #[test]
    fn t_012_split_pooled_single_batch_for_embed_query_path() {
        let flat: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let split = split_pooled(&flat, 1, 5).expect("single-batch must split ok");

        assert_eq!(split.len(), 1, "[T-012] batch=1 yields a single inner row");
        assert_eq!(
            split[0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            "[T-012] inner row must equal the full flat buffer"
        );
    }

    // T-014 / FR-002b / AC-1
    //
    // [T-014] Compile-time signature lock for `pool_output`. Mirrors T-004
    // in pooling.rs which guards `gpu_pool_and_normalize` against a future
    // refactor that relaxes `output: Array` to `&Array` (which would
    // defeat the drop-before-clear contract carried by
    // `release_inference_output`). `pool_output` is the layer above and
    // must not relax the same contract — relaxing it here would re-expose
    // the same regression vector at the higher abstraction.
    //
    // The coercion also pins the **return** type as owned `Array` (not
    // `&Array`); a refactor returning `Result<&Array, _>` would similarly
    // defeat `release_inference_output(pooled)` and is caught by the same
    // line below.
    #[test]
    fn t_014_pool_output_signature_consumes_output_by_value() {
        let _coerce: fn(Array, &[u32], i32, i32) -> Result<Array, EmbedError> = pool_output;
    }

    // T-015 / FR-002c / AC-1
    //
    // [T-015] `split_pooled` rejects `NaN` with `NonFiniteOutput`. The
    // `is_finite` guard catches non-finite outputs that Phase 3b would
    // otherwise miss: the all-zero-mask `0/0` source is
    // already rejected by `validate_attention_mask` upstream, but other
    // sources (corrupt weights, kernel overflow) are not. The check runs
    // on the already-readback flat buffer so it does not defeat the
    // readback-free hot path (ADR 0002 primary lever).
    #[test]
    fn t_015_split_pooled_rejects_nan_with_non_finite_output() {
        let flat: Vec<f32> = vec![0.0, 1.0, f32::NAN, 3.0];
        match split_pooled(&flat, 1, 4) {
            Err(EmbedError::NonFiniteOutput) => {}
            other => panic!("[T-015] expected Err(NonFiniteOutput), got {other:?}"),
        }
    }

    // T-015 / FR-002c / AC-1
    //
    // [T-015] `split_pooled` rejects `±Inf` with `NonFiniteOutput`. Same
    // safety-net contract as the NaN case; covers the kernel-overflow
    // failure mode separately so a regression that handles only `NaN` is
    // caught.
    #[test]
    fn t_015_split_pooled_rejects_positive_inf_with_non_finite_output() {
        let flat: Vec<f32> = vec![0.0, f32::INFINITY, 2.0, 3.0];
        match split_pooled(&flat, 1, 4) {
            Err(EmbedError::NonFiniteOutput) => {}
            other => panic!("[T-015] expected Err(NonFiniteOutput), got {other:?}"),
        }
    }

    // T-015 / FR-002c / AC-1
    //
    // [T-015] `split_pooled` also rejects `-Inf` (not just positive
    // infinity). f32::is_finite returns false for both — guarding the
    // assumption explicitly so a future check that uses `> f32::MAX`
    // alone would fail.
    #[test]
    fn t_015_split_pooled_rejects_negative_inf_with_non_finite_output() {
        let flat: Vec<f32> = vec![0.0, 1.0, 2.0, f32::NEG_INFINITY];
        match split_pooled(&flat, 1, 4) {
            Err(EmbedError::NonFiniteOutput) => {}
            other => panic!("[T-015] expected Err(NonFiniteOutput), got {other:?}"),
        }
    }
}
