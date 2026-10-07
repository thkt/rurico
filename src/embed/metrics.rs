//! Phase timing and batch shape counters for the embed pipeline.
//!
//! One `tracing::debug!` line per `embed_*` call so that bottlenecks can be read
//! from a run log without extra infrastructure.
//!
//! [`PhaseMetrics`] is the internal accumulator, `pub(super)` so only the
//! `embed` module tree mutates it. [`BatchMetrics`] is the public
//! downstream-facing snapshot returned by
//! [`Embedder::embed_documents_batch_with_metrics`](super::Embedder::embed_documents_batch_with_metrics)
//! — it carries the same numbers in millisecond integers so consumers avoid
//! coupling to `std::time::Duration` and the internal `kind` tag.

use std::time::Duration;

use super::ChunkedEmbedding;

/// Output and optional telemetry from the same, single batch invocation.
#[derive(Debug)]
pub struct MeasuredEmbedding {
    /// Embeddings in the same order as the input documents.
    pub embeddings: Vec<ChunkedEmbedding>,
    /// `None` means the provider does not support measurement, not zero cost.
    pub metrics: Option<InferenceMetrics>,
}

/// Actual padded input shape of one completed forward, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ForwardShape {
    /// Number of chunk rows passed to the model.
    pub batch_size: usize,
    /// Padded bucket length passed to the model.
    pub sequence_length: usize,
}

/// Host-side timings for one successful MLX batch call, preserving sub-ms precision.
///
/// `wall` encloses lock acquisition, preprocessing, all forwards, readbacks,
/// cleanup, pauses, output reconstruction and logging. Other durations are
/// disjoint subsets of wall, not an exhaustive partition: bookkeeping, Array
/// destruction and result reconstruction are not separately timed. Do not add
/// wall to its children. These are host elapsed times, **not GPU kernel times**.
/// Errors return the original `EmbedError`, without a partial metrics snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferenceMetrics {
    /// Tokenization, chunk planning, bucket routing/sorting and CPU padding.
    pub preprocessing: Duration,
    /// Not separated from preprocessing in MLX batches; always `None` there.
    pub tokenize: Option<Duration>,
    /// Waiting for this Embedder's mutex. The global cleanup lock is instead
    /// included in `cache_clear`; it is not separately measured.
    pub lock_wait: Duration,
    /// Graph construction, forward, GPU pooling/normalization and pooled eval.
    pub forward_eval: Duration,
    /// Pooled host slice access, finite/shape validation and splitting into rows.
    pub readback: Duration,
    /// Best-effort buffer/compile cache cleanup after temporary Arrays drop,
    /// including global cache lock wait and any cleanup diagnostics.
    pub cache_clear: Duration,
    /// Actual host time spent sleeping after each completed forward, including
    /// the last forward. Zero means no sleep when no pause was requested.
    pub pause: Duration,
    /// Number of requested sleeps executed (including zero-duration requests).
    pub pause_count: usize,
    /// Elapsed duration of the measured public call through snapshot assembly.
    pub wall: Duration,
    /// One entry per completed forward; length is the forward count.
    pub forwards: Vec<ForwardShape>,
    /// Non-padding token positions across forwards.
    pub real_tokens: usize,
    /// All token positions across forwards, including padding.
    pub padded_tokens: usize,
    /// Element count at each actual host slice access, in execution order.
    /// `None` denotes legacy/unavailable telemetry; `Some([])` means no access.
    /// Counts are observed before output validation, never inferred from shape.
    #[serde(default)]
    pub readback_elements: Option<Vec<usize>>,
    /// Total number of output chunks.
    pub num_chunks: usize,
    /// Chunk counts in the fixed 128/512/2048/8192 buckets.
    pub bucket_hist: [usize; 4],
}

impl From<PhaseMetrics> for InferenceMetrics {
    fn from(m: PhaseMetrics) -> Self {
        Self {
            preprocessing: m.preprocessing,
            tokenize: None,
            forward_eval: m.forward_eval,
            readback: m.readback_pool,
            cache_clear: m.cache_clear,
            pause: m.pause,
            pause_count: m.pause_count,
            forwards: m.forwards,
            readback_elements: Some(m.readback_elements),
            real_tokens: m.real_tokens,
            padded_tokens: m.padded_tokens,
            num_chunks: m.num_chunks,
            bucket_hist: m.bucket_hist,
            ..Self::default()
        }
    }
}

/// Identifies which embed entry point a phase record or warn emit came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum EmbedKind {
    #[default]
    Query,
    Batch,
}

impl EmbedKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Batch => "batch",
        }
    }
}

/// Public batch-level metrics snapshot mirroring the `"batch"` `PhaseMetrics`
/// record. Returned alongside embeddings by
/// [`Embedder::embed_documents_batch_with_metrics`](super::Embedder::embed_documents_batch_with_metrics)
/// so callers (smoke harness, downstream indexers) can observe padding,
/// linearity, and bucket distribution without parsing a debug-log line.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BatchMetrics {
    /// `padded_tokens / real_tokens` — 1.0 means zero padding overhead.
    pub padding_ratio: f32,
    /// Tokens whose attention mask is non-zero (real work).
    pub real_tokens: usize,
    /// Total positions processed, including padding.
    pub padded_tokens: usize,
    /// Wall-clock of the forward + eval phase in milliseconds.
    pub forward_eval_ms: u128,
    /// Legacy batch placeholder: zero means tokenization was not separated.
    /// Use [`InferenceMetrics::tokenize`] to distinguish unavailable measurement.
    pub tokenize_ms: u128,
    /// Legacy chunk-planning time, including tokenization, truncated to ms.
    pub chunk_plan_ms: u128,
    /// Number of chunks produced across all input texts.
    pub num_chunks: usize,
    /// Chunk count per length bucket (indexed by `assign_bucket`).
    pub bucket_hist: [usize; 4],
    /// Largest `max_seq_len` observed across sub-batches.
    pub max_seq_len: usize,
    /// Largest sub-batch size observed.
    pub batch_size: usize,
}

impl From<&PhaseMetrics> for BatchMetrics {
    fn from(m: &PhaseMetrics) -> Self {
        Self {
            padding_ratio: m.padding_ratio(),
            real_tokens: m.real_tokens,
            padded_tokens: m.padded_tokens,
            forward_eval_ms: m.forward_eval.as_millis(),
            tokenize_ms: m.tokenize.as_millis(),
            chunk_plan_ms: m.chunk_plan.as_millis(),
            num_chunks: m.num_chunks,
            bucket_hist: m.bucket_hist,
            max_seq_len: m.max_seq_len,
            batch_size: m.batch_size,
        }
    }
}

/// Phase timings and batch counters for one `embed_*` call.
#[derive(Debug, Clone, Default)]
pub(super) struct PhaseMetrics {
    /// Identifies which entry point produced this record.
    pub kind: EmbedKind,
    pub preprocessing: Duration,
    pub pause: Duration,
    /// Number of requested sleeps executed (including zero-duration requests).
    pub pause_count: usize,
    pub forwards: Vec<ForwardShape>,
    pub tokenize: Duration,
    pub chunk_plan: Duration,
    pub forward_eval: Duration,
    pub readback_pool: Duration,
    pub readback_elements: Vec<usize>,
    pub cache_clear: Duration,
    /// Number of tokens whose attention mask is non-zero (real work).
    pub real_tokens: usize,
    /// Total positions processed, including padding. Equals the sum of
    /// `batch_size × max_seq_len` across all sub-batches.
    pub padded_tokens: usize,
    pub num_chunks: usize,
    /// Largest sub-batch size observed in this call.
    pub batch_size: usize,
    /// Largest `max_seq_len` observed in this call.
    pub max_seq_len: usize,
    /// Chunk count per length bucket, indexed by `assign_bucket`. The sum
    /// equals `num_chunks`; exposes length distribution from a single log line.
    pub bucket_hist: [usize; 4],
}

impl PhaseMetrics {
    pub(super) fn new(kind: EmbedKind) -> Self {
        Self {
            kind,
            ..Self::default()
        }
    }

    /// `padded_tokens / real_tokens` — a value of 1.0 means no padding.
    pub(super) fn padding_ratio(&self) -> f32 {
        padding_ratio(self.real_tokens, self.padded_tokens)
    }

    /// Emit one structured debug record summarising this call.
    ///
    /// Each phase timing and counter is a named field so subscribers can
    /// filter or aggregate without parsing a format string.
    pub(super) fn log(&self) {
        tracing::debug!(
            kind = self.kind.as_str(),
            tokenize_ms = self.tokenize.as_millis(),
            chunk_plan_ms = self.chunk_plan.as_millis(),
            forward_eval_ms = self.forward_eval.as_millis(),
            readback_pool_ms = self.readback_pool.as_millis(),
            cache_clear_ms = self.cache_clear.as_millis(),
            real_tokens = self.real_tokens,
            padded_tokens = self.padded_tokens,
            padding_ratio = self.padding_ratio(),
            num_chunks = self.num_chunks,
            batch_size = self.batch_size,
            max_seq_len = self.max_seq_len,
            bucket_hist_0 = self.bucket_hist[0],
            bucket_hist_1 = self.bucket_hist[1],
            bucket_hist_2 = self.bucket_hist[2],
            bucket_hist_3 = self.bucket_hist[3],
            "embed phase metrics",
        );
    }
}

/// `padded / real`. Returns `0.0` when `real == 0` to avoid division by zero.
///
/// Computed in f64 to preserve precision when token counts exceed 2^24
/// (production batches at MAX_SEQ_LEN × TOKEN_BUDGET approach this range);
/// the f64→f32 narrowing is safe because the ratio is bounded near 1.0–10.0.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub(super) fn padding_ratio(real: usize, padded: usize) -> f32 {
    if real == 0 {
        return 0.0;
    }
    let ratio = padded as f64 / real as f64;
    ratio as f32
}

#[cfg(test)]
mod tests;
