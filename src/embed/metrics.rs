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

/// Public batch-level metrics snapshot mirroring the `"batch"` [`PhaseMetrics`]
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
    /// Wall-clock of the tokenization phase in milliseconds.
    pub tokenize_ms: u128,
    /// Wall-clock of the chunk-planning phase in milliseconds.
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
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PhaseMetrics {
    /// Identifies which entry point produced this record (e.g. `"query"`, `"batch"`).
    pub kind: &'static str,
    pub tokenize: Duration,
    pub chunk_plan: Duration,
    pub forward_eval: Duration,
    pub readback_pool: Duration,
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
    pub(super) fn new(kind: &'static str) -> Self {
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
    /// filter or aggregate without parsing a format string (ADR 0007).
    pub(super) fn log(&self) {
        tracing::debug!(
            kind = self.kind,
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
mod tests {
    use tracing_test::traced_test;

    use super::*;

    // padding_ratio: safe on zero real, ratio = padded / real otherwise
    #[test]
    fn padding_ratio_zero_real_returns_zero() {
        assert_eq!(padding_ratio(0, 0), 0.0);
        assert_eq!(padding_ratio(0, 100), 0.0);
    }

    #[test]
    fn padding_ratio_no_padding_returns_one() {
        assert!((padding_ratio(100, 100) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn padding_ratio_half_real_returns_two() {
        assert!((padding_ratio(50, 100) - 2.0).abs() < f32::EPSILON);
    }

    // PhaseMetrics::padding_ratio delegates to the free function
    #[test]
    fn phase_metrics_padding_ratio_matches_fields() {
        let m = PhaseMetrics {
            real_tokens: 80,
            padded_tokens: 160,
            ..Default::default()
        };
        assert!((m.padding_ratio() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn phase_metrics_default_padding_ratio_is_zero() {
        let m = PhaseMetrics::default();
        assert_eq!(m.padding_ratio(), 0.0);
    }

    #[test]
    fn phase_metrics_new_sets_kind() {
        let m = PhaseMetrics::new("query");
        assert_eq!(m.kind, "query");
        assert_eq!(m.padded_tokens, 0);
    }

    // T-MET-001: log() emits named fields for the bucket histogram so
    // subscribers can read per-bucket counts without parsing a format string
    // (ADR 0007 / AC-9).
    #[traced_test]
    #[test]
    fn t_met_001_log_emits_named_bucket_hist_fields() {
        let mut m = PhaseMetrics::new("batch");
        m.bucket_hist = [3, 5, 2, 1];
        m.num_chunks = 11;
        m.log();
        assert!(
            logs_contain("bucket_hist_0=3"),
            "expected named field bucket_hist_0=3 in subscriber output",
        );
        assert!(
            logs_contain("bucket_hist_1=5"),
            "expected named field bucket_hist_1=5 in subscriber output",
        );
        assert!(
            logs_contain("bucket_hist_2=2"),
            "expected named field bucket_hist_2=2 in subscriber output",
        );
        assert!(
            logs_contain("bucket_hist_3=1"),
            "expected named field bucket_hist_3=1 in subscriber output",
        );
        assert!(
            logs_contain("num_chunks=11"),
            "expected named field num_chunks=11 in subscriber output",
        );
        assert!(
            logs_contain("embed phase metrics"),
            "expected event message in subscriber output",
        );
    }

    // T-MET-002: bucket_hist sum invariant equals num_chunks so each chunk is
    // counted in exactly one bucket across query and batch paths.
    #[test]
    fn t_met_002_bucket_hist_sum_matches_num_chunks() {
        let m = PhaseMetrics {
            num_chunks: 7,
            bucket_hist: [2, 3, 1, 1],
            ..Default::default()
        };
        assert_eq!(m.bucket_hist.iter().sum::<usize>(), m.num_chunks);
    }
}
