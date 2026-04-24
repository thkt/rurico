//! Phase timing and batch shape counters for the embed pipeline.
//!
//! One `log::debug!` line per `embed_*` call so that bottlenecks can be read
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

    fn format_log(&self) -> String {
        format!(
            "embed[{kind}] \
             tokenize_ms={tokenize} chunk_plan_ms={plan} forward_eval_ms={forward} \
             readback_pool_ms={readback} cache_clear_ms={clear} \
             real_tokens={real} padded_tokens={padded} padding_ratio={ratio:.3} \
             num_chunks={chunks} batch_size={batch} max_seq_len={max_seq} \
             bucket_hist=[{h0},{h1},{h2},{h3}]",
            kind = self.kind,
            tokenize = self.tokenize.as_millis(),
            plan = self.chunk_plan.as_millis(),
            forward = self.forward_eval.as_millis(),
            readback = self.readback_pool.as_millis(),
            clear = self.cache_clear.as_millis(),
            real = self.real_tokens,
            padded = self.padded_tokens,
            ratio = self.padding_ratio(),
            chunks = self.num_chunks,
            batch = self.batch_size,
            max_seq = self.max_seq_len,
            h0 = self.bucket_hist[0],
            h1 = self.bucket_hist[1],
            h2 = self.bucket_hist[2],
            h3 = self.bucket_hist[3],
        )
    }

    /// Emit one structured debug line summarising this call.
    pub(super) fn log(&self) {
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("{}", self.format_log());
        }
    }
}

/// `padded / real`. Returns `0.0` when `real == 0` to avoid division by zero.
pub(super) fn padding_ratio(real: usize, padded: usize) -> f32 {
    if real == 0 {
        return 0.0;
    }
    padded as f32 / real as f32
}

#[cfg(test)]
mod tests {
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

    // T-MET-001: format_log includes bucket_hist in the expected schema so
    // smoke logs expose chunk-length distribution from a single line (AC-9).
    #[test]
    fn t_met_001_format_log_contains_bucket_hist() {
        let mut m = PhaseMetrics::new("batch");
        m.bucket_hist = [3, 5, 2, 1];
        m.num_chunks = 11;
        let line = m.format_log();
        assert!(
            line.contains("bucket_hist=[3,5,2,1]"),
            "log line must match spec schema [N0,N1,N2,N3] (got: {line})"
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
