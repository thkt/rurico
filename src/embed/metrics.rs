//! Phase timing and batch shape counters for the embed pipeline.
//!
//! One `log::debug!` line per `embed_*` call so that bottlenecks can be read
//! from a run log without extra infrastructure.

use std::time::Duration;

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

    /// Emit one structured debug line summarising this call.
    pub(super) fn log(&self) {
        log::debug!(
            "embed[{kind}] \
             tokenize_ms={tokenize} chunk_plan_ms={plan} forward_eval_ms={forward} \
             readback_pool_ms={readback} cache_clear_ms={clear} \
             real_tokens={real} padded_tokens={padded} padding_ratio={ratio:.3} \
             num_chunks={chunks} batch_size={batch} max_seq_len={max_seq}",
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
        );
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
}
