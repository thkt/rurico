//! Retrieval pipeline contract (ADR 0004, Issue #67 / Phase 3).
//!
//! Plug-in points frozen by ADR 0004:
//! - Stage 3 aggregation: [`Aggregator`] trait between merge and rerank.
//!   Default impl is [`IdentityAggregator`].
//!
//! Concrete strategies ([`MaxChunkAggregator`], [`DedupeAggregator`],
//! [`TopKAverageAggregator`]) live alongside the trait; downstream crates can
//! also supply their own `impl Aggregator` for domain-specific dedupers.
//!
//! ## Pipeline shape note
//!
//! `rrf_merge` (`src/storage/search.rs`) fuses ranks via a `HashMap` keyed by
//! `doc_id`, so its output already carries unique identifiers. With the
//! current pipeline that indexes one chunk per `EvalDocument`, every
//! non-identity aggregator therefore behaves as identity on the eval baseline.
//! Strategy correctness is validated via synthetic multi-hit unit tests;
//! non-vacuous evaluation arrives once chunk-level retrieval lands (parent-
//! child / `chunk_id` follow-up).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Single candidate after Stage 2 (RRF merge) — input/output for [`Aggregator`].
///
/// Score sign is "higher is better" (post-RRF fused score). Stage 3 may
/// rewrite, drop, or re-order entries; the resulting `Vec` becomes the input
/// to Stage 4 (rerank). Also serves as the public ranked-hit type returned
/// from `eval::pipeline::QueryResult` — Serialize/Deserialize keep the
/// pipeline output JSON shape unchanged across the merge boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedHit {
    /// Document or chunk identifier carried through Stage 1+2.
    pub doc_id: String,
    /// Aggregated relevance score (RRF score).
    pub score: f64,
}

/// Stage 3 hook: aggregate / dedupe / re-order [`MergedHit`]s before Stage 4.
///
/// ADR 0004 fixes the position (between Stage 2 merge and Stage 4 rerank) and
/// the surface; this trait is the only stable extension point. Granularity
/// (chunk vs document) is strategy-specific — implementations document their
/// own dedupe / collapse contract.
pub trait Aggregator {
    /// Aggregate `hits` into the form Stage 4 expects.
    ///
    /// **Output MUST be sorted by `score` descending** (with deterministic
    /// tiebreaking by `doc_id` when scores are equal). The pipeline truncates
    /// the result to `config.k` after this call, so non-sorted output would
    /// silently drop higher-scoring hits. Output length may be ≤ input
    /// length (dedupe / max-chunk collapse duplicates).
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit>;
}

/// Identity aggregator — returns the input unchanged.
///
/// Default for the reference pipeline composition; preserves the pre-Phase-3
/// behaviour where each `EvalDocument` produces exactly one hit. The sort
/// invariant is upheld by pass-through because Stage 2 RRF merge already
/// emits score-descending output.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityAggregator;

impl Aggregator for IdentityAggregator {
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit> {
        hits.to_vec()
    }
}

/// Same-document max-score aggregator — keeps the maximum score per `doc_id`,
/// then re-orders by score descending (ties broken by `doc_id` ascending).
#[derive(Debug, Default, Clone, Copy)]
pub struct MaxChunkAggregator;

impl Aggregator for MaxChunkAggregator {
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit> {
        let mut best: HashMap<&str, f64> = HashMap::new();
        for hit in hits {
            let entry = best.entry(hit.doc_id.as_str()).or_insert(f64::MIN);
            if hit.score > *entry {
                *entry = hit.score;
            }
        }
        let mut output: Vec<MergedHit> = best
            .into_iter()
            .map(|(id, score)| MergedHit {
                doc_id: id.to_owned(),
                score,
            })
            .collect();
        output.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        output
    }
}

/// First-occurrence dedupe aggregator — preserves the rank-order of the first
/// hit per `doc_id`, drops subsequent duplicates.
///
/// Unlike [`MaxChunkAggregator`] this keeps the first hit's score (typically
/// the higher rank from RRF), making it a structural dedupe rather than a
/// score-aware collapse. The sort invariant is upheld because Stage 2 RRF
/// merge already emits score-descending output and dedupe drops later
/// duplicates without re-ordering.
#[derive(Debug, Default, Clone, Copy)]
pub struct DedupeAggregator;

impl Aggregator for DedupeAggregator {
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit> {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        let mut output = Vec::with_capacity(hits.len());
        for hit in hits {
            if seen.insert(hit.doc_id.as_str(), ()).is_none() {
                output.push(hit.clone());
            }
        }
        output
    }
}

/// Top-k average aggregator — for each `doc_id`, averages its highest `k`
/// scores. Output is sorted by aggregate score descending.
#[derive(Debug, Clone, Copy)]
pub struct TopKAverageAggregator {
    /// Number of top-scoring hits per `doc_id` to include in the average. A
    /// `doc_id` with fewer than `k` hits averages over all of them. `k = 0`
    /// returns an empty output.
    pub k: usize,
}

impl TopKAverageAggregator {
    /// Construct with the given top-k cutoff.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Aggregator for TopKAverageAggregator {
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit> {
        if self.k == 0 {
            return Vec::new();
        }
        let mut buckets: HashMap<&str, Vec<f64>> = HashMap::new();
        for hit in hits {
            buckets
                .entry(hit.doc_id.as_str())
                .or_default()
                .push(hit.score);
        }
        let mut output: Vec<MergedHit> = buckets
            .into_iter()
            .map(|(id, mut scores)| {
                scores.sort_by(|a, b| b.total_cmp(a));
                scores.truncate(self.k);
                #[allow(clippy::cast_precision_loss)]
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                MergedHit {
                    doc_id: id.to_owned(),
                    score: mean,
                }
            })
            .collect();
        output.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f64) -> MergedHit {
        MergedHit {
            doc_id: id.to_owned(),
            score,
        }
    }

    // T-067-001: identity_returns_input_unchanged
    #[test]
    fn identity_returns_input_unchanged() {
        let aggregator = IdentityAggregator;
        let input = vec![hit("d1", 0.9), hit("d2", 0.7), hit("d1", 0.5)];
        let output = aggregator.aggregate(&input);
        assert_eq!(output, input, "Identity must preserve input verbatim");
    }

    // T-067-002: max_chunk_collapses_duplicates_to_max_score
    #[test]
    fn max_chunk_collapses_duplicates_to_max_score() {
        let aggregator = MaxChunkAggregator;
        let input = vec![
            hit("d1", 0.5),
            hit("d2", 0.7),
            hit("d1", 0.9),
            hit("d3", 0.6),
        ];
        let output = aggregator.aggregate(&input);
        assert_eq!(
            output,
            vec![hit("d1", 0.9), hit("d2", 0.7), hit("d3", 0.6)],
            "MaxChunk should keep d1=0.9 (max) and sort by score desc"
        );
    }

    // T-067-003: dedupe_keeps_first_occurrence_per_doc_id
    #[test]
    fn dedupe_keeps_first_occurrence_per_doc_id() {
        let aggregator = DedupeAggregator;
        let input = vec![
            hit("d1", 0.9),
            hit("d2", 0.7),
            hit("d1", 0.5),
            hit("d3", 0.6),
            hit("d2", 0.4),
        ];
        let output = aggregator.aggregate(&input);
        assert_eq!(
            output,
            vec![hit("d1", 0.9), hit("d2", 0.7), hit("d3", 0.6)],
            "Dedupe should keep input rank-order and drop later duplicates"
        );
    }

    // T-067-004: topk_average_averages_top_k_per_doc_id
    //
    // Uses IEEE-754-exact dyadic fractions (1.0, 0.5, 0.25, 0.125) so the
    // observed averages compare bit-equal to the expected values.
    #[test]
    fn topk_average_averages_top_k_per_doc_id() {
        let aggregator = TopKAverageAggregator::new(2);
        let input = vec![
            hit("d1", 1.0),
            hit("d1", 0.5),
            hit("d1", 0.25),
            hit("d2", 0.5),
            hit("d2", 0.25),
            hit("d3", 0.125),
        ];
        let output = aggregator.aggregate(&input);
        // d1 top-2 = (1.0 + 0.5) / 2  = 0.75
        // d2 top-2 = (0.5 + 0.25) / 2 = 0.375
        // d3 top-1 = 0.125            = 0.125
        assert_eq!(
            output,
            vec![hit("d1", 0.75), hit("d2", 0.375), hit("d3", 0.125)],
            "TopK avg should take top-k scores per doc_id and sort by mean desc"
        );
    }

    // T-067-005: topk_average_with_zero_k_returns_empty
    #[test]
    fn topk_average_with_zero_k_returns_empty() {
        let aggregator = TopKAverageAggregator::new(0);
        let input = vec![hit("d1", 0.9), hit("d2", 0.7)];
        let output = aggregator.aggregate(&input);
        assert!(output.is_empty(), "k=0 should return empty");
    }

    // T-067-006: max_chunk_unique_input_only_resorts
    //
    // Documents the structural-identity fact called out in the module doc:
    // when input doc_ids are already unique (current pipeline shape), the
    // observable difference between MaxChunk and Identity is only the sort.
    #[test]
    fn max_chunk_unique_input_only_resorts() {
        let aggregator = MaxChunkAggregator;
        let input = vec![hit("d1", 0.9), hit("d2", 0.7), hit("d3", 0.5)];
        let output = aggregator.aggregate(&input);
        assert_eq!(
            output, input,
            "MaxChunk on already-unique input must equal input"
        );
    }
}
