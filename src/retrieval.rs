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

/// Source of a Stage 1 candidate hit (ADR 0004 Stage 1 output).
///
/// Closed enum: misspelled labels become compile errors. Phase 6 (#70) may
/// add `PrefixEnsemble` variants when the prefix-fanout retrieval lands.
///
/// Used as a `HashMap` key in [`MergedHit::source_scores`] — `lowercase`
/// rename-all keeps JSON round-trip stable (`"fts"`, `"vector"`).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CandidateSource {
    /// Full-text-search candidate (FTS5 / trigram).
    Fts,
    /// Vector-similarity candidate (sqlite-vec).
    Vector,
}

/// Single Stage 1 retrieval candidate (ADR 0004 Stage 1).
///
/// Source-tagged so Stage 2 merge can apply per-source weighting and
/// preserve `source_scores` into [`MergedHit`]. `rank` is 0-based (0 = best
/// per source); `score` is the raw per-source score (BM25 for FTS, distance
/// for vector — sign convention depends on source).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Originating retrieval source.
    pub source: CandidateSource,
    /// Document or chunk identifier.
    pub doc_id: String,
    /// Raw per-source score (BM25, distance, etc.). Sign convention is
    /// source-specific — Stage 2 fuses by `rank`, not by `score`.
    pub score: f64,
    /// 0-based rank within the source's result list (0 = best).
    pub rank: usize,
}

/// Single candidate after Stage 2 (RRF merge) — input/output for [`Aggregator`].
///
/// Score sign is "higher is better" (post-RRF fused score). Stage 3 may
/// rewrite, drop, or re-order entries; the resulting `Vec` becomes the input
/// to Stage 4 (rerank). Also serves as the public ranked-hit type returned
/// from `eval::pipeline::QueryResult` — Serialize/Deserialize keep the
/// pipeline output JSON shape unchanged across the merge boundary.
///
/// `source_scores` records per-source contributions so downstream UIs can
/// display score breakdown and debugging surfaces can attribute fusion
/// outcomes. It is `serde(default)` for backward-compat with pre-Phase-4
/// JSON fixtures that omit the field; an empty map represents "source
/// information unavailable" rather than "no source contributed".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedHit {
    /// Document or chunk identifier carried through Stage 1+2.
    pub doc_id: String,
    /// Aggregated relevance score (RRF score).
    pub score: f64,
    /// Per-source contributions to the fused score. Empty when source
    /// information is not preserved through the pipeline.
    #[serde(default)]
    pub source_scores: HashMap<CandidateSource, f64>,
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
        let mut best: HashMap<&str, &MergedHit> = HashMap::new();
        for hit in hits {
            best.entry(hit.doc_id.as_str())
                .and_modify(|cur| {
                    if hit.score > cur.score {
                        *cur = hit;
                    }
                })
                .or_insert(hit);
        }
        let mut output: Vec<MergedHit> = best.into_values().cloned().collect();
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
        let mut buckets: HashMap<&str, Vec<&MergedHit>> = HashMap::new();
        for hit in hits {
            buckets.entry(hit.doc_id.as_str()).or_default().push(hit);
        }
        let mut output: Vec<MergedHit> = buckets
            .into_iter()
            .map(|(id, mut bucket)| {
                bucket.sort_by(|a, b| b.score.total_cmp(&a.score));
                bucket.truncate(self.k);
                #[allow(clippy::cast_precision_loss)]
                let len = bucket.len() as f64;
                let mean = bucket.iter().map(|h| h.score).sum::<f64>() / len;
                let mut source_scores: HashMap<CandidateSource, f64> = HashMap::new();
                for hit in &bucket {
                    for (src, val) in &hit.source_scores {
                        *source_scores.entry(*src).or_default() += val;
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                for value in source_scores.values_mut() {
                    *value /= len;
                }
                MergedHit {
                    doc_id: id.to_owned(),
                    score: mean,
                    source_scores,
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
            source_scores: HashMap::new(),
        }
    }

    fn hit_with_sources(id: &str, score: f64, sources: &[(CandidateSource, f64)]) -> MergedHit {
        MergedHit {
            doc_id: id.to_owned(),
            score,
            source_scores: sources.iter().copied().collect(),
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

    // T-068-001: identity_preserves_source_scores
    #[test]
    fn identity_preserves_source_scores() {
        let aggregator = IdentityAggregator;
        let input = vec![hit_with_sources(
            "d1",
            0.9,
            &[(CandidateSource::Fts, 0.6), (CandidateSource::Vector, 0.3)],
        )];
        let output = aggregator.aggregate(&input);
        assert_eq!(
            output[0].source_scores, input[0].source_scores,
            "Identity must preserve source_scores verbatim"
        );
    }

    // T-068-002: max_chunk_keeps_max_scoring_hits_source_scores
    #[test]
    fn max_chunk_keeps_max_scoring_hits_source_scores() {
        let aggregator = MaxChunkAggregator;
        let input = vec![
            hit_with_sources("d1", 0.5, &[(CandidateSource::Fts, 0.5)]),
            hit_with_sources("d1", 0.9, &[(CandidateSource::Vector, 0.9)]),
        ];
        let output = aggregator.aggregate(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].source_scores.get(&CandidateSource::Vector),
            Some(&0.9),
            "MaxChunk must keep source_scores from the max-scoring hit (the Vector contribution)"
        );
        assert!(
            !output[0].source_scores.contains_key(&CandidateSource::Fts),
            "MaxChunk must drop source_scores from non-winning hits"
        );
    }

    // T-068-003: dedupe_keeps_first_hits_source_scores
    #[test]
    fn dedupe_keeps_first_hits_source_scores() {
        let aggregator = DedupeAggregator;
        let input = vec![
            hit_with_sources("d1", 0.9, &[(CandidateSource::Fts, 0.9)]),
            hit_with_sources("d1", 0.5, &[(CandidateSource::Vector, 0.5)]),
        ];
        let output = aggregator.aggregate(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].source_scores.get(&CandidateSource::Fts),
            Some(&0.9),
            "Dedupe must keep first occurrence's source_scores"
        );
    }

    // T-068-004: topk_average_averages_source_scores
    //
    // Uses dyadic fractions so the averaged source_scores compare bit-equal.
    #[test]
    fn topk_average_averages_source_scores() {
        let aggregator = TopKAverageAggregator::new(2);
        let input = vec![
            hit_with_sources("d1", 1.0, &[(CandidateSource::Fts, 1.0)]),
            hit_with_sources("d1", 0.5, &[(CandidateSource::Fts, 0.5)]),
        ];
        let output = aggregator.aggregate(&input);
        assert_eq!(output.len(), 1);
        // (1.0 + 0.5) / 2 = 0.75
        assert_eq!(
            output[0].source_scores.get(&CandidateSource::Fts),
            Some(&0.75),
            "TopKAverage must average source_scores across the top-k hits"
        );
    }
}
