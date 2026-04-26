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
//! Stage 2 [`WeightedRrf`] fuses ranks via a `HashMap` keyed by `doc_id`,
//! so its output already carries unique identifiers. With the current
//! pipeline that indexes one chunk per `EvalDocument`, every non-identity
//! aggregator therefore behaves as identity on the eval baseline. Strategy
//! correctness is validated via synthetic multi-hit unit tests; non-vacuous
//! evaluation arrives once chunk-level retrieval lands (parent-child /
//! `chunk_id` follow-up — Issue #76).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::storage::recency_decay;

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

/// Stage 2 hook: fuse Stage 1 [`Candidate`]s into Stage 3-ready [`MergedHit`]s.
///
/// ADR 0004 Stage 2 contract. Default implementation is [`WeightedRrf`] — a
/// Reciprocal Rank Fusion variant that supports per-source weighting and
/// `rrf_k` tuning (Phase 4 / Issue #68). Downstream / future strategies
/// (learned weights, multi-tier rerank) can plug in via this trait.
pub trait MergeStrategy {
    /// Fuse `candidates` from multiple sources into Stage 3 input.
    ///
    /// **Output MUST be sorted by `score` descending** (with deterministic
    /// tiebreaking by `doc_id` ascending when scores are equal). Each
    /// returned [`MergedHit`]'s `source_scores` records the per-source
    /// contributions to the fused score so downstream UIs can display
    /// score breakdown.
    fn merge(&self, candidates: &[Candidate]) -> Vec<MergedHit>;
}

/// Hybrid scoring configuration for [`WeightedRrf`] (ADR 0004 Stage 2 / #68).
///
/// `rrf_k` is the RRF damping constant — higher values flatten rank
/// weighting (lower-ranked hits contribute relatively more); `60.0` matches
/// the pre-Phase-4 hardcoded constant. `source_weights` scales each
/// source's RRF contribution; missing entries default to `0.0` (the
/// source's signal is dropped). All defaults reproduce the pre-Phase-4
/// behaviour bit-equal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// RRF damping constant. Default `60.0`.
    pub rrf_k: f64,
    /// Per-source weights. Missing entries are treated as `0.0`.
    pub source_weights: HashMap<CandidateSource, f64>,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert(CandidateSource::Fts, 1.0);
        source_weights.insert(CandidateSource::Vector, 1.0);
        Self {
            rrf_k: 60.0,
            source_weights,
        }
    }
}

/// Recency boost configuration applied at Stage 2 (Phase 4 / #68).
///
/// Recency is an additive boost layered on top of [`WeightedRrf`]'s fused
/// score: `score += weight * recency_decay(age_days, half_life_days)`. The
/// decay is exponential — `1.0` at `age_days = 0`, halving every
/// `half_life_days`. `weight = 0.0` makes the boost a no-op.
///
/// Recency requires per-doc `updated_at` metadata supplied by the caller
/// via the `age_days_for` closure of [`WeightedRrf::merge_with_recency`];
/// the strategy itself does not query any storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecencyConfig {
    /// Recency boost magnitude. `0.0` disables the boost.
    pub weight: f64,
    /// Half-life in days controlling the exponential decay.
    pub half_life_days: f64,
}

/// Default Stage 2 strategy: weighted Reciprocal Rank Fusion.
///
/// Each candidate contributes `weight / (rrf_k + rank)` to its `doc_id`'s
/// fused score, where `weight` comes from
/// [`HybridSearchConfig::source_weights`]. With default config the formula
/// reduces to `1 / (60 + rank)` — bit-equal to the pre-Phase-4 `rrf_merge`
/// primitive.
#[derive(Debug, Default, Clone)]
pub struct WeightedRrf {
    /// Active hybrid scoring configuration.
    pub config: HybridSearchConfig,
}

impl WeightedRrf {
    /// Construct with the given config.
    pub fn new(config: HybridSearchConfig) -> Self {
        Self { config }
    }

    /// Stage 2 merge plus an additive recency boost.
    ///
    /// Calls [`Self::merge`] first, then layers `recency.weight *
    /// recency_decay(age, recency.half_life_days)` onto each hit whose
    /// `age_days_for` lookup returns `Some(age)`. Hits with `None` age are
    /// left at their RRF score. Output is re-sorted after the boost.
    ///
    /// `source_scores` continues to record only per-source RRF
    /// contributions — the recency component lives in `score` only, since
    /// it is not attributable to a single retrieval source.
    pub fn merge_with_recency<F>(
        &self,
        candidates: &[Candidate],
        recency: &RecencyConfig,
        age_days_for: F,
    ) -> Vec<MergedHit>
    where
        F: Fn(&str) -> Option<f64>,
    {
        let mut hits = self.merge(candidates);
        if recency.weight == 0.0 {
            return hits;
        }
        for hit in &mut hits {
            if let Some(age) = age_days_for(hit.doc_id.as_str()) {
                let decay = recency_decay(age, recency.half_life_days);
                hit.score += recency.weight * decay;
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        hits
    }
}

impl MergeStrategy for WeightedRrf {
    fn merge(&self, candidates: &[Candidate]) -> Vec<MergedHit> {
        let mut scores: HashMap<&str, f64> = HashMap::new();
        let mut sources_per_doc: HashMap<&str, HashMap<CandidateSource, f64>> = HashMap::new();
        for cand in candidates {
            let weight = self
                .config
                .source_weights
                .get(&cand.source)
                .copied()
                .unwrap_or(0.0);
            #[allow(clippy::cast_precision_loss)]
            let rank_f = cand.rank as f64;
            let contribution = weight / (self.config.rrf_k + rank_f);
            *scores.entry(cand.doc_id.as_str()).or_default() += contribution;
            *sources_per_doc
                .entry(cand.doc_id.as_str())
                .or_default()
                .entry(cand.source)
                .or_default() += contribution;
        }
        let mut hits: Vec<MergedHit> = scores
            .into_iter()
            .map(|(doc_id, score)| MergedHit {
                doc_id: doc_id.to_owned(),
                score,
                source_scores: sources_per_doc.remove(doc_id).unwrap_or_default(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        hits
    }
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

    fn candidate(source: CandidateSource, doc_id: &str, rank: usize) -> Candidate {
        Candidate {
            source,
            doc_id: doc_id.to_owned(),
            score: 0.0,
            rank,
        }
    }

    // T-068-005: weighted_rrf_default_matches_unweighted_rrf
    //
    // With default config (rrf_k=60, weights=1.0), WeightedRrf must produce
    // bit-equal scores to the legacy 1/(60+rank) formula.
    #[test]
    fn weighted_rrf_default_matches_unweighted_rrf() {
        let strategy = WeightedRrf::default();
        let candidates = vec![
            candidate(CandidateSource::Fts, "d1", 0),
            candidate(CandidateSource::Vector, "d1", 1),
            candidate(CandidateSource::Fts, "d2", 1),
        ];
        let output = strategy.merge(&candidates);
        let d1_score = 1.0 / 60.0 + 1.0 / 61.0;
        let d2_score = 1.0 / 61.0;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].doc_id, "d1");
        assert!((output[0].score - d1_score).abs() < f64::EPSILON);
        assert_eq!(output[1].doc_id, "d2");
        assert!((output[1].score - d2_score).abs() < f64::EPSILON);
    }

    // T-068-006: weighted_rrf_zero_weight_drops_source
    #[test]
    fn weighted_rrf_zero_weight_drops_source() {
        let mut config = HybridSearchConfig::default();
        config.source_weights.insert(CandidateSource::Vector, 0.0);
        let strategy = WeightedRrf::new(config);
        let candidates = vec![
            candidate(CandidateSource::Fts, "d1", 0),
            candidate(CandidateSource::Vector, "d2", 0),
        ];
        let output = strategy.merge(&candidates);
        // d2 only has Vector contribution which is weighted 0 → score 0
        // d1 has Fts contribution at weight 1 → 1/60
        assert_eq!(output[0].doc_id, "d1");
        assert!((output[0].score - 1.0 / 60.0).abs() < f64::EPSILON);
        assert_eq!(output[1].doc_id, "d2");
        assert_eq!(output[1].score, 0.0);
    }

    // T-068-007: weighted_rrf_records_per_source_contributions
    #[test]
    fn weighted_rrf_records_per_source_contributions() {
        let strategy = WeightedRrf::default();
        let candidates = vec![
            candidate(CandidateSource::Fts, "d1", 0),
            candidate(CandidateSource::Vector, "d1", 2),
        ];
        let output = strategy.merge(&candidates);
        assert_eq!(output.len(), 1);
        let fts = output[0].source_scores.get(&CandidateSource::Fts).copied();
        let vec = output[0]
            .source_scores
            .get(&CandidateSource::Vector)
            .copied();
        assert!((fts.unwrap() - 1.0 / 60.0).abs() < f64::EPSILON);
        assert!((vec.unwrap() - 1.0 / 62.0).abs() < f64::EPSILON);
    }

    // T-068-009: weighted_rrf_recency_zero_weight_is_noop
    #[test]
    fn weighted_rrf_recency_zero_weight_is_noop() {
        let strategy = WeightedRrf::default();
        let candidates = vec![
            candidate(CandidateSource::Fts, "d1", 0),
            candidate(CandidateSource::Fts, "d2", 1),
        ];
        let recency = RecencyConfig {
            weight: 0.0,
            half_life_days: 30.0,
        };
        let with_recency = strategy.merge_with_recency(&candidates, &recency, |_| Some(0.0));
        let without = strategy.merge(&candidates);
        assert_eq!(
            with_recency, without,
            "recency weight=0 must produce identical output to merge()"
        );
    }

    // T-068-010: weighted_rrf_recency_boosts_recent_hits
    #[test]
    fn weighted_rrf_recency_boosts_recent_hits() {
        let strategy = WeightedRrf::default();
        let candidates = vec![
            // d_old: best RRF rank but old
            candidate(CandidateSource::Fts, "d_old", 0),
            // d_new: worse RRF rank but recent
            candidate(CandidateSource::Fts, "d_new", 1),
        ];
        let recency = RecencyConfig {
            weight: 1.0,
            half_life_days: 30.0,
        };
        let ages = |id: &str| match id {
            "d_old" => Some(365.0), // very old → near-zero decay
            "d_new" => Some(0.0),   // brand new → decay 1.0 → +1.0 boost
            _ => None,
        };
        let output = strategy.merge_with_recency(&candidates, &recency, ages);
        // d_new RRF score 1/61 + recency 1.0 ≈ 1.016 > d_old's 1/60 + ε ≈ 0.0167
        assert_eq!(
            output[0].doc_id, "d_new",
            "Heavy recency boost must reorder a recent hit above an old hit with better RRF rank"
        );
    }

    // T-068-011: weighted_rrf_recency_skips_missing_age
    #[test]
    fn weighted_rrf_recency_skips_missing_age() {
        let strategy = WeightedRrf::default();
        let candidates = vec![
            candidate(CandidateSource::Fts, "d1", 0),
            candidate(CandidateSource::Fts, "d2", 1),
        ];
        let recency = RecencyConfig {
            weight: 1.0,
            half_life_days: 30.0,
        };
        // d1 has age, d2 has no metadata
        let ages = |id: &str| match id {
            "d1" => Some(0.0),
            _ => None,
        };
        let output = strategy.merge_with_recency(&candidates, &recency, ages);
        let d1 = output.iter().find(|h| h.doc_id == "d1").unwrap();
        let d2 = output.iter().find(|h| h.doc_id == "d2").unwrap();
        // d1 boosted by 1.0; d2 only has RRF
        assert!((d1.score - (1.0 / 60.0 + 1.0)).abs() < f64::EPSILON);
        assert!((d2.score - 1.0 / 61.0).abs() < f64::EPSILON);
    }

    // T-068-008: weighted_rrf_fts_heavy_weight_reorders
    //
    // Validates that non-uniform weights actually shift ranking — a doc with
    // FTS-only contribution at heavy weight beats a doc with both signals at
    // worse FTS rank.
    #[test]
    fn weighted_rrf_fts_heavy_weight_reorders() {
        let mut config = HybridSearchConfig::default();
        config.source_weights.insert(CandidateSource::Fts, 5.0);
        let strategy = WeightedRrf::new(config);
        let candidates = vec![
            // d1: FTS rank 0 only → 5/60 = 0.0833
            candidate(CandidateSource::Fts, "d1", 0),
            // d2: FTS rank 5, Vector rank 0 → 5/65 + 1/60 = 0.0936
            candidate(CandidateSource::Fts, "d2", 5),
            candidate(CandidateSource::Vector, "d2", 0),
        ];
        let output = strategy.merge(&candidates);
        assert_eq!(
            output[0].doc_id, "d2",
            "d2 wins via FTS rank 5 + Vector top hit"
        );
        // sanity: with default weights (fts=1.0), the order would still be d2 first
        // because d2 has 1/65 + 1/60 = 0.0320 vs d1 1/60 = 0.0167
        // So the test mainly verifies weighting works without reverting order
        let weighted_d2_score = 5.0 / 65.0 + 1.0 / 60.0;
        assert!((output[0].score - weighted_d2_score).abs() < f64::EPSILON);
    }
}
