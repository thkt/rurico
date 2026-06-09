//! Retrieval pipeline contract (ADR 0004, Issue #67 / Phase 3).
//!
//! Plug-in points frozen by ADR 0004:
//! - Stage 3 aggregation: [`Aggregator`](crate::retrieval::Aggregator) trait
//!   between merge and rerank. Default impl is
//!   [`IdentityAggregator`](crate::retrieval::IdentityAggregator).
//!
//! Concrete strategies
//! ([`MaxChunkAggregator`](crate::retrieval::MaxChunkAggregator),
//! [`DedupeAggregator`](crate::retrieval::DedupeAggregator),
//! [`TopKAverageAggregator`](crate::retrieval::TopKAverageAggregator)) live
//! alongside the trait; downstream crates can also supply their own
//! `impl Aggregator` for domain-specific dedupers.

use std::collections::HashMap;
use std::f64::consts::LN_2;

use serde::{Deserialize, Serialize};

/// Exponential recency decay: 1.0 at age=0, 0.5 at one half-life, approaching 0.0.
///
/// Negative `age_days` is clamped to 0 (returns 1.0).
/// Returns 0.0 when `half_life_days <= 0.0` (avoids division by zero).
pub(crate) fn recency_decay(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 0.0;
    }
    (-LN_2 * age_days.max(0.0) / half_life_days).exp()
}

/// Source of a Stage 1 candidate hit (ADR 0004 Stage 1 output).
///
/// Closed enum: misspelled labels become compile errors. The set is frozen
/// at `Fts` and `Vector`; ADR 0005 records the Phase 6 (#70) prefix-ensemble
/// experiment that considered adding new variants and decided against it on
/// measured results.
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
///
/// `chunk_id` carries the child chunk identifier when chunk-level retrieval
/// is active. `None` means whole-document indexing — the legacy posture,
/// where every aggregator collapses to identity because Stage 2 fusion
/// already keys by `doc_id` alone. With `Some(id)`, Stage 2 fuses on
/// `(doc_id, chunk_id)` so distinct chunks survive merge and Stage 3
/// aggregators can produce non-vacuous rankings.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Originating retrieval source.
    pub source: CandidateSource,
    /// Parent document identifier.
    pub doc_id: String,
    /// Optional child chunk identifier within `doc_id`. `None` indicates
    /// whole-document granularity.
    pub chunk_id: Option<String>,
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
/// to Stage 4 (rerank). Also serves as the public ranked-hit type for
/// downstream pipeline outputs — Serialize/Deserialize keep the pipeline
/// output JSON shape unchanged across the merge boundary.
///
/// `source_scores` records per-source contributions so downstream UIs can
/// display score breakdown and debugging surfaces can attribute fusion
/// outcomes. It is `serde(default)` for backward-compat with pre-Phase-4
/// JSON fixtures that omit the field; an empty map represents "source
/// information unavailable" rather than "no source contributed".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedHit {
    /// Parent document identifier carried through Stage 1+2.
    pub doc_id: String,
    /// Optional child chunk identifier. `None` is the pre-chunk-level
    /// posture — Stage 3 aggregators that collapse parents also strip
    /// `chunk_id` to `None` so Stage 4 sees parent-granular hits.
    /// `serde(default)` keeps pre-#76 baseline.json files round-tripping.
    #[serde(default)]
    pub chunk_id: Option<String>,
    /// Aggregated relevance score (RRF score).
    pub score: f64,
    /// Per-source contributions to the fused score. Empty when source
    /// information is not preserved through the pipeline.
    #[serde(default)]
    pub source_scores: HashMap<CandidateSource, f64>,
}

impl MergedHit {
    /// Build a parent-granular hit (`chunk_id = None`). Single seam every
    /// Stage 3 collapse aggregator routes through, so the "strip chunk_id"
    /// contract is enforced once instead of repeated at three call sites.
    pub(crate) fn parent_granular(
        doc_id: String,
        score: f64,
        source_scores: HashMap<CandidateSource, f64>,
    ) -> Self {
        Self {
            doc_id,
            chunk_id: None,
            score,
            source_scores,
        }
    }
}

/// Group chunk-level [`MergedHit`]s by their parent `doc_id`.
///
/// Returns one bucket per parent doc, with the original input order preserved
/// inside each bucket. Hits whose `chunk_id` is `None` still bucket under
/// their `doc_id` — the helper does not enforce a chunk-level invariant
/// because mixed-granularity input is legitimate (a parent-collapsed hit
/// alongside chunk-level siblings from a different parent).
///
/// Designed for downstream Stage 4 wiring that needs to reconstruct
/// "周辺文脈込みの parent text" from sibling chunks before reranking, and
/// for custom [`Aggregator`] impls that want shared bucketing without
/// re-implementing the parent-grouping pass.
pub fn group_by_parent<'a>(hits: &'a [MergedHit]) -> HashMap<&'a str, Vec<&'a MergedHit>> {
    let mut buckets: HashMap<&'a str, Vec<&'a MergedHit>> = HashMap::new();
    for hit in hits {
        buckets.entry(hit.doc_id.as_str()).or_default().push(hit);
    }
    buckets
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
    ///
    /// Must be finite and keep `rrf_k + rank` positive for every candidate
    /// rank; a positive `rrf_k` always suffices. A candidate whose
    /// `rrf_k + rank` is non-positive or non-finite (NaN, ±inf, or
    /// `rrf_k <= -rank`) has its contribution dropped — the same handling as a
    /// zero-weight source — so the fused score never becomes inf or NaN.
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
/// reduces to `1 / (60 + rank)`.
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

/// Per-doc accumulator for [`WeightedRrf::merge`].
///
/// One `entry()` lookup per candidate fills both the fused score and the
/// per-source breakdown — replaces a parallel pair of HashMaps that
/// otherwise required matched updates and a second-pass `remove` per hit.
#[derive(Default)]
struct WeightedRrfAcc {
    score: f64,
    source_scores: HashMap<CandidateSource, f64>,
}

impl MergeStrategy for WeightedRrf {
    fn merge(&self, candidates: &[Candidate]) -> Vec<MergedHit> {
        let fts_w = self
            .config
            .source_weights
            .get(&CandidateSource::Fts)
            .copied()
            .unwrap_or(0.0);
        let vec_w = self
            .config
            .source_weights
            .get(&CandidateSource::Vector)
            .copied()
            .unwrap_or(0.0);
        // Fusion key is (doc_id, chunk_id) — chunk-level retrieval (Issue #76)
        // requires distinct child chunks to survive Stage 2 so Stage 3
        // aggregators can collapse them on their own contract. With
        // `chunk_id == None` the tuple degenerates to `(doc_id, None)` and
        // pre-Issue-#76 fusion-on-doc_id behaviour is preserved bit-equal.
        let mut acc: HashMap<(&str, Option<&str>), WeightedRrfAcc> = HashMap::new();
        for cand in candidates {
            let weight = match cand.source {
                CandidateSource::Fts => fts_w,
                CandidateSource::Vector => vec_w,
            };
            // Zero-weight sources are disabled: skipping prevents docs that only
            // hit the disabled source from leaking into the merged set with a
            // score of 0.0, which would otherwise survive truncate-to-k and
            // reach the reranker as if they were valid candidates.
            if weight == 0.0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let rank_f = cand.rank as f64;
            let denom = self.config.rrf_k + rank_f;
            // A non-finite or non-positive denominator (rrf_k is NaN/±inf, or
            // rrf_k + rank <= 0) would emit inf/NaN fused scores that poison
            // Stage 3/4 ranking, JSON output, and debug surfaces. Drop the
            // contribution — same posture as a zero-weight source — so a
            // misconfigured rrf_k degrades to "candidate skipped" rather than a
            // non-finite score. Default rrf_k = 60.0 keeps denom >= 60, so valid
            // configs stay bit-equal.
            if !denom.is_finite() || denom <= 0.0 {
                continue;
            }
            let contribution = weight / denom;
            let key = (cand.doc_id.as_str(), cand.chunk_id.as_deref());
            let entry = acc.entry(key).or_default();
            entry.score += contribution;
            *entry.source_scores.entry(cand.source).or_default() += contribution;
        }
        let mut hits: Vec<MergedHit> = acc
            .into_iter()
            .map(|((doc_id, chunk_id), a)| MergedHit {
                doc_id: doc_id.to_owned(),
                chunk_id: chunk_id.map(str::to_owned),
                score: a.score,
                source_scores: a.source_scores,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
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
    /// **Output MUST be sorted by `score` descending**, then `doc_id`
    /// ascending, then `chunk_id` ascending (3-key total order, matching the
    /// `MergeStrategy::merge` contract in this module). The pipeline
    /// truncates the result to `config.k` after this call, so non-sorted
    /// output would silently drop higher-scoring hits. Output length may be
    /// ≤ input length (dedupe / max-chunk collapse duplicates).
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit>;
}

/// Identity aggregator — returns the input unchanged.
///
/// Default for the reference pipeline composition; preserves the pre-Phase-3
/// behaviour where each input document produces exactly one hit. The sort
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
///
/// Output `chunk_id` is `None` (parent-granular). When chunk-level retrieval
/// is active, this collapses sibling child chunks into a single parent hit
/// carrying the best chunk's score and `source_scores`.
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
        let mut output: Vec<MergedHit> = best
            .into_values()
            .map(|h| MergedHit::parent_granular(h.doc_id.clone(), h.score, h.source_scores.clone()))
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
///
/// Output `chunk_id` is `None` (parent-granular) — Dedupe's contract is
/// "one hit per parent doc", so retaining a chunk_id would imply parents
/// inadvertently surface their first-seen child's identity to Stage 4.
#[derive(Debug, Default, Clone, Copy)]
pub struct DedupeAggregator;

impl Aggregator for DedupeAggregator {
    fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit> {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        // Output holds at most one entry per parent doc_id — chunk-level input
        // can be `chunks_per_doc`× larger than the realised output, so sizing
        // off `hits.len()` over-allocates. Defer to the default growth path.
        let mut output = Vec::new();
        for hit in hits {
            if seen.insert(hit.doc_id.as_str(), ()).is_none() {
                output.push(MergedHit::parent_granular(
                    hit.doc_id.clone(),
                    hit.score,
                    hit.source_scores.clone(),
                ));
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
        let mut output: Vec<MergedHit> = group_by_parent(hits)
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
                MergedHit::parent_granular(id.to_owned(), mean, source_scores)
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
mod tests;
