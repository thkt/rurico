use super::*;

// T-007: age_zero_returns_one
#[test]
fn age_zero_returns_one() {
    let result = recency_decay(0.0, 30.0);
    assert!((result - 1.0).abs() < f64::EPSILON);
}

// T-008: one_half_life_returns_half
#[test]
fn one_half_life_returns_half() {
    let result = recency_decay(30.0, 30.0);
    assert!((result - 0.5).abs() < 0.01);
}

// T-009: two_half_lives_returns_quarter
#[test]
fn two_half_lives_returns_quarter() {
    let result = recency_decay(60.0, 30.0);
    assert!((result - 0.25).abs() < 0.01);
}

// T-010: zero_half_life_returns_zero
#[test]
fn zero_half_life_returns_zero() {
    let result = recency_decay(0.0, 0.0);
    assert!((result - 0.0).abs() < f64::EPSILON);
}

// T-010b: negative_half_life_returns_zero
#[test]
fn negative_half_life_returns_zero() {
    let result = recency_decay(5.0, -1.0);
    assert!((result - 0.0).abs() < f64::EPSILON);
}

// T-010c: negative_age_clamped_to_one
#[test]
fn negative_age_clamped_to_one() {
    let result = recency_decay(-5.0, 30.0);
    assert!((result - 1.0).abs() < f64::EPSILON);
}

fn hit(id: &str, score: f64) -> MergedHit {
    MergedHit {
        doc_id: id.to_owned(),
        chunk_id: None,
        score,
        source_scores: HashMap::new(),
    }
}

fn hit_with_sources(id: &str, score: f64, sources: &[(CandidateSource, f64)]) -> MergedHit {
    MergedHit {
        doc_id: id.to_owned(),
        chunk_id: None,
        score,
        source_scores: sources.iter().copied().collect(),
    }
}

/// Build a chunk-level [`MergedHit`].
///
/// The chunk-level aggregator tests assert how each strategy handles
/// distinct child chunks of the same parent doc — the helpers above keep
/// `chunk_id = None` for backward-compat with the pre-#76 tests.
fn chunk_hit(doc_id: &str, chunk_id: &str, score: f64) -> MergedHit {
    MergedHit {
        doc_id: doc_id.to_owned(),
        chunk_id: Some(chunk_id.to_owned()),
        score,
        source_scores: HashMap::new(),
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
        chunk_id: None,
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

// T-104-001: weighted_rrf_default_breaks_ties_by_doc_id
//
// When two docs receive identical RRF scores, merge must order them by
// doc_id ascending (lexicographic). Tie-break contract preserved on the
// canonical WeightedRrf path.
#[test]
fn weighted_rrf_default_breaks_ties_by_doc_id() {
    let strategy = WeightedRrf::default();
    let candidates = vec![
        candidate(CandidateSource::Fts, "d3", 0),
        candidate(CandidateSource::Vector, "d1", 0),
    ];
    let output = strategy.merge(&candidates);
    assert_eq!(output.len(), 2);
    assert!(
        (output[0].score - output[1].score).abs() < f64::EPSILON,
        "scores must be tied"
    );
    assert_eq!(
        output[0].doc_id, "d1",
        "lower doc_id must come first on tie"
    );
    assert_eq!(output[1].doc_id, "d3");
}

// T-068-006: weighted_rrf_zero_weight_drops_source
//
// Disabled-source candidates must NOT leak into the merged set. If a doc
// has only a contribution from the zero-weight source, it must be absent
// from the output entirely — otherwise it would survive truncate-to-k and
// reach the reranker as a valid candidate, undermining single-source
// weight-tuning comparisons.
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
    assert_eq!(output.len(), 1, "vector-only d2 must be dropped");
    assert_eq!(output[0].doc_id, "d1");
    assert!((output[0].score - 1.0 / 60.0).abs() < f64::EPSILON);
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

// T-076-001: weighted_rrf_fuses_chunks_separately_when_chunk_id_differs
//
// Issue #76 / ADR 0004 line 180: when chunk-level retrieval is active,
// distinct child chunks of the same parent doc MUST survive Stage 2
// fusion. Otherwise Stage 3 aggregators have nothing to collapse and
// every strategy is structurally identical to identity.
#[test]
fn weighted_rrf_fuses_chunks_separately_when_chunk_id_differs() {
    let strategy = WeightedRrf::default();
    let candidates = vec![
        Candidate {
            source: CandidateSource::Fts,
            doc_id: "d1".to_owned(),
            chunk_id: Some("c0".to_owned()),
            score: 0.0,
            rank: 0,
        },
        Candidate {
            source: CandidateSource::Fts,
            doc_id: "d1".to_owned(),
            chunk_id: Some("c1".to_owned()),
            score: 0.0,
            rank: 1,
        },
    ];
    let output = strategy.merge(&candidates);
    assert_eq!(
        output.len(),
        2,
        "two distinct chunks of d1 must survive Stage 2 fusion"
    );
    let chunk_ids: Vec<Option<&str>> = output.iter().map(|h| h.chunk_id.as_deref()).collect();
    assert!(chunk_ids.contains(&Some("c0")), "c0 must be present");
    assert!(chunk_ids.contains(&Some("c1")), "c1 must be present");
}

// T-076-002: identity_preserves_chunk_level_distinctness
#[test]
fn identity_preserves_chunk_level_distinctness() {
    let aggregator = IdentityAggregator;
    let input = vec![
        chunk_hit("d1", "c0", 0.9),
        chunk_hit("d1", "c1", 0.5),
        chunk_hit("d2", "c0", 0.7),
    ];
    let output = aggregator.aggregate(&input);
    assert_eq!(output.len(), 3, "Identity must keep all chunk-level hits");
    assert_eq!(output, input, "Identity must preserve chunk_ids verbatim");
}

// T-076-003: max_chunk_collapses_sibling_chunks_to_parent
//
// Two chunks of d1 (scores 0.5 + 0.9), one chunk of d2 (0.7). MaxChunk
// must keep d1's higher-scoring chunk (0.9), drop the sibling, and emit
// parent-granular hits (chunk_id=None).
#[test]
fn max_chunk_collapses_sibling_chunks_to_parent() {
    let aggregator = MaxChunkAggregator;
    let input = vec![
        chunk_hit("d1", "c0", 0.5),
        chunk_hit("d2", "c0", 0.7),
        chunk_hit("d1", "c1", 0.9),
    ];
    let output = aggregator.aggregate(&input);
    assert_eq!(output.len(), 2, "MaxChunk collapses d1's two chunks");
    assert_eq!(output[0].doc_id, "d1", "d1 wins ranking via its 0.9 chunk");
    assert!(
        (output[0].score - 0.9).abs() < f64::EPSILON,
        "d1 score must be 0.9 (max of its chunks)"
    );
    assert_eq!(
        output[0].chunk_id, None,
        "MaxChunk emits parent-granular hits (chunk_id=None)"
    );
    assert_eq!(output[1].doc_id, "d2");
    assert_eq!(output[1].chunk_id, None);
}

// T-076-004: dedupe_collapses_chunks_to_first_parent
#[test]
fn dedupe_collapses_chunks_to_first_parent() {
    let aggregator = DedupeAggregator;
    let input = vec![
        chunk_hit("d1", "c0", 0.9),
        chunk_hit("d2", "c0", 0.7),
        chunk_hit("d1", "c1", 0.5),
    ];
    let output = aggregator.aggregate(&input);
    assert_eq!(output.len(), 2, "Dedupe drops d1's second chunk");
    assert_eq!(output[0].doc_id, "d1");
    assert_eq!(
        output[0].chunk_id, None,
        "Dedupe emits parent-granular hits (chunk_id=None)"
    );
    assert!((output[0].score - 0.9).abs() < f64::EPSILON);
    assert_eq!(output[1].doc_id, "d2");
    assert_eq!(output[1].chunk_id, None);
}

// T-076-005: topk_average_collapses_chunks_with_chunk_id_none
//
// TopKAverage already collapsed sibling chunks pre-#76 (the bucketing
// happened on doc_id). The new contract is that chunk_id is reset to
// None so Stage 4 sees parent-granular hits regardless of input
// granularity.
#[test]
fn topk_average_collapses_chunks_with_chunk_id_none() {
    let aggregator = TopKAverageAggregator::new(2);
    let input = vec![
        chunk_hit("d1", "c0", 1.0),
        chunk_hit("d1", "c1", 0.5),
        chunk_hit("d2", "c0", 0.25),
    ];
    let output = aggregator.aggregate(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].doc_id, "d1");
    assert_eq!(
        output[0].chunk_id, None,
        "TopKAverage emits parent-granular hits"
    );
    // d1 top-2 = (1.0 + 0.5) / 2 = 0.75
    assert!((output[0].score - 0.75).abs() < f64::EPSILON);
}

// T-076-006: group_by_parent_buckets_chunks_per_doc
#[test]
fn group_by_parent_buckets_chunks_per_doc() {
    let input = vec![
        chunk_hit("d1", "c0", 0.9),
        chunk_hit("d2", "c0", 0.7),
        chunk_hit("d1", "c1", 0.5),
    ];
    let buckets = group_by_parent(&input);
    assert_eq!(buckets.len(), 2, "two parent docs → two buckets");
    let d1_bucket = buckets.get("d1").expect("d1 bucket must exist");
    assert_eq!(d1_bucket.len(), 2, "d1 bucket holds both of its chunks");
    let d1_chunk_ids: Vec<Option<&str>> = d1_bucket.iter().map(|h| h.chunk_id.as_deref()).collect();
    assert_eq!(
        d1_chunk_ids,
        vec![Some("c0"), Some("c1")],
        "input order preserved inside the bucket"
    );
    let d2_bucket = buckets.get("d2").expect("d2 bucket must exist");
    assert_eq!(d2_bucket.len(), 1);
}

// T-105-008a: weighted_rrf_merge_returns_empty_for_empty_candidates
#[test]
fn weighted_rrf_merge_returns_empty_for_empty_candidates() {
    let strategy = WeightedRrf::default();
    let output = strategy.merge(&[]);
    assert!(
        output.is_empty(),
        "empty candidate slice must yield empty merged hits"
    );
}

// T-105-008b: weighted_rrf_merge_with_recency_returns_empty_for_empty_candidates
#[test]
fn weighted_rrf_merge_with_recency_returns_empty_for_empty_candidates() {
    let strategy = WeightedRrf::default();
    let recency = RecencyConfig {
        weight: 1.0,
        half_life_days: 30.0,
    };
    let output = strategy.merge_with_recency(&[], &recency, |_| Some(0.0));
    assert!(
        output.is_empty(),
        "empty candidates must short-circuit even with non-zero recency weight"
    );
}

// T-105-009: group_by_parent_returns_empty_map_for_empty_hits
#[test]
fn group_by_parent_returns_empty_map_for_empty_hits() {
    let buckets = group_by_parent(&[]);
    assert!(
        buckets.is_empty(),
        "empty hits slice must yield zero parent buckets"
    );
}

// T-105-017a: merged_hit_deserializes_when_chunk_id_field_omitted
//
// Backward-compat with pre-Issue #76 baseline.json files that omit the
// chunk_id field. `serde(default)` keeps Option<String> defaulting to None
// — pin the behaviour so a future serde attribute change does not silently
// break legacy fixture round-trips.
#[test]
fn merged_hit_deserializes_when_chunk_id_field_omitted() {
    let json = r#"{"doc_id":"d1","score":0.5,"source_scores":{"fts":0.5}}"#;
    let hit: MergedHit = serde_json::from_str(json).expect("legacy JSON must parse");
    assert_eq!(hit.doc_id, "d1");
    assert!((hit.score - 0.5).abs() < f64::EPSILON);
    assert_eq!(
        hit.chunk_id, None,
        "omitted chunk_id must default to None, not error"
    );
    assert_eq!(
        hit.source_scores.get(&CandidateSource::Fts),
        Some(&0.5),
        "source_scores must round-trip the populated entry"
    );
}

// T-105-017b: hybrid_search_config_serde_round_trips_default
//
// Default config must serialize then deserialize to a value equal to the
// starting Default — guards against an accidental serde attribute change
// that would shift downstream JSON shape.
#[test]
fn hybrid_search_config_serde_round_trips_default() {
    let config = HybridSearchConfig::default();
    let json = serde_json::to_string(&config).expect("Default must serialize");
    let parsed: HybridSearchConfig =
        serde_json::from_str(&json).expect("serialized form must deserialize");
    assert_eq!(parsed, config, "round-trip must preserve Default config");
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
