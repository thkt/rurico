//! Isolated research candidates: no new public API or default contract.
#![allow(dead_code)] // Production APIs compiled unchanged for the CPU probe.
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

#[path = "../../../../../src/retrieval.rs"]
mod retrieval;
use retrieval::{
    Aggregator, Candidate, CandidateSource, IdentityAggregator, MergeStrategy, MergedHit,
    TopKAverageAggregator, WeightedRrf, group_by_parent,
};

// Every allocation uses System unchanged. Live requested bytes include input;
// the measured peak is additional live bytes above the pre-call snapshot.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
fn allocated(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    if ACTIVE.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(size, Relaxed);
        PEAK.fetch_max(live, Relaxed);
    }
}
// SAFETY: forward the exact pointer and layout to System. Counters allocate
// nothing and do not alter alignment, ownership, or allocation failure.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        new
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

// Keep accumulation order and omission semantics; convert slots to the existing
// HashMap wire type only at output. A new source requires extending this match.
type Slots = (f64, [Option<f64>; 2]);
fn slots(strategy: &WeightedRrf, input: &[Candidate]) -> Vec<MergedHit> {
    let weights = [CandidateSource::Fts, CandidateSource::Vector].map(|source| {
        strategy
            .config
            .source_weights
            .get(&source)
            .copied()
            .unwrap_or(0.0)
    });
    let mut acc: HashMap<(&str, Option<&str>), Slots> = HashMap::new();
    for c in input {
        let slot = match c.source {
            CandidateSource::Fts => 0,
            CandidateSource::Vector => 1,
        };
        let weight = weights[slot];
        let denom = strategy.config.rrf_k + c.rank as f64;
        if weight == 0.0 || !weight.is_finite() || !denom.is_finite() || denom <= 0.0 {
            continue;
        }
        let value = weight / denom;
        if !value.is_finite() {
            continue;
        }
        let entry = acc
            .entry((&c.doc_id, c.chunk_id.as_deref()))
            .or_insert((0.0, [None; 2]));
        entry.0 += value;
        *entry.1[slot].get_or_insert(0.0) += value;
    }
    let mut out: Vec<_> = acc
        .into_iter()
        .filter_map(|((doc, chunk), (score, values))| {
            if !score.is_finite() || values.iter().flatten().any(|v| !v.is_finite()) {
                return None;
            }
            let source_scores = [CandidateSource::Fts, CandidateSource::Vector]
                .into_iter()
                .zip(values)
                .filter_map(|(src, value)| value.map(|v| (src, v)))
                .collect();
            Some(MergedHit {
                doc_id: doc.into(),
                chunk_id: chunk.map(str::to_owned),
                score,
                source_scores,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    out
}

// Select by score plus original position, then restore the original summation
// order among ties. Reuse production averaging (including #298 overflow) rather
// than introducing another numerical implementation. The extra grouping/clones
// are deliberately included in the cost of this prototype.
fn select_topk(input: &[MergedHit], k: usize) -> Vec<MergedHit> {
    if k == 0 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    for bucket in group_by_parent(input).into_values() {
        let mut indexed: Vec<_> = bucket.into_iter().enumerate().collect();
        let cmp = |a: &(usize, &MergedHit), b: &(usize, &MergedHit)| {
            b.1.score.total_cmp(&a.1.score).then_with(|| a.0.cmp(&b.0))
        };
        if indexed.len() > k {
            indexed.select_nth_unstable_by(k, cmp);
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(cmp);
        selected.extend(indexed.into_iter().map(|(_, hit)| hit.clone()));
    }
    TopKAverageAggregator::new(k).aggregate(&selected)
}

// Contract proposals deliberately separate from internal optimizations.
// Duplicate key is source/doc/chunk (rank is payload); the best rank wins.
fn best_rank(input: &[Candidate]) -> Vec<Candidate> {
    let mut positions = HashMap::new();
    let mut out: Vec<Candidate> = Vec::new();
    for c in input {
        let key = (c.source, c.doc_id.as_str(), c.chunk_id.as_deref());
        if let Some(&index) = positions.get(&key) {
            let previous: &mut Candidate = &mut out[index];
            if c.rank < previous.rank {
                *previous = c.clone();
            }
        } else {
            positions.insert(key, out.len());
            out.push(c.clone());
        }
    }
    out
}
fn reject_duplicates(input: &[Candidate]) -> Result<(), usize> {
    let mut seen = HashMap::new();
    for (index, c) in input.iter().enumerate() {
        if seen
            .insert((c.source, c.doc_id.as_str(), c.chunk_id.as_deref()), ())
            .is_some()
        {
            return Err(index);
        }
    }
    Ok(())
}

fn input(parents: usize, chunks: usize) -> (Vec<Candidate>, Vec<MergedHit>) {
    let mut candidates = Vec::new();
    let mut hits = Vec::new();
    for p in 0..parents {
        for c in 0..chunks {
            let score = ((c * 37 + p * 13) % 128) as f64 / 128.0;
            hits.push(MergedHit {
                doc_id: format!("d{p:06}"),
                chunk_id: Some(format!("c{c:04}")),
                score,
                source_scores: HashMap::from([
                    (CandidateSource::Fts, score * 0.5),
                    (CandidateSource::Vector, score * 0.25),
                ]),
            });
            for source in [CandidateSource::Fts, CandidateSource::Vector] {
                candidates.push(Candidate {
                    source,
                    doc_id: format!("d{p:06}"),
                    chunk_id: Some(format!("c{c:04}")),
                    score: 0.0,
                    rank: p * chunks + c,
                });
            }
        }
    }
    (candidates, hits)
}
fn measure(workload: &str, variant: &str, repeat: usize, run: impl FnOnce() -> Vec<MergedHit>) {
    if repeat == 0 {
        black_box(run());
        return;
    }
    let baseline = LIVE.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    ACTIVE.store(true, Relaxed);
    let start = Instant::now();
    let result = black_box(run());
    let elapsed = start.elapsed().as_nanos();
    ACTIVE.store(false, Relaxed);
    let calls = CALLS.load(Relaxed);
    let bytes = BYTES.load(Relaxed);
    let peak = PEAK.load(Relaxed).saturating_sub(baseline);
    let count = result.len();
    drop(result);
    println!(
        "{}",
        serde_json::json!({"workload":workload,"variant":variant,"repeat":repeat,"ns":elapsed,"alloc_calls":calls,"requested_bytes":bytes,"additional_peak_bytes":peak,"output_count":count})
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let verify = args.len() == 2 && args[1] == "--verify";
    assert!(
        args.len() == 1 || verify || args.len() == 3,
        "usage: retrieval-spike-313 [--verify | small|large variant]"
    );
    if args.len() == 3 {
        assert!(["small", "large"].contains(&args[1].as_str()));
        assert!(
            [
                "rrf_map",
                "rrf_slots",
                "identity_borrowed",
                "identity_owned",
                "topk_sort",
                "topk_select"
            ]
            .contains(&args[2].as_str())
        );
    }
    let rrf = WeightedRrf::default();
    for (workload, parents, chunks) in [("small", 10, 3), ("large", 1000, 100)] {
        if args.len() == 3 && args[1] != workload {
            continue;
        }
        let (candidates, hits) = input(parents, chunks);
        if verify {
            assert_eq!(slots(&rrf, &candidates), rrf.merge(&candidates));
            assert_eq!(
                select_topk(&hits, 3),
                TopKAverageAggregator::new(3).aggregate(&hits)
            );
            continue;
        }
        for repeat in 0..8 {
            let mut routes = [
                "rrf_map",
                "rrf_slots",
                "identity_borrowed",
                "identity_owned",
                "topk_sort",
                "topk_select",
            ];
            if repeat % 2 == 1 {
                routes.reverse();
            }
            for route in routes {
                if args.len() == 3 && args[2] != route {
                    continue;
                }
                // Ownership preparation is excluded for both identity paths:
                // each receives an equal pre-existing Vec, then consumes/clones.
                let prepared = if route.starts_with("identity") {
                    Some(hits.clone())
                } else {
                    None
                };
                if route == "identity_owned" {
                    measure(workload, route, repeat, || prepared.unwrap());
                    continue;
                }
                measure(workload, route, repeat, || match route {
                    "rrf_map" => rrf.merge(&candidates),
                    "rrf_slots" => slots(&rrf, &candidates),
                    "identity_borrowed" => IdentityAggregator.aggregate(prepared.as_ref().unwrap()),
                    "topk_sort" => TopKAverageAggregator::new(3).aggregate(&hits),
                    "topk_select" => select_topk(&hits, 3),
                    _ => unreachable!(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrieval::HybridSearchConfig;
    #[test]
    fn contract_proposals_keep_distinct_sources_and_chunks() {
        let (mut candidates, _) = input(1, 2);
        candidates[0].rank = 4;
        let mut duplicate = candidates[0].clone();
        duplicate.rank = 0;
        candidates.push(duplicate.clone());
        assert_eq!(reject_duplicates(&candidates), Err(4));
        let best = best_rank(&candidates);
        assert_eq!(best.len(), 4);
        assert_eq!(best[0], duplicate);
        assert_eq!(&best[1..], &candidates[1..4]);
        assert_eq!(reject_duplicates(&best), Ok(()));
        let config = HybridSearchConfig {
            rrf_k: 4.0,
            ..HybridSearchConfig::default()
        };
        let out = WeightedRrf::new(config).merge(&best);
        assert_eq!(out[0].score, 0.5);
        assert_eq!(
            out[0].source_scores,
            HashMap::from([
                (CandidateSource::Fts, 0.25),
                (CandidateSource::Vector, 0.25)
            ])
        );
    }
    #[test]
    fn topk_selection_keeps_literal_boundary_sources() {
        let make = |id: &str, score, source, value| MergedHit {
            doc_id: id.into(),
            chunk_id: Some("child".into()),
            score,
            source_scores: HashMap::from([(source, value)]),
        };
        let hits = vec![
            make("b", 0.25, CandidateSource::Vector, 0.125),
            make("a", 0.75, CandidateSource::Fts, 0.5),
            make("b", 0.75, CandidateSource::Vector, 0.25),
            make("a", 0.75, CandidateSource::Vector, 0.75),
            make("a", 0.5, CandidateSource::Fts, 0.5),
        ];
        let expected = vec![
            MergedHit {
                doc_id: "a".into(),
                chunk_id: None,
                score: 0.75,
                source_scores: HashMap::from([
                    (CandidateSource::Fts, 0.25),
                    (CandidateSource::Vector, 0.375),
                ]),
            },
            MergedHit {
                doc_id: "b".into(),
                chunk_id: None,
                score: 0.5,
                source_scores: HashMap::from([(CandidateSource::Vector, 0.1875)]),
            },
        ];
        assert_eq!(select_topk(&hits, 2), expected);
        assert_eq!(
            select_topk(&hits, 1),
            vec![
                MergedHit {
                    doc_id: "a".into(),
                    chunk_id: None,
                    score: 0.75,
                    source_scores: HashMap::from([(CandidateSource::Fts, 0.5)])
                },
                MergedHit {
                    doc_id: "b".into(),
                    chunk_id: None,
                    score: 0.75,
                    source_scores: HashMap::from([(CandidateSource::Vector, 0.25)])
                },
            ]
        );
    }
    #[test]
    fn prototypes_preserve_boundaries_and_finite_omission() {
        let (mut candidates, mut hits) = input(2, 5);
        for k in [0, 1, 3, 5, 10] {
            assert_eq!(
                select_topk(&hits, k),
                TopKAverageAggregator::new(k).aggregate(&hits)
            );
        }
        hits[0].score = f64::MAX;
        hits[1].score = f64::MAX;
        hits[0].source_scores.insert(CandidateSource::Fts, f64::MAX);
        hits[1].source_scores.insert(CandidateSource::Fts, f64::MAX);
        for score in [f64::MAX, f64::NAN, f64::INFINITY] {
            hits[0].score = score;
            assert_eq!(
                select_topk(&hits, 3),
                TopKAverageAggregator::new(3).aggregate(&hits)
            );
        }
        candidates.push(candidates[0].clone());
        for weight in [1.0, -1.0, 0.0, f64::MAX, f64::NAN, f64::INFINITY] {
            let strategy = WeightedRrf::new(HybridSearchConfig {
                rrf_k: 1.0,
                source_weights: HashMap::from([
                    (CandidateSource::Fts, weight),
                    (CandidateSource::Vector, -1.0),
                ]),
            });
            assert_eq!(slots(&strategy, &candidates), strategy.merge(&candidates));
        }
    }
}
