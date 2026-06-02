use std::assert_matches;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use super::super::MockReranker;
use super::*;

fn counting_init(
    counter: Arc<AtomicUsize>,
) -> impl Fn() -> Result<MockReranker, String> + Send + Sync + 'static {
    move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(MockReranker::with_score(0.5))
    }
}

// T-154-001: new_does_not_invoke_init
#[test]
fn new_does_not_invoke_init() {
    let counter = Arc::new(AtomicUsize::new(0));
    let _lazy = LazyReranker::new(counting_init(Arc::clone(&counter)));
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "init must not run on construction"
    );
}

// T-154-003: first_call_initialises_then_caches
#[test]
fn first_call_initialises_then_caches() {
    let counter = Arc::new(AtomicUsize::new(0));
    let lazy = LazyReranker::new(counting_init(Arc::clone(&counter)));
    lazy.score("q", "d").unwrap();
    lazy.score("q", "d2").unwrap();
    lazy.score_batch(&[("q", "d")]).unwrap();
    lazy.rerank("q", &["d"]).unwrap();
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "init must run exactly once across mixed method calls"
    );
}

// T-154-004: init_failure_returns_init_failed
#[test]
fn init_failure_returns_init_failed() {
    let lazy: LazyReranker<MockReranker> =
        LazyReranker::new(|| Err(String::from("artifact missing")));
    let err = lazy.score("q", "d").unwrap_err();
    assert!(
        matches!(&err, RerankerError::InitFailed(m) if m == "artifact missing"),
        "expected InitFailed(\"artifact missing\"), got: {err:?}"
    );
}

// T-154-005: failure_is_cached_across_methods
#[test]
fn failure_is_cached_across_methods() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let lazy: LazyReranker<MockReranker> = LazyReranker::new(move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        Err(String::from("boom"))
    });
    assert_matches!(lazy.score("q", "d"), Err(RerankerError::InitFailed(_)));
    assert_matches!(
        lazy.score_batch(&[("q", "d")]),
        Err(RerankerError::InitFailed(_))
    );
    assert_matches!(lazy.rerank("q", &["d"]), Err(RerankerError::InitFailed(_)));
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "failure must be cached, init runs only once"
    );
}

// T-154-006: concurrent_calls_run_init_once
#[test]
fn concurrent_calls_run_init_once() {
    let counter = Arc::new(AtomicUsize::new(0));
    let lazy = Arc::new(LazyReranker::new(counting_init(Arc::clone(&counter))));
    let n = 16;
    let barrier = Arc::new(Barrier::new(n));
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let lazy = Arc::clone(&lazy);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                lazy.score("q", "d").unwrap()
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "OnceLock must serialise concurrent first-call init"
    );
}

// T-154-007: rerank_delegates_to_inner
#[test]
fn rerank_delegates_to_inner() {
    let lazy: LazyReranker<MockReranker> = LazyReranker::new(|| Ok(MockReranker::with_score(0.42)));
    let docs = ["a", "b", "c"];
    let results = lazy.rerank("q", &docs).unwrap();
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(
            (r.score - 0.42).abs() < 1e-6,
            "score must come from wrapped MockReranker, got: {}",
            r.score
        );
    }
}

// T-154-008: score_batch_delegates_pair_count
#[test]
fn score_batch_delegates_pair_count() {
    let lazy: LazyReranker<MockReranker> = LazyReranker::new(|| Ok(MockReranker::with_score(0.3)));
    let pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")];
    let scores = lazy.score_batch(&pairs).unwrap();
    assert_eq!(scores.len(), 3, "one score per input pair");
    for &s in &scores {
        assert!((s - 0.3).abs() < 1e-6, "expected 0.3, got {s}");
    }
}

// T-154-009: debug_reflects_initialisation_state
#[test]
fn debug_reflects_initialisation_state() {
    let lazy: LazyReranker<MockReranker> = LazyReranker::new(|| Ok(MockReranker::default()));
    assert!(format!("{lazy:?}").contains("initialized: false"));
    lazy.score("q", "d").unwrap();
    assert!(format!("{lazy:?}").contains("initialized: true"));
}

// T-154-010: empty_inputs_short_circuit_without_init
//
// Pins parity with `Reranker::score_batch` / `rerank`, which return
// `Ok(vec![])` before touching the model. Without this, replay-first
// paths that pass empty candidates would surface `InitFailed` from a
// missing model instead of the expected empty result.
#[test]
fn empty_inputs_short_circuit_without_init() {
    let counter = Arc::new(AtomicUsize::new(0));
    let lazy: LazyReranker<MockReranker> = LazyReranker::new({
        let counter = Arc::clone(&counter);
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(String::from("model unavailable"))
        }
    });
    assert!(lazy.score_batch(&[]).unwrap().is_empty());
    assert!(lazy.rerank("q", &[]).unwrap().is_empty());
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "empty inputs must not trigger init (parity with Reranker)"
    );
}
