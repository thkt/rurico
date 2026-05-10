use std::fmt::{self, Debug, Formatter};
use std::sync::OnceLock;

use super::{RankedResult, Rerank, RerankerError};

type InitFn<R> = dyn Fn() -> Result<R, String> + Send + Sync;

/// Defers wrapped reranker construction until the first [`Rerank`] call.
///
/// Wraps any [`Rerank`] implementation so model artifacts, tokenizer, and MLX
/// state are only allocated when actually needed. Useful for replay-first
/// retrieval paths (e.g. `replay-first-search`,
/// `verify-baseline kind=first_search_replay`) that may skip reranking
/// entirely — without [`LazyReranker`] those paths still pay the load cost
/// of an unused reranker.
///
/// The init closure runs at most once: [`OnceLock`] synchronises concurrent
/// callers and both success and failure are cached. A failed init is **not**
/// retried — model-load errors are deterministic in-process (missing
/// artifacts, corrupted cache, malformed config) so re-running the closure
/// would burn IO without changing the outcome.
///
/// # Examples
///
/// ```no_run
/// # use rurico::reranker::{RankedResult, Rerank, RerankerError};
/// # struct Stub;
/// # impl Rerank for Stub {
/// #     fn score(&self, _q: &str, _d: &str) -> Result<f32, RerankerError> { Ok(0.5) }
/// #     fn score_batch(&self, p: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
/// #         Ok(vec![0.5; p.len()])
/// #     }
/// #     fn rerank(&self, _q: &str, d: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
/// #         Ok(d.iter().enumerate().map(|(i, _)| RankedResult { index: i, score: 0.5 }).collect())
/// #     }
/// # }
/// use rurico::reranker::LazyReranker;
///
/// let lazy = LazyReranker::new(|| Ok(Stub));
/// // No init has happened yet — closure runs only on the first call below.
/// let score = lazy.score("query", "document")?;
/// # Ok::<(), rurico::reranker::RerankerError>(())
/// ```
pub struct LazyReranker<R: Rerank> {
    cell: OnceLock<Result<R, String>>,
    init: Box<InitFn<R>>,
}

impl<R: Rerank> LazyReranker<R> {
    /// Wrap a fallible init closure. The closure runs on the first
    /// [`Rerank`] method call and at most once.
    pub fn new<F>(init: F) -> Self
    where
        F: Fn() -> Result<R, String> + Send + Sync + 'static,
    {
        Self {
            cell: OnceLock::new(),
            init: Box::new(init),
        }
    }

    fn inner(&self) -> Result<&R, RerankerError> {
        match self.cell.get_or_init(|| (self.init)()) {
            Ok(r) => Ok(r),
            Err(msg) => Err(RerankerError::InitFailed(msg.clone())),
        }
    }
}

impl<R: Rerank> Debug for LazyReranker<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyReranker")
            .field("initialized", &self.cell.get().is_some())
            .finish_non_exhaustive()
    }
}

impl<R: Rerank> Rerank for LazyReranker<R> {
    fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError> {
        self.inner()?.score(query, document)
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        self.inner()?.score_batch(pairs)
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
        self.inner()?.rerank(query, documents)
    }
}

#[cfg(test)]
mod tests {
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

    // T-154-002: untouched_lazy_never_invokes_init
    #[test]
    fn untouched_lazy_never_invokes_init() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyReranker::new(counting_init(Arc::clone(&counter)));
        drop(lazy);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "dropping without method calls must not run init"
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
        assert!(matches!(
            lazy.score("q", "d"),
            Err(RerankerError::InitFailed(_))
        ));
        assert!(matches!(
            lazy.score_batch(&[("q", "d")]),
            Err(RerankerError::InitFailed(_))
        ));
        assert!(matches!(
            lazy.rerank("q", &["d"]),
            Err(RerankerError::InitFailed(_))
        ));
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
        let lazy: LazyReranker<MockReranker> =
            LazyReranker::new(|| Ok(MockReranker::with_score(0.42)));
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
        let lazy: LazyReranker<MockReranker> =
            LazyReranker::new(|| Ok(MockReranker::with_score(0.3)));
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
}
