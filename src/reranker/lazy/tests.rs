use std::error::Error;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use mlx_rs::error::Exception;

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

#[test]
fn init_failure_returns_init_failed() {
    let lazy: LazyReranker<MockReranker> =
        LazyReranker::new(|| Err(String::from("artifact missing")));
    let err = lazy.score("q", "d").unwrap_err();
    assert!(
        matches!(&err, RerankerError::InitFailed { message, .. } if message == "artifact missing"),
        "expected InitFailed, got: {err:?}"
    );
    assert_eq!(err.to_string(), "init failed: artifact missing");
}

#[test]
fn failure_is_cached_across_methods() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    let lazy: LazyReranker<MockReranker> = LazyReranker::with_error(move || {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        Err(super::super::ModelInitError::backend(Exception::custom(
            "load failed",
        )))
    });
    let errors = [
        lazy.score("q", "d").unwrap_err(),
        lazy.score_batch(&[("q", "d")]).unwrap_err(),
        lazy.rerank("q", &["d"]).unwrap_err(),
    ];
    // The cached non-Clone cause must outlive the lazy wrapper itself.
    drop(lazy);
    for error in &errors {
        assert!(matches!(error, RerankerError::InitFailed { .. }));
        let init = error.source().unwrap();
        assert!(init.is::<super::super::ModelInitError>());
        let backend = init.source().unwrap();
        assert!(backend.is::<Exception>());
        assert!(backend.source().is_none());
        assert!(ptr::eq(init, errors[0].source().unwrap()));
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "failure must be cached, init runs only once"
    );
}

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

// A spy at the wrapped Rerank boundary checks arguments and returned values;
// constant-score mocks cannot detect dropped/reordered/replaced input.
#[test]
fn delegates_contents_order_results_and_errors_once_per_call() {
    use std::sync::Mutex;

    #[derive(Debug, PartialEq)]
    enum Call {
        Score(String, String),
        Batch(Vec<(String, String)>),
        Rerank(String, Vec<String>),
    }
    struct Spy(Arc<Mutex<Vec<Call>>>);
    impl Rerank for Spy {
        fn score(&self, query: &str, doc: &str) -> Result<f32, RerankerError> {
            self.0
                .lock()
                .unwrap()
                .push(Call::Score(query.into(), doc.into()));
            if query == "fail" {
                return Err(RerankerError::NonFiniteOutput);
            }
            Ok(0.25)
        }
        fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
            self.0.lock().unwrap().push(Call::Batch(
                pairs.iter().map(|&(q, d)| (q.into(), d.into())).collect(),
            ));
            if pairs[0].0 == "fail" {
                return Err(RerankerError::NonFiniteOutput);
            }
            Ok(vec![0.8, 0.2, 0.6])
        }
        fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
            self.0.lock().unwrap().push(Call::Rerank(
                query.into(),
                docs.iter().map(|&d| d.into()).collect(),
            ));
            if query == "fail" {
                return Err(RerankerError::NonFiniteOutput);
            }
            Ok(vec![
                RankedResult {
                    index: 2,
                    score: 0.9,
                },
                RankedResult {
                    index: 0,
                    score: 0.7,
                },
                RankedResult {
                    index: 1,
                    score: 0.3,
                },
            ])
        }
    }
    let calls = Arc::new(Mutex::new(Vec::new()));
    let inits = Arc::new(AtomicUsize::new(0));
    let lazy = LazyReranker::new({
        let calls = Arc::clone(&calls);
        let inits = Arc::clone(&inits);
        move || {
            inits.fetch_add(1, Ordering::SeqCst);
            Ok(Spy(Arc::clone(&calls)))
        }
    });
    assert_eq!(lazy.score("単独 query", "単独 doc").unwrap(), 0.25);
    assert_eq!(
        lazy.score_batch(&[("q2", "d2"), ("q1", "d1"), ("q2", "d3")])
            .unwrap(),
        [0.8, 0.2, 0.6]
    );
    let results = lazy
        .rerank("順位 query", &["doc B", "doc A", "doc C"])
        .unwrap();
    assert_eq!(
        results
            .iter()
            .map(|r| (r.index, r.score))
            .collect::<Vec<_>>(),
        [(2, 0.9), (0, 0.7), (1, 0.3)]
    );
    for error in [
        lazy.score("fail", "bad doc").unwrap_err(),
        lazy.score_batch(&[("fail", "bad pair")]).unwrap_err(),
        lazy.rerank("fail", &["bad rank"]).unwrap_err(),
    ] {
        assert!(matches!(error, RerankerError::NonFiniteOutput));
    }
    assert_eq!(
        *calls.lock().unwrap(),
        [
            Call::Score("単独 query".into(), "単独 doc".into()),
            Call::Batch(vec![
                ("q2".into(), "d2".into()),
                ("q1".into(), "d1".into()),
                ("q2".into(), "d3".into())
            ]),
            Call::Rerank(
                "順位 query".into(),
                vec!["doc B".into(), "doc A".into(), "doc C".into()]
            ),
            Call::Score("fail".into(), "bad doc".into()),
            Call::Batch(vec![("fail".into(), "bad pair".into())]),
            Call::Rerank("fail".into(), vec!["bad rank".into()]),
        ]
    );
    assert_eq!(inits.load(Ordering::SeqCst), 1);
}

#[test]
fn debug_reflects_initialisation_state() {
    let lazy: LazyReranker<MockReranker> = LazyReranker::new(|| Ok(MockReranker::default()));
    assert!(format!("{lazy:?}").contains("initialized: false"));
    lazy.score("q", "d").unwrap();
    assert!(format!("{lazy:?}").contains("initialized: true"));
}

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
