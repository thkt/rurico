//! Failure injection is test-only and thread-local; production has no fault mode.
use std::cell::Cell;

use mlx_rs::error::Exception;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    Forward,
    Pool,
    Eval,
    Readback,
}

thread_local! {
    static FAILURE: Cell<Option<Stage>> = const { Cell::new(None) };
    static CLEANUPS: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn checkpoint(stage: Stage) -> Result<(), Exception> {
    if FAILURE.get() == Some(stage) {
        Err(Exception::custom(format!("injected {stage:?} failure")))
    } else {
        Ok(())
    }
}

pub(crate) fn record_cleanup() {
    CLEANUPS.set(CLEANUPS.get() + 1);
}

#[test]
fn inference_drops_resources_before_one_cleanup_and_preserves_result() {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::run_inference;

    struct Resource<'a>(&'static str, &'a RefCell<Vec<&'static str>>);
    impl Drop for Resource<'_> {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    // No GPU: deterministic lifetime/error regression, not memory measurement.
    for failure in [
        None,
        Some(Stage::Forward),
        Some(Stage::Pool),
        Some(Stage::Eval),
        Some(Stage::Readback),
    ] {
        let events = RefCell::new(Vec::new());
        let original = Rc::new(failure);
        let fail = |stage| {
            if failure == Some(stage) {
                Err(Rc::clone(&original))
            } else {
                Ok(())
            }
        };
        let readback_capture = Resource("readback capture", &events);
        let result = run_inference(
            || {
                let partial = Resource("forward", &events);
                fail(Stage::Forward)?;
                Ok(partial)
            },
            |output| {
                let _capture = readback_capture;
                fail(Stage::Pool)?;
                let _pooled = Resource("pooled", &events);
                drop(output);
                fail(Stage::Eval)?;
                fail(Stage::Readback)?;
                Ok(vec![0.25_f32, -0.5])
            },
            || events.borrow_mut().push("cleanup"),
        );
        if failure.is_some() {
            assert!(Rc::ptr_eq(&result.unwrap_err(), &original));
        } else {
            assert_eq!(result.unwrap(), vec![0.25, -0.5]);
        }
        let mut events = events.into_inner();
        assert_eq!(events.pop(), Some("cleanup"), "{failure:?}");
        // Relative order among Arrays is immaterial; every resource must be
        // released exactly once, and all releases must precede cleanup.
        events.sort_unstable();
        let expected = if matches!(failure, Some(Stage::Forward | Stage::Pool)) {
            vec!["forward", "readback capture"]
        } else {
            vec!["forward", "pooled", "readback capture"]
        };
        assert_eq!(events, expected, "{failure:?}");
    }
}

#[cfg(feature = "test-mlx")]
mod runtime {
    use std::iter::once;
    use std::time::Instant;

    use mlx_rs::memory;

    use super::{CLEANUPS, FAILURE, Stage};
    use crate::embed::{Embed, Embedder, ModelId, cached_artifacts};
    use crate::model_io::ModelArtifact;
    use crate::reranker::{
        Reranker, RerankerModelId, cached_artifacts as cached_reranker_artifacts,
    };
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    fn observe(path: &str, mut infer: impl FnMut() -> Result<Vec<f32>, String>) {
        struct ResetFailure;
        impl Drop for ResetFailure {
            fn drop(&mut self) {
                FAILURE.set(None);
            }
        }
        let _reset = ResetFailure;
        // First sample is the baseline; model weights/masks remain resident.
        let mut baseline: Option<Vec<f32>> = None;
        let attempts = (0..6)
            .flat_map(|cycle| {
                [
                    None,
                    Some(Stage::Forward),
                    Some(Stage::Pool),
                    Some(Stage::Eval),
                    Some(Stage::Readback),
                ]
                .map(|failure| (cycle, failure))
            })
            .chain(once((6, None))); // Verify and record recovery after the final failure.
        for (cycle, failure) in attempts {
            FAILURE.set(failure);
            CLEANUPS.set(0);
            memory::reset_peak_memory().unwrap();
            let start = Instant::now();
            let result = infer();
            let elapsed = start.elapsed();
            FAILURE.set(None);
            assert_eq!(CLEANUPS.get(), 1, "{path} {failure:?}");
            match failure {
                Some(stage) => assert!(
                    result
                        .unwrap_err()
                        .contains(&format!("injected {stage:?} failure"))
                ),
                None => {
                    let values = result.unwrap();
                    assert!(!values.is_empty());
                    assert!(values.iter().all(|v| v.is_finite()));
                    if let Some(previous) = &baseline {
                        assert_eq!(values.len(), previous.len());
                        assert!(
                            values
                                .iter()
                                .zip(previous)
                                .all(|(a, b)| (a - b).abs() <= 1e-5)
                        );
                    } else {
                        baseline = Some(values);
                    }
                }
            }
            eprintln!(
                "cleanup_memory path={path} cycle={cycle} failure={failure:?} batch={} bucket=128 elapsed_ms={} active_bytes={} cache_bytes={} peak_bytes={}",
                if path == "query" { 1 } else { 2 },
                elapsed.as_millis(),
                memory::active_memory().unwrap(),
                memory::cache_memory().unwrap(),
                memory::peak_memory().unwrap(),
            );
        }
    }

    /// Host-only observation, run alone in a nextest subprocess (CONTRIBUTING).
    /// No arbitrary memory threshold: retain every sample for trend review.
    #[test]
    #[ignore = "requires cached embed/reranker 310m models and unsandboxed MLX"]
    fn real_model_cleanup_memory() {
        require_unsandboxed_mlx_runtime();
        let model = ModelId::DEFAULT;
        eprintln!(
            "cleanup_memory model={} revision={}",
            model.repo_id(),
            model.revision()
        );
        let artifacts = cached_artifacts(model)
            .unwrap()
            .expect("cache embed model first");
        let embedder = Embedder::new(&artifacts).unwrap();
        observe("query", || {
            embedder
                .embed_query("東京の人口")
                .map_err(|e| e.to_string())
        });
        observe("batch", || {
            embedder
                .embed_documents_batch(&["東京は日本の都市です。", "京都も日本の都市です。"])
                .map(|docs| {
                    docs.iter()
                        .flat_map(|doc| doc.chunks().iter().flatten().copied())
                        .collect()
                })
                .map_err(|e| e.to_string())
        });
        drop(embedder);

        let model = RerankerModelId::default();
        eprintln!(
            "cleanup_memory model={} revision={}",
            model.repo_id(),
            ModelArtifact::revision(model)
        );
        let artifacts = cached_reranker_artifacts(model)
            .unwrap()
            .expect("cache reranker model first");
        let reranker = Reranker::new(&artifacts).unwrap();
        observe("reranker", || {
            reranker
                .score_batch(&[
                    ("東京の人口", "東京は日本の都市です。"),
                    ("東京の人口", "京都も日本の都市です。"),
                ])
                .map_err(|e| e.to_string())
        });
    }
}
