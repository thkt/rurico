//! Host-only cached-model observation; kept separate from CI-tested injection controls.
use std::iter::once;
use std::time::Instant;

use mlx_rs::memory;

use super::{CLEANUPS, FAILURE, Stage};
use crate::embed::{Embed, Embedder, ModelId, cached_artifacts};
use crate::model_io::ModelArtifact;
use crate::reranker::{Reranker, RerankerModelId, cached_artifacts as cached_reranker_artifacts};
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
