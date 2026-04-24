use super::mlx::truncate_pair;
use super::*;
use crate::model_io::artifacts_from_cache;
use crate::test_support::{VALID_CONFIG_JSON, setup_fake_hf_cache};
use std::fs;

#[test]
fn t_003_candidate_verify_returns_missing_file_for_nonexistent_paths() {
    let candidate = CandidateArtifacts::from_paths(
        "/nonexistent/model.safetensors".into(),
        "/nonexistent/config.json".into(),
        "/nonexistent/tokenizer.json".into(),
    );
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::MissingFile { .. }),
        "expected MissingFile, got: {err}"
    );
}

#[test]
fn t_004_truncate_pair_short_input_unchanged() {
    let mut ids: Vec<u32> = (0..100).collect();
    let mut mask: Vec<u32> = vec![1; 100];
    let original_ids = ids.clone();
    let original_mask = mask.clone();
    truncate_pair(&mut ids, &mut mask, 8192, 0);
    assert_eq!(ids, original_ids);
    assert_eq!(mask, original_mask);
}

#[test]
fn t_005_truncate_pair_long_input_truncated_with_eos() {
    let mut ids: Vec<u32> = (0..8193).map(|i| i as u32).collect();
    let mut mask: Vec<u32> = vec![1; 8193];
    truncate_pair(&mut ids, &mut mask, 8192, 0);
    assert_eq!(ids.len(), 8192);
    assert_eq!(mask.len(), 8192);
    assert_eq!(ids[8191], 2, "last token should be EOS(2)");
}

#[test]
fn t_007_sort_results_descending_by_score() {
    let scores = vec![0.3, 0.9, 0.1];
    let results = sort_results(&scores);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].index, 1);
    assert!((results[0].score - 0.9).abs() < 1e-6);
    assert_eq!(results[1].index, 0);
    assert!((results[1].score - 0.3).abs() < 1e-6);
    assert_eq!(results[2].index, 2);
    assert!((results[2].score - 0.1).abs() < 1e-6);
}

#[test]
fn t_013_cache_lookup_returns_some_when_all_files_present() {
    let dir = tempfile::tempdir().unwrap();
    let model = RerankerModelId::RuriV3Reranker310m;
    setup_fake_hf_cache(
        dir.path(),
        model.repo_id(),
        model.revision(),
        &[
            ("model.safetensors", b"fake"),
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
        ],
    );
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, model).unwrap();
    let paths = result.expect("should be Some");
    assert!(paths.model.ends_with("model.safetensors"));
    assert!(paths.config.ends_with("config.json"));
    assert!(paths.tokenizer.ends_with("tokenizer.json"));
}

#[test]
fn t_021_cache_lookup_returns_none_when_cache_empty() {
    let dir = tempfile::tempdir().unwrap();
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, RerankerModelId::default()).unwrap();
    assert!(result.is_none());
}

#[test]
fn t_014_candidate_verify_returns_invalid_config_for_empty_config() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    fs::write(dir.path().join("config.json"), b"{}").unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidConfig { .. }),
        "expected InvalidConfig error, got: {err}"
    );
}

#[test]
fn t_015_candidate_verify_returns_invalid_tokenizer_for_bad_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    fs::write(dir.path().join("config.json"), VALID_CONFIG_JSON.as_bytes()).unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"not json").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidTokenizer(_)),
        "expected InvalidTokenizer error, got: {err}"
    );
}

#[test]
fn probe_env_to_paths_returns_ok_when_all_present() {
    let result = super::probe_env_to_paths(Some("/m".into()), Some("/c".into()), Some("/t".into()));
    // CandidateArtifacts is returned — verify it was constructed (path accessible in submodule)
    let candidate = result.unwrap().unwrap();
    assert_eq!(candidate.paths.model, PathBuf::from("/m"));
    assert_eq!(candidate.paths.config, PathBuf::from("/c"));
    assert_eq!(candidate.paths.tokenizer, PathBuf::from("/t"));
}

#[test]
fn truncate_pair_zero_max_len_returns_unchanged() {
    let mut ids: Vec<u32> = vec![1, 100, 200, 2];
    let mut mask: Vec<u32> = vec![1; 4];
    let original_ids = ids.clone();
    let original_mask = mask.clone();
    truncate_pair(&mut ids, &mut mask, 0, 0);
    assert_eq!(ids, original_ids, "max_len=0 should return unchanged");
    assert_eq!(mask, original_mask);
}

#[test]
fn truncate_pair_exact_boundary_unchanged() {
    let mut ids: Vec<u32> = (0..8192).map(|i| i as u32).collect();
    let mut mask: Vec<u32> = vec![1; 8192];
    let original_last = ids[8191];
    truncate_pair(&mut ids, &mut mask, 8192, 0);
    assert_eq!(ids.len(), 8192);
    assert_eq!(mask.len(), 8192);
    assert_eq!(
        ids[8191], original_last,
        "last token should NOT be overwritten at exact boundary"
    );
}

#[test]
fn sort_results_handles_nan_without_panic() {
    let scores = vec![0.5, f32::NAN, 0.3];
    let results = sort_results(&scores);
    assert_eq!(results.len(), 3);
    let mut indices: Vec<usize> = results.iter().map(|r| r.index).collect();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2]);
    // total_cmp: NaN is greatest → descending sort puts NaN first
    assert!(
        results[0].score.is_nan(),
        "NaN should be first (greatest in total_cmp)"
    );
    assert_eq!(results[1].index, 0, "0.5 should be second");
    assert_eq!(results[2].index, 2, "0.3 should be third");
}

#[test]
fn from_probe_error_maps_correctly() {
    use crate::model_probe::ProbeError;

    let err: RerankerInitError = ProbeError::HandlerNotInstalled.into();
    assert!(
        matches!(err, RerankerInitError::Backend(ref m) if m.contains("probe handler not installed")),
        "{err}"
    );

    let err: RerankerInitError = ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    assert!(
        matches!(err, RerankerInitError::ModelCorrupt { ref reason } if reason == "bad weights"),
        "{err}"
    );

    let err: RerankerInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
    assert!(
        matches!(err, RerankerInitError::Backend(ref m) if m == "spawn failed"),
        "{err}"
    );
}

use super::mlx::sigmoid;

#[test]
fn sigmoid_zero_returns_half() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
}

#[test]
fn sigmoid_large_positive_approaches_one() {
    assert!(sigmoid(20.0) > 0.999);
}

#[test]
fn sigmoid_large_negative_approaches_zero() {
    assert!(sigmoid(-20.0) < 0.001);
}

#[test]
fn sigmoid_nan_returns_nan() {
    assert!(sigmoid(f32::NAN).is_nan());
}

#[test]
fn mock_reranker_score_batch_empty_returns_ok_empty() {
    let r = MockReranker::default();
    assert!(r.score_batch(&[]).unwrap().is_empty());
}

#[test]
fn mock_reranker_score_returns_configured_value() {
    let r = MockReranker::with_score(0.7);
    let s = r.score("q", "d").unwrap();
    assert!((s - 0.7).abs() < 1e-6, "expected 0.7, got {s}");
}

/// MLX runtime tests — run with `cargo test --features test-mlx -- --ignored`
/// outside Codex seatbelt.
#[cfg(feature = "test-mlx")]
mod mlx_runtime_tests {
    use serial_test::serial;

    use super::*;
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    fn load_cached_artifacts() -> Artifacts {
        cached_artifacts(RerankerModelId::default())
            .expect("cache lookup should not fail")
            .expect("model should be cached for test-mlx tests")
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_006_score_batch_empty_returns_ok_empty() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let scores = reranker.score_batch(&[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_008_rerank_empty_returns_ok_empty() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let results = reranker.rerank("query", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_011_score_returns_value_in_unit_interval() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let score = reranker.score("test", "テスト文").unwrap();
        assert!(
            (0.0..=1.0).contains(&score) && score.is_finite(),
            "score should be in [0,1], got: {score}"
        );
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_012_rerank_returns_descending_scores_with_valid_indices() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let docs = ["related document", "unrelated text", "somewhat relevant"];
        let results = reranker.rerank("test query", &docs).unwrap();
        assert_eq!(results.len(), 3);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        let mut indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_016_score_batch_preserves_input_order_and_ranking() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let pair_a = ("東京の人口", "東京は日本最大の都市");
        let pair_b = ("東京の人口", "りんごは果物");
        let scores = reranker.score_batch(&[pair_a, pair_b]).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "related pair should score higher: {} vs {}",
            scores[0],
            scores[1]
        );
        for &s in &scores {
            assert!(
                (0.0..=1.0).contains(&s),
                "score should be in [0,1], got: {s}"
            );
        }
    }

    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_018_new_succeeds_with_cached_model() {
        require_unsandboxed_mlx_runtime();
        let result = Reranker::new(&load_cached_artifacts());
        assert!(
            result.is_ok(),
            "Reranker::new should succeed: {:?}",
            result.err()
        );
    }
}
