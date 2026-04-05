use super::mlx::truncate_pair;
use super::*;

#[test]
fn t_001_default_model_id_is_ruri_v3_reranker_310m() {
    assert_eq!(
        RerankerModelId::default(),
        RerankerModelId::RuriV3Reranker310m
    );
}

#[test]
fn t_002_repo_id_returns_correct_string() {
    assert_eq!(
        RerankerModelId::RuriV3Reranker310m.repo_id(),
        "cl-nagoya/ruri-v3-reranker-310m"
    );
}

#[test]
fn t_022_revision_returns_pinned_commit_hash() {
    assert_eq!(
        RerankerModelId::RuriV3Reranker310m.revision(),
        "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"
    );
}

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
    crate::test_support::setup_fake_hf_cache(
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
    let result = crate::model_io::artifacts_from_cache(&cache, model).unwrap();
    let paths = result.expect("should be Some");
    assert!(paths.model.ends_with("model.safetensors"));
    assert!(paths.config.ends_with("config.json"));
    assert!(paths.tokenizer.ends_with("tokenizer.json"));
}

#[test]
fn t_021_cache_lookup_returns_none_when_cache_empty() {
    let dir = tempfile::tempdir().unwrap();
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = crate::model_io::artifacts_from_cache(&cache, RerankerModelId::default()).unwrap();
    assert!(result.is_none());
}

/// Valid config JSON (all required ModernBERT fields) for testing errors
/// that occur after config parsing.
const VALID_CONFIG_JSON: &str = r#"{
    "vocab_size": 1000,
    "hidden_size": 768,
    "num_hidden_layers": 2,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "layer_norm_eps": 1e-5,
    "pad_token_id": 0,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_attention": 128,
    "local_rope_theta": 10000.0
}"#;

#[test]
fn t_014_candidate_verify_returns_invalid_config_for_empty_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
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
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("config.json"), VALID_CONFIG_JSON.as_bytes()).unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"not json").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidTokenizer(_)),
        "expected InvalidTokenizer error, got: {err}"
    );
}


#[test]
fn probe_env_to_paths_returns_none_when_model_absent() {
    let result = super::probe_env_to_paths(None, Some("c".into()), Some("t".into()));
    assert!(result.is_none());
}

#[test]
fn probe_env_to_paths_returns_err_when_config_missing() {
    let result = super::probe_env_to_paths(Some("m".into()), None, Some("t".into()));
    let err = result.unwrap().unwrap_err();
    assert_eq!(err, crate::model_probe::PROBE_EXIT_ENV_INCOMPLETE);
}

#[test]
fn probe_env_to_paths_returns_err_when_tokenizer_missing() {
    let result = super::probe_env_to_paths(Some("m".into()), Some("c".into()), None);
    let err = result.unwrap().unwrap_err();
    assert_eq!(err, crate::model_probe::PROBE_EXIT_ENV_INCOMPLETE);
}

#[test]
fn probe_env_to_paths_returns_ok_when_all_present() {
    let result = super::probe_env_to_paths(Some("/m".into()), Some("/c".into()), Some("/t".into()));
    // CandidateArtifacts is returned — verify it was constructed (path accessible in submodule)
    let candidate = result.unwrap().unwrap();
    assert_eq!(candidate.paths.model, std::path::PathBuf::from("/m"));
    assert_eq!(candidate.paths.config, std::path::PathBuf::from("/c"));
    assert_eq!(candidate.paths.tokenizer, std::path::PathBuf::from("/t"));
}

#[test]
fn lock_inner_poison_maps_to_inference_error() {
    // Verify that lock_inner's map_err closure produces the expected error variant
    // and message when the mutex is poisoned. We can't construct RerankerInner
    // without model files, so we test the error mapping directly.
    let mutex: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = mutex.lock().unwrap();
        panic!("poison");
    }));
    assert!(mutex.is_poisoned());
    let err = mutex
        .lock()
        .map_err(|_| RerankerError::inference("reranker lock poisoned"))
        .unwrap_err();
    assert!(
        matches!(err, RerankerError::Inference(ref msg) if msg == "reranker lock poisoned"),
        "unexpected error: {err}"
    );
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
fn from_probe_error_handler_not_installed_maps_to_backend() {
    let err: RerankerInitError = crate::model_probe::ProbeError::HandlerNotInstalled.into();
    assert!(
        matches!(err, RerankerInitError::Backend(ref msg) if msg.contains("probe handler not installed")),
        "{err}"
    );
}

#[test]
fn from_probe_error_model_load_failed_maps_to_model_corrupt() {
    let err: RerankerInitError = crate::model_probe::ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    assert!(
        matches!(err, RerankerInitError::ModelCorrupt { ref reason } if reason == "bad weights"),
        "{err}"
    );
}

#[test]
fn from_probe_error_subprocess_failed_maps_to_backend() {
    let err: RerankerInitError =
        crate::model_probe::ProbeError::SubprocessFailed("spawn failed".into()).into();
    assert!(
        matches!(err, RerankerInitError::Backend(ref msg) if msg == "spawn failed"),
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
fn mock_reranker_rerank_empty_returns_ok_empty() {
    let r = MockReranker::default();
    assert!(r.rerank("query", &[]).unwrap().is_empty());
}

#[test]
fn mock_reranker_score_returns_configured_value() {
    let r = MockReranker::with_score(0.7);
    let s = r.score("q", "d").unwrap();
    assert!((s - 0.7).abs() < 1e-6, "expected 0.7, got {s}");
}

#[test]
fn mock_reranker_score_batch_returns_score_per_pair() {
    let r = MockReranker::with_score(0.3);
    let pairs = [("q", "a"), ("q", "b"), ("q", "c")];
    let scores = r.score_batch(&pairs).unwrap();
    assert_eq!(scores.len(), 3);
    for &s in &scores {
        assert!((s - 0.3).abs() < 1e-6, "expected 0.3, got {s}");
    }
}

#[test]
fn mock_reranker_rerank_returns_all_document_indices() {
    let r = MockReranker::default();
    let docs = ["a", "b", "c"];
    let results = r.rerank("q", &docs).unwrap();
    assert_eq!(results.len(), 3);
    let mut indices: Vec<usize> = results.iter().map(|r| r.index).collect();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2]);
}

/// MLX runtime tests — require `cargo test --features test-mlx`.
#[cfg(feature = "test-mlx")]
mod mlx_runtime_tests {
    use serial_test::serial;

    use super::*;

    fn load_cached_artifacts() -> Artifacts {
        cached_artifacts(RerankerModelId::default())
            .expect("cache lookup should not fail")
            .expect("model should be cached for test-mlx tests")
    }

    #[test]
    #[serial]
    fn t_006_score_batch_empty_returns_ok_empty() {
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let scores = reranker.score_batch(&[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    #[serial]
    fn t_008_rerank_empty_returns_ok_empty() {
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let results = reranker.rerank("query", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    #[serial]
    fn t_011_score_returns_value_in_unit_interval() {
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        let score = reranker.score("test", "テスト文").unwrap();
        assert!(
            score >= 0.0 && score <= 1.0 && score.is_finite(),
            "score should be in [0,1], got: {score}"
        );
    }

    #[test]
    #[serial]
    fn t_012_rerank_returns_descending_scores_with_valid_indices() {
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
    #[serial]
    fn t_016_score_batch_preserves_input_order_and_ranking() {
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
            assert!(s >= 0.0 && s <= 1.0, "score should be in [0,1], got: {s}");
        }
    }

    #[test]
    #[serial]
    fn t_018_new_succeeds_with_cached_model() {
        let result = Reranker::new(&load_cached_artifacts());
        assert!(
            result.is_ok(),
            "Reranker::new should succeed: {:?}",
            result.err()
        );
    }
}
