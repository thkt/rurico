use super::mlx::truncate_pair;
use super::*;
use crate::artifacts::RerankerKind;
use crate::model_lifecycle::probe_env_to_paths;
#[cfg(unix)]
use crate::test_support::setup_fake_hf_cache_with_symlinks;
use crate::test_support::{
    VALID_CONFIG_JSON, assert_cache_lookup_returns_none_when_empty,
    assert_cache_lookup_returns_some_when_all_files_present,
    assert_from_probe_error_maps_correctly,
};
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
fn t_007b_sort_results_ties_break_by_original_index() {
    // scores: [0.5, 0.7, 0.7, 0.5] — desc by score, asc by index on ties.
    let scores = vec![0.5, 0.7, 0.7, 0.5];
    let results = sort_results(&scores);
    let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
    assert_eq!(indices, vec![1, 2, 0, 3]);
}

#[test]
fn t_013_cache_lookup_returns_some_when_all_files_present() {
    assert_cache_lookup_returns_some_when_all_files_present(RerankerModelId::RuriV3Reranker310m);
}

#[test]
fn t_021_cache_lookup_returns_none_when_cache_empty() {
    assert_cache_lookup_returns_none_when_empty(RerankerModelId::default());
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

// T-019: regression — same invariant as T-018 for the reranker module.
#[cfg(unix)]
#[test]
fn t_019_probe_env_to_paths_preserves_snapshot_symlink_filename() {
    let hf_home = tempfile::tempdir().unwrap();
    let cache_root = hf_home.path().join("hub");
    fs::create_dir_all(&cache_root).unwrap();
    setup_fake_hf_cache_with_symlinks(
        &cache_root,
        "org/reranker",
        "dev",
        &[
            ("model.safetensors", b"weights"),
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
        ],
    );
    let snapshot_dir = cache_root.join("models--org--reranker/snapshots/abc123");
    let m = snapshot_dir.join("model.safetensors");
    let c = snapshot_dir.join("config.json");
    let t = snapshot_dir.join("tokenizer.json");

    let hf_home_path = hf_home.path().to_path_buf();
    temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
        let candidate = probe_env_to_paths::<RerankerKind>(
            Some(m.to_string_lossy().into_owned()),
            Some(c.to_string_lossy().into_owned()),
            Some(t.to_string_lossy().into_owned()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            candidate.paths.model.file_name().and_then(|s| s.to_str()),
            Some("model.safetensors"),
            "regression: probe_env_to_paths must preserve snapshot symlink filename"
        );
        assert_eq!(
            candidate.paths.config.file_name().and_then(|s| s.to_str()),
            Some("config.json")
        );
        assert_eq!(
            candidate
                .paths
                .tokenizer
                .file_name()
                .and_then(|s| s.to_str()),
            Some("tokenizer.json")
        );
    });
}

#[test]
fn probe_env_to_paths_returns_ok_when_all_present() {
    let hf_home = tempfile::tempdir().unwrap();
    let cache_root = hf_home.path().join("hub");
    fs::create_dir_all(&cache_root).unwrap();
    let m = cache_root.join("model.safetensors");
    let c = cache_root.join("config.json");
    let t = cache_root.join("tokenizer.json");
    fs::write(&m, b"").unwrap();
    fs::write(&c, b"{}").unwrap();
    fs::write(&t, b"{}").unwrap();

    let hf_home_path = hf_home.path().to_path_buf();
    temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
        let result = probe_env_to_paths::<RerankerKind>(
            Some(m.to_string_lossy().into_owned()),
            Some(c.to_string_lossy().into_owned()),
            Some(t.to_string_lossy().into_owned()),
        );
        // CandidateArtifacts is returned with original (non-canonicalized) paths
        let candidate = result.unwrap().unwrap();
        assert_eq!(candidate.paths.model, m);
        assert_eq!(candidate.paths.config, c);
        assert_eq!(candidate.paths.tokenizer, t);
    });
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
    assert_from_probe_error_maps_correctly();
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
    use std::time::Instant;

    use serial_test::serial;
    use tracing_test::traced_test;

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

    // Without sub-batching, 5000 short pairs in bucket 0 build a single
    // `5000 × 128 = 640_000` token matrix that exhausts GPU memory on most
    // M-series devices. Pins the OOM regression that motivated sub-batching.
    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_score_batch_5000_pairs_short_docs_completes_without_oom() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();

        let pairs: Vec<(&str, &str)> = (0..5000).map(|_| ("query", "doc")).collect();
        let scores = reranker
            .score_batch(&pairs)
            .expect("5000 short pairs must not OOM after sub-batching");

        assert_eq!(scores.len(), 5000, "one score per input pair");
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} must be finite");
            assert!(
                (0.0..=1.0).contains(&s),
                "score[{i}] = {s} must be in [0,1]"
            );
        }
    }

    // Pin small-N (50 pairs, single sub-batch) latency floor as a smoke signal.
    // A future refactor that introduces real per-call work in score_batch shows
    // up here. Median printed via --nocapture; thresholds intentionally not
    // asserted (device-dependent).
    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_score_batch_50_pairs_small_n_latency_smoke() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();

        let pairs: Vec<(&str, &str)> = (0..50).map(|_| ("query", "doc")).collect();

        const WARMUP: usize = 3;
        const RUNS: usize = 30;
        for _ in 0..WARMUP {
            reranker.score_batch(&pairs).unwrap();
        }

        let mut times_us: Vec<u128> = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t0 = Instant::now();
            reranker.score_batch(&pairs).unwrap();
            times_us.push(t0.elapsed().as_micros());
        }
        times_us.sort_unstable();
        let p50 = times_us[RUNS / 2];
        let p95 = times_us[(RUNS * 95) / 100];

        println!("score_batch(50 pairs, bucket=128) p50={p50}µs p95={p95}µs over {RUNS} runs",);
        assert!(p50 > 0, "timing must be non-zero (sanity)");
    }

    // Pin the dispatch log emission so a future refactor cannot silently drop
    // the per-call sub-batching telemetry. Field names are part of the
    // contract subscribers consume; renaming them must update this assertion.
    #[traced_test]
    #[test]
    #[ignore = "requires unsandboxed MLX runtime"]
    #[serial]
    fn t_score_batch_emits_dispatch_log() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();

        let pairs: Vec<(&str, &str)> = vec![("query", "doc"); 3];
        reranker.score_batch(&pairs).unwrap();

        assert!(logs_contain("reranker score_batch dispatch"));
        assert!(logs_contain("batch_size=3"));
        assert!(
            logs_contain("sub_batch_count=1"),
            "3 pairs in bucket 0 fit a single sub-batch",
        );
        assert!(
            logs_contain("bucket_len=128"),
            "short pairs must land in bucket 0 (len=128)",
        );
    }
}
