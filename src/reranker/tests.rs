use super::processing::truncate_pair;
use super::*;
use crate::test_support::{
    VALID_CONFIG_JSON, assert_cache_lookup_returns_none_when_empty,
    assert_cache_lookup_returns_some_when_all_files_present,
};
use std::fs;

#[test]
fn candidate_verify_returns_missing_file_for_nonexistent_paths() {
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
fn truncate_pair_short_input_unchanged() {
    let mut ids: Vec<u32> = (0..100).collect();
    let mut mask: Vec<u32> = vec![1; 100];
    let original_ids = ids.clone();
    let original_mask = mask.clone();
    truncate_pair(&mut ids, &mut mask, 8192, 0);
    assert_eq!(ids, original_ids);
    assert_eq!(mask, original_mask);
}

#[test]
fn truncate_pair_long_input_truncated_with_eos() {
    let mut ids: Vec<u32> = (0..8193).map(|i| i as u32).collect();
    let mut mask: Vec<u32> = vec![1; 8193];
    truncate_pair(&mut ids, &mut mask, 8192, 0);
    assert_eq!(ids.len(), 8192);
    assert_eq!(mask.len(), 8192);
    assert_eq!(ids[8191], 2, "last token should be EOS(2)");
}

#[test]
fn sort_results_descending_by_score() {
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
fn sort_results_ties_break_by_original_index() {
    // scores: [0.5, 0.7, 0.7, 0.5] — desc by score, asc by index on ties.
    let scores = vec![0.5, 0.7, 0.7, 0.5];
    let results = sort_results(&scores);
    let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
    assert_eq!(indices, vec![1, 2, 0, 3]);

    // Distinct finite logits collapse to equal endpoint scores. Keep input order,
    // not raw-logit order, within each endpoint.
    let scores =
        super::processing::scores_from_logits(&[20.0, 30.0, -100.0, -90.0], 4, 128).unwrap();
    assert_eq!(scores, [1.0, 1.0, 0.0, 0.0]);
    let indices: Vec<_> = sort_results(&scores).iter().map(|r| r.index).collect();
    assert_eq!(indices, [0, 1, 2, 3]);
}

#[test]
fn sort_results_with_empty_scores_returns_empty_vec() {
    let results = sort_results(&[]);
    assert!(
        results.is_empty(),
        "empty scores slice must yield zero RankedResult entries"
    );
}

#[test]
fn cache_lookup_returns_some_when_all_files_present() {
    assert_cache_lookup_returns_some_when_all_files_present(RerankerModelId::RuriV3Reranker310m);
}

#[test]
fn cache_lookup_returns_none_when_cache_empty() {
    assert_cache_lookup_returns_none_when_empty(RerankerModelId::default());
}

#[test]
fn candidate_verify_returns_invalid_config_for_empty_config() {
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
fn candidate_verify_returns_invalid_tokenizer_for_bad_tokenizer() {
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
fn score_readback_preserves_finite_scores_and_rejects_invalid_logits() {
    use super::processing::scores_from_logits;
    use std::error::Error;

    let scores = scores_from_logits(&[20.0, -20.0, 0.0, 1.0], 4, 128).unwrap();
    assert!(scores[0] > 0.999);
    assert!(scores[1] < 0.001);
    assert!((scores[2] - 0.5).abs() < 1e-7);
    assert!((scores[3] - 0.731_058_6).abs() < 1e-7);
    assert_eq!(
        scores_from_logits(&[f32::MAX, f32::MIN], 2, 128).unwrap(),
        [1.0, 0.0]
    );
    for wrong in [vec![0.0], vec![0.0; 3]] {
        assert!(matches!(
            scores_from_logits(&wrong, 2, 128),
            Err(RerankerError::Inference { source: None, .. })
        ));
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = scores_from_logits(&[0.0, invalid], 2, 128).unwrap_err();
        assert!(matches!(error, RerankerError::NonFiniteOutput));
        assert_eq!(
            error.to_string(),
            "non-finite values in reranker output (NaN or inf)"
        );
        assert!(error.source().is_none());
    }
}

#[test]
fn runtime_errors_preserve_typed_sources_and_display() {
    use mlx_rs::error::Exception;
    use std::error::Error;
    use tokenizers::models::wordlevel::{Error as WordLevelError, WordLevel};

    let cause = Exception::custom("forward failed");
    let location = cause.location();
    let display = format!("inference error: {cause}");
    let error = RerankerError::inference(cause);
    assert!(matches!(error, RerankerError::Inference { .. }));
    assert_eq!(error.to_string(), display);
    let source = error.source().unwrap();
    assert_eq!(
        source.downcast_ref::<Exception>().unwrap().location(),
        location
    );
    assert!(source.source().is_none());

    // Real tokenizer failure: missing [UNK] entry in an otherwise valid model.
    let tokenizer = tokenizers::Tokenizer::new(WordLevel::default());
    let cause = tokenizer.encode(("query", "document"), true).unwrap_err();
    let display = format!("tokenizer error: {cause}");
    let error = RerankerError::tokenizer(cause);
    assert!(matches!(error, RerankerError::Tokenizer { .. }));
    assert_eq!(error.to_string(), display);
    let source = error.source().unwrap();
    assert!(matches!(
        source.downcast_ref::<WordLevelError>(),
        Some(WordLevelError::MissingUnkToken)
    ));
    assert!(source.source().is_none());
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

    // Fixed-model API inference, not a general search-quality evaluation.
    // Reuse one load across singleton, ranking, and batch order contracts.
    #[test]
    #[ignore = "requires cached official reranker and unsandboxed MLX runtime"]
    #[serial]
    #[tracing_test::traced_test]
    fn cached_model_scores_singleton_batch_and_reranks_in_input_order() {
        require_unsandboxed_mlx_runtime();
        let reranker = Reranker::new(&load_cached_artifacts()).unwrap();
        assert!(reranker.score_batch(&[]).unwrap().is_empty());
        assert!(reranker.rerank("query", &[]).unwrap().is_empty());
        let score = reranker.score("test", "テスト文").unwrap();
        assert!((0.0..=1.0).contains(&score) && score.is_finite(), "{score}");

        let docs = ["related document", "unrelated text", "somewhat relevant"];
        let results = reranker.rerank("test query", &docs).unwrap();
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.score.is_finite() && (0.0..=1.0).contains(&result.score));
        }
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        let mut indices: Vec<_> = results.iter().map(|r| r.index).collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2]);

        let pairs = [
            ("東京の人口", "東京は日本最大の都市"),
            ("東京の人口", "りんごは果物"),
        ];
        let scores = reranker.score_batch(&pairs).unwrap();
        assert!(logs_contain("reranker score_batch dispatch"));
        for field in [
            "batch_size=2",
            "sub_batch_count=1",
            "sub_batch_size=2000",
            "bucket_len=128",
        ] {
            assert!(logs_contain(field), "missing {field}");
        }
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "related pair must score higher: {scores:?}"
        );
        assert!(
            scores
                .iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s))
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
}

#[test]
fn empty_inputs_bypass_inference_and_nonempty_inputs_delegate_in_order() {
    use super::processing::{rerank_with, score_batch_with};
    assert!(
        score_batch_with(&[], |_| panic!("empty batch reached inference"))
            .unwrap()
            .is_empty()
    );
    assert!(
        rerank_with("query", &[], |_| panic!("empty rerank reached inference"))
            .unwrap()
            .is_empty()
    );
    let pairs = [("query", "first"), ("query", "second")];
    assert_eq!(
        score_batch_with(&pairs, |actual| {
            assert_eq!(actual, pairs);
            Ok(vec![0.25, 0.75])
        })
        .unwrap(),
        vec![0.25, 0.75]
    );
    let ranked = rerank_with("query", &["first", "second"], |actual| {
        assert_eq!(actual, pairs);
        Ok(vec![0.25, 0.75])
    })
    .unwrap();
    assert_eq!(
        ranked
            .iter()
            .map(|r| (r.index, r.score))
            .collect::<Vec<_>>(),
        vec![(1, 0.75), (0, 0.25)]
    );
    for err in [
        score_batch_with(&pairs, |_| Err(RerankerError::NonFiniteOutput)).unwrap_err(),
        rerank_with("query", &["first"], |_| Err(RerankerError::NonFiniteOutput)).unwrap_err(),
    ] {
        assert!(matches!(err, RerankerError::NonFiniteOutput));
    }
}

#[tracing_test::traced_test]
#[test]
fn dispatch_plan_emits_batch_bucket_and_sub_batch_fields_without_model() {
    use super::processing::dispatch_sub_batch_size;
    assert_eq!(dispatch_sub_batch_size(3, 3), (128, 2000));
    assert!(logs_contain("reranker score_batch dispatch"));
    assert!(logs_contain("batch_size=3"));
    assert!(logs_contain("sub_batch_count=1"));
    assert!(logs_contain("sub_batch_size=2000"));
    assert!(logs_contain("bucket_len=128"));
    assert_eq!(dispatch_sub_batch_size(129, 501), (512, 500));
    assert!(logs_contain("sub_batch_count=2"));
}
