use super::*;
use crate::artifacts::EmbedKind;
use crate::model_io::{EOS_TOKEN_ID, ModelArtifact, artifacts_from_cache, load_tokenizer};
#[cfg(unix)]
use crate::test_support::assert_probe_env_to_paths_preserves_snapshot_symlink_filename;
use crate::test_support::{
    assert_cache_lookup_returns_none_when_empty,
    assert_cache_lookup_returns_some_when_all_files_present,
    assert_from_probe_error_maps_correctly,
    assert_probe_env_to_paths_returns_paths_when_all_present, hf_client_for_cache,
    setup_fake_hf_cache,
};
use std::error::Error;
use std::fs;
use std::path::Path;

#[test]
fn validate_partial_download_reports_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    let ArtifactError::MissingFile { path } = &err else {
        panic!("{err}");
    };
    assert!(path.ends_with("config.json"), "{path:?}");
}

fn setup_fake_cache_for(hub_dir: &Path, model: ModelId) {
    setup_fake_hf_cache(
        hub_dir,
        model.repo_id(),
        model.revision(),
        &[
            ("model.safetensors", b"fake"),
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
        ],
    );
}

#[test]
fn cache_lookup_returns_none_when_empty() {
    assert_cache_lookup_returns_none_when_empty(ModelId::DEFAULT);
}

#[test]
fn cache_lookup_returns_some_when_all_files_present() {
    assert_cache_lookup_returns_some_when_all_files_present(ModelId::DEFAULT);
}

#[test]
fn cache_lookup_returns_none_when_partial() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache_for(dir.path(), ModelId::DEFAULT);
    let repo_slug = ModelId::DEFAULT.repo_id().replace('/', "--");
    let snapshot_dir = dir.path().join(format!(
        "models--{repo_slug}/snapshots/{}",
        ModelId::DEFAULT.revision()
    ));
    fs::remove_file(snapshot_dir.join("config.json")).unwrap();
    fs::remove_file(snapshot_dir.join("tokenizer.json")).unwrap();

    let client = hf_client_for_cache(dir.path());
    let result = artifacts_from_cache(&client, ModelId::DEFAULT).unwrap();
    assert!(result.is_none());
}

#[test]
fn cache_lookup_each_model_has_separate_cache_dir() {
    let all_models = [
        ModelId::RuriV3_30m,
        ModelId::RuriV3_70m,
        ModelId::RuriV3_130m,
        ModelId::RuriV3_310m,
    ];

    for &target in &all_models {
        let dir = tempfile::tempdir().unwrap();
        setup_fake_cache_for(dir.path(), target);
        let client = hf_client_for_cache(dir.path());

        // The populated model should be found
        assert!(
            artifacts_from_cache(&client, target).unwrap().is_some(),
            "{:?} should be cached",
            target
        );
        // All other models should not be found
        for &other in &all_models {
            if other == target {
                continue;
            }
            assert!(
                artifacts_from_cache(&client, other).unwrap().is_none(),
                "{:?} should not be cached when only {:?} is populated",
                other,
                target
            );
        }
    }
}

#[test]
fn candidate_verify_returns_invalid_config_for_malformed_config() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    fs::write(dir.path().join("config.json"), b"not json").unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidConfig { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

// T-018: regression — probe_env_to_paths must return the snapshot symlink path,
// not the canonicalized blob path. Failed branch fix/issue-107-canonicalize
// (commit 33c91c1) violated this and broke MLX `load_safetensors`'s
// extension-based dispatch across all standard HF cache usage.
#[cfg(unix)]
#[test]
fn probe_env_to_paths_preserves_snapshot_symlink_filename() {
    assert_probe_env_to_paths_preserves_snapshot_symlink_filename::<EmbedKind>();
}

#[test]
fn probe_env_to_paths_returns_paths_when_all_present() {
    assert_probe_env_to_paths_returns_paths_when_all_present::<EmbedKind>();
}

// --- T-001: max_content computation ---

// T-001: max_content_equals_max_seq_len_minus_2_minus_prefix_len
#[test]
fn max_content_equals_max_seq_len_minus_2_minus_prefix_len() {
    // [T-001] FR-001, FR-002: max_content reserves BOS(1) + EOS(1) + prefix.
    // 8180 = MAX_SEQ_LEN(8192) - 2 - prefix_len(10).
    assert_eq!(max_content(10), 8180);
}

/// Requires HF Hub model download (network access).
#[test]
#[ignore]
fn g_001_real_tokenizer_extract_prefix_tokens() {
    let artifacts = download_model(ModelId::DEFAULT).expect("download model");
    let tokenizer = load_tokenizer(&artifacts.paths.tokenizer).unwrap();
    let prefix_tokens = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    assert!(!prefix_tokens.is_empty());
}

// T-008: truncate_for_query_shortens_to_max_len
#[test]
fn truncate_for_query_shortens_to_max_len() {
    // [T-008] FR-005: query truncation
    let input_ids: Vec<u32> = (0..100).collect();
    let first_token = input_ids[0];
    let attention_mask = vec![1u32; 100];
    let max_len = 50;

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, max_len);
    assert_eq!(seq_len, max_len);
    assert_eq!(ids.len(), max_len);
    assert_eq!(mask.len(), max_len);
    // Last token should be EOS
    assert_eq!(ids[max_len - 1], EOS_TOKEN_ID);
    // First token preserved (BOS)
    assert_eq!(ids[0], first_token);
}

// T-008b: truncate_for_query_noop_when_short
#[test]
fn truncate_for_query_noop_when_short() {
    let input_ids: Vec<u32> = vec![1, 10, 20, 2]; // BOS, text, text, EOS
    let attention_mask = vec![1u32; 4];
    let expected_ids = input_ids.clone();
    let expected_mask = attention_mask.clone();

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 50);
    assert_eq!(seq_len, 4);
    assert_eq!(ids, expected_ids);
    assert_eq!(mask, expected_mask);
}

// T-008c: truncate_for_query_noop_at_exact_boundary
#[test]
fn truncate_for_query_noop_at_exact_boundary() {
    // [TC-003] Boundary: len == max_len → no truncation
    let max_len = 50;
    #[allow(clippy::cast_possible_truncation)]
    let input_ids: Vec<u32> = (0..max_len as u32).collect();
    let attention_mask = vec![1u32; max_len];
    let expected_ids = input_ids.clone();
    let expected_mask = attention_mask.clone();

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, max_len);
    assert_eq!(seq_len, max_len);
    assert_eq!(ids, expected_ids, "exact boundary should not truncate");
    assert_eq!(mask, expected_mask);
}

// T-008e: truncate_for_query_max_len_1_produces_eos_only
#[test]
fn truncate_for_query_max_len_1_produces_eos_only() {
    // [TC-003] max_len=1: the only slot is overwritten with EOS → output is [EOS]
    let input_ids: Vec<u32> = vec![1, 100, 200, 2]; // BOS, text, text, EOS
    let attention_mask = vec![1u32; 4];

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 1);
    assert_eq!(seq_len, 1);
    assert_eq!(ids, vec![EOS_TOKEN_ID], "single slot must be EOS");
    assert_eq!(mask.len(), 1);
}

// T-008d: truncate_for_query_zero_max_len_returns_unchanged
#[test]
fn truncate_for_query_zero_max_len_returns_unchanged() {
    let input_ids: Vec<u32> = vec![1, 100, 200, 2];
    let attention_mask = vec![1u32; 4];
    let expected_ids = input_ids.clone();
    let expected_mask = attention_mask.clone();

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 0);
    assert_eq!(seq_len, 4);
    assert_eq!(ids, expected_ids, "max_len=0 should return unchanged");
    assert_eq!(mask, expected_mask);
}

// --- Regression: prefix boundary merge cases ---
//
// The tokenizer merges the trailing space of DOCUMENT_PREFIX with the first
// text token for many inputs (e.g. "apple", "the", "Rust"). These tests
// verify that chunked embedding handles such texts correctly by re-tokenizing
// each chunk with the prefix (Approach A).
//
// MLX-dependent embedding tests (short text, long text, batch, prefix-merge)
// are covered by `tests/mlx_smoke.rs` via subprocess isolation.

/// Requires HF Hub model download (network access).
#[test]
#[ignore]
fn regression_prefix_merge_standalone_vs_full_tokenization_diverges() {
    // Verify that the prefix boundary actually diverges for these texts,
    // confirming the need for Approach A.
    let artifacts = download_model(ModelId::DEFAULT).expect("download model");
    let tokenizer = load_tokenizer(&artifacts.paths.tokenizer).unwrap();
    let prefix_tokens = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    let pe = 1 + prefix_tokens.len(); // BOS + prefix length

    let merge_texts = &["apple pie", "the cat", "Rust"];
    for &text in merge_texts {
        let full = tokenize_with_prefix(&tokenizer, text, DOCUMENT_PREFIX).unwrap();
        let ids = &full.input_ids;
        // The prefix portion in full tokenization should NOT match standalone
        // prefix tokens — this confirms the merge behavior exists.
        let full_prefix = &ids[1..pe.min(ids.len())];
        assert_ne!(
            full_prefix,
            &prefix_tokens[..],
            "expected prefix merge for text '{text}', but tokens matched — \
             this test guards against the merge being silently fixed upstream"
        );
    }
}

#[test]
fn embed_documents_batch_empty_returns_empty() {
    let embedder = super::MockEmbedder::default();
    let result = embedder.embed_documents_batch(&[]).unwrap();
    assert!(result.is_empty(), "empty input should return empty vec");
}

// --- 1+3 prefix scheme ---

#[test]
fn embed_text_returns_embedding_dims() {
    // MockEmbedder ignores prefix, so a single call exercises the dim contract.
    let e = super::MockEmbedder::default();
    let vec = e.embed_text("テスト", SEMANTIC_PREFIX).unwrap();
    assert_eq!(vec.len(), EMBEDDING_DIMS);
}

#[test]
fn from_probe_error_maps_correctly() {
    assert_from_probe_error_maps_correctly();
}

#[derive(Debug, thiserror::Error)]
#[error("outer embed failure")]
struct OuterEmbedTestError {
    #[source]
    source: InnerEmbedTestError,
}

#[derive(Debug, thiserror::Error)]
#[error("inner embed failure")]
struct InnerEmbedTestError;

#[test]
fn embed_error_helpers_preserve_source_chain() {
    let inference = EmbedError::inference(OuterEmbedTestError {
        source: InnerEmbedTestError,
    });
    assert!(
        matches!(
            inference,
            EmbedError::Inference {
                ref message,
                source: Some(_),
            } if message == "outer embed failure"
        ),
        "expected Inference with boxed source, got: {inference:?}"
    );
    let inference_source = inference.source().expect("inference source");
    assert_eq!(inference_source.to_string(), "outer embed failure");
    assert_eq!(
        inference_source
            .source()
            .expect("nested source")
            .to_string(),
        "inner embed failure"
    );

    let tokenizer = EmbedError::tokenizer(OuterEmbedTestError {
        source: InnerEmbedTestError,
    });
    assert!(
        matches!(
            tokenizer,
            EmbedError::Tokenizer {
                ref message,
                source: Some(_),
            } if message == "outer embed failure"
        ),
        "expected Tokenizer with boxed source, got: {tokenizer:?}"
    );
    let tokenizer_source = tokenizer.source().expect("tokenizer source");
    assert_eq!(tokenizer_source.to_string(), "outer embed failure");
    assert_eq!(
        tokenizer_source
            .source()
            .expect("nested source")
            .to_string(),
        "inner embed failure"
    );
}

// T-187-001: empty chunk lists are rejected by the public constructor.
#[test]
fn chunked_embedding_try_new_rejects_empty_input() {
    let err = ChunkedEmbedding::try_new(Vec::new()).unwrap_err();
    assert_eq!(err, EmptyChunksError);
}

#[test]
fn chunked_embedding_try_new_generates_matching_chunk_ids() {
    let ce = ChunkedEmbedding::try_new(vec![vec![0.0], vec![1.0]]).unwrap();
    assert_eq!(ce.chunk_ids(), ["c0", "c1"]);
}

// ── EmbedOptions tests ──────────────────────────────────────────────────────

#[test]
fn embed_options_default_has_no_budget_override_and_no_pause() {
    let opts = EmbedOptions::default();
    assert_eq!(opts.token_budget, None);
    assert_eq!(opts.forward_pause, None);
}

#[test]
fn embed_documents_batch_with_options_default_impl_ignores_options() {
    let embedder = MockEmbedder::default();
    let texts = ["alpha", "beta"];
    let opts = EmbedOptions {
        token_budget: Some(2048),
        forward_pause: Some(Duration::from_millis(700)),
    };
    let with_opts = embedder
        .embed_documents_batch_with_options(&texts, &opts)
        .unwrap();
    let without = embedder.embed_documents_batch(&texts).unwrap();
    assert_eq!(with_opts.len(), without.len());
    for (a, b) in with_opts.iter().zip(&without) {
        assert_eq!(a.chunks(), b.chunks());
    }
}
