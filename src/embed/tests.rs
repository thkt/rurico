use super::mlx::shrink_chunk_to_fit;
use super::pooling::{l2_normalize, mean_pooling};
use super::probe::probe_env_to_paths;
use super::*;
use crate::model_io::{EOS_TOKEN_ID, artifacts_from_cache, load_tokenizer};
use crate::test_support::setup_fake_hf_cache;
use std::fs;

#[test]
fn mean_pooling_excludes_masked_tokens() {
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,       // token 0, mask=1
        5.0, 6.0, 7.0, 8.0,       // token 1, mask=1
        100.0, 100.0, 100.0, 100.0, // token 2, mask=0 (excluded)
    ];
    let mask = vec![1u32, 1, 0];
    let result = mean_pooling(&data, 3, 4, &mask);
    assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn mean_pooling_all_masked() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0u32, 0];
    let result = mean_pooling(&data, 2, 2, &mask);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn l2_normalize_produces_unit_norm() {
    let mut v = vec![3.0f32, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn l2_normalize_zero_vector() {
    let mut v = vec![0.0f32, 0.0];
    l2_normalize(&mut v);
    assert_eq!(v, vec![0.0, 0.0]);
}

#[test]
fn mean_pooling_single_token() {
    let data = vec![1.0f32, 2.0];
    let mask = vec![1u32];
    let result = mean_pooling(&data, 1, 2, &mask);
    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
fn mean_pooling_weighted_mask() {
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 0.0,   // token 0, mask=1
        3.0, 6.0,   // token 1, mask=2
    ];
    let mask = vec![1u32, 2];
    let result = mean_pooling(&data, 2, 2, &mask);
    let expected_0 = 7.0 / 3.0;
    let expected_1 = 4.0;
    assert!((result[0] - expected_0).abs() < 1e-6, "[0]={}", result[0]);
    assert!((result[1] - expected_1).abs() < 1e-6, "[1]={}", result[1]);
}

#[test]
fn postprocess_embedding_zero_seq_len() {
    let result = postprocess_embedding(&[], 0, &[]);
    let err = result.unwrap_err();
    assert!(matches!(err, EmbedError::EmptySequence), "{err}");
}

#[test]
fn postprocess_embedding_accepts_any_dims() {
    for hidden_size in [3, 64, 256, 384, 512, 768] {
        let seq_len = 2;
        let data = vec![1.0f32; seq_len * hidden_size];
        let mask = vec![1u32; seq_len];
        let result = postprocess_embedding(&data, seq_len, &mask);
        assert!(
            result.is_ok(),
            "hidden_size={hidden_size} should be accepted, got: {:?}",
            result.unwrap_err()
        );
        assert_eq!(result.unwrap().len(), hidden_size);
    }
}

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

#[test]
fn mean_pooling_short_mask_truncates_safely() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![1u32];
    let result = mean_pooling(&data, 2, 2, &mask);
    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
fn postprocess_embedding_rejects_indivisible_length() {
    let hidden_size = EMBEDDING_DIMS;
    let seq_len = 2;
    let data = vec![0.0f32; seq_len * hidden_size + 1];
    let mask = vec![1u32; seq_len];
    let err = postprocess_embedding(&data, seq_len, &mask).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("not divisible")),
        "{err}"
    );
}

#[test]
fn postprocess_embedding_happy_path() {
    let hidden_size = EMBEDDING_DIMS;
    let seq_len = 2;
    let mut data = vec![0.0f32; seq_len * hidden_size];
    data[0] = 1.0;
    data[1] = 2.0;
    data[hidden_size] = 3.0;
    data[hidden_size + 1] = 4.0;
    let mask = vec![1u32; seq_len];

    let result = postprocess_embedding(&data, seq_len, &mask).unwrap();
    assert_eq!(result.len(), hidden_size);

    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");

    let ratio = result[0] / result[1];
    let expected_ratio = 2.0 / 3.0;
    assert!((ratio - expected_ratio).abs() < 1e-5, "ratio={ratio}");
}

#[test]
fn postprocess_embedding_rejects_nan_output() {
    let hidden_size = EMBEDDING_DIMS;
    let seq_len = 1;
    let mut data = vec![0.0f32; seq_len * hidden_size];
    data[0] = f32::NAN;
    let mask = vec![1u32; seq_len];
    let err = postprocess_embedding(&data, seq_len, &mask).unwrap_err();
    assert!(
        matches!(err, EmbedError::NonFiniteOutput),
        "expected NonFiniteOutput, got: {err}"
    );
}

#[test]
fn postprocess_embedding_rejects_inf_output() {
    let hidden_size = EMBEDDING_DIMS;
    let seq_len = 1;
    let mut data = vec![0.0f32; seq_len * hidden_size];
    data[0] = f32::INFINITY;
    let mask = vec![1u32; seq_len];
    let err = postprocess_embedding(&data, seq_len, &mask).unwrap_err();
    assert!(
        matches!(err, EmbedError::NonFiniteOutput),
        "expected NonFiniteOutput, got: {err}"
    );
}

#[test]
fn postprocess_embedding_rejects_short_attention_mask() {
    // SF-001 regression: short mask must error (not silently produce wrong embedding)
    let hidden_size = EMBEDDING_DIMS;
    let seq_len = 3;
    let data = vec![1.0f32; seq_len * hidden_size];
    let short_mask = vec![1u32; seq_len - 1]; // one entry too few
    let err = postprocess_embedding(&data, seq_len, &short_mask).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("attention_mask length")),
        "expected attention_mask length error, got: {err}"
    );
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
    let dir = tempfile::tempdir().unwrap();
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, ModelId::default()).unwrap();
    assert!(result.is_none());
}

#[test]
fn cache_lookup_returns_some_when_all_files_present() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache_for(dir.path(), ModelId::default());
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, ModelId::default()).unwrap();
    let paths = result.expect("should return Some when all files cached");
    assert!(paths.model.ends_with("model.safetensors"));
    assert!(paths.config.ends_with("config.json"));
    assert!(paths.tokenizer.ends_with("tokenizer.json"));
}

#[test]
fn cache_lookup_returns_none_when_partial() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache_for(dir.path(), ModelId::default());
    let repo_slug = ModelId::default().repo_id().replace('/', "--");
    let snapshot_dir = dir
        .path()
        .join(format!("models--{repo_slug}/snapshots/abc123"));
    fs::remove_file(snapshot_dir.join("config.json")).unwrap();
    fs::remove_file(snapshot_dir.join("tokenizer.json")).unwrap();

    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, ModelId::default()).unwrap();
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
        let cache = hf_hub::Cache::new(dir.path().to_path_buf());

        // The populated model should be found
        assert!(
            artifacts_from_cache(&cache, target).unwrap().is_some(),
            "{:?} should be cached",
            target
        );
        // All other models should not be found
        for &other in &all_models {
            if other == target {
                continue;
            }
            assert!(
                artifacts_from_cache(&cache, other).unwrap().is_none(),
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

#[test]
fn probe_env_to_paths_returns_paths_when_all_present() {
    let candidate = probe_env_to_paths(Some("/m".into()), Some("/c".into()), Some("/t".into()))
        .unwrap()
        .unwrap();
    // paths field is accessible within the embed module (child module can access parent's private)
    assert_eq!(candidate.paths.model, PathBuf::from("/m"));
    assert_eq!(candidate.paths.config, PathBuf::from("/c"));
    assert_eq!(candidate.paths.tokenizer, PathBuf::from("/t"));
}

#[test]
fn shrink_chunk_to_fit_rejects_empty_range() {
    // end <= start → Inference error without calling tokenizer
    let dir = tempfile::TempDir::new().unwrap();
    let tok_path = dir.path().join("tokenizer.json");
    fs::write(
        &tok_path,
        r#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#,
    )
    .unwrap();
    let tokenizer = load_tokenizer(&tok_path).unwrap();
    let offsets = vec![(0, 5)];
    let mut end = 0usize;
    let err = shrink_chunk_to_fit(&tokenizer, "hello", &offsets, 0, &mut end).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("cannot fit")),
        "{err}"
    );
}

#[test]
fn shrink_chunk_to_fit_short_text_returns_immediately() {
    // [TC-006] Convergence happy path: text already fits MAX_SEQ_LEN → returns on
    // first iteration without decrementing end.
    let dir = tempfile::TempDir::new().unwrap();
    let tok_path = dir.path().join("tokenizer.json");
    fs::write(
        &tok_path,
        r#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#,
    )
    .unwrap();
    let tokenizer = load_tokenizer(&tok_path).unwrap();

    let text = "hello";
    // Character-level offsets: each byte maps to a (start, end) span
    let offsets: Vec<(usize, usize)> = text
        .char_indices()
        .map(|(s, c)| (s, s + c.len_utf8()))
        .collect();
    let n = offsets.len();

    let mut end = n;
    let result = shrink_chunk_to_fit(&tokenizer, text, &offsets, 0, &mut end);
    assert!(
        result.is_ok(),
        "short text should fit MAX_SEQ_LEN: {:?}",
        result.err()
    );
    // end must be unchanged (no shrinking occurred)
    assert_eq!(end, n, "end should not be decremented for short text");
}

// --- T-001: max_content computation ---

#[test]
fn t_001_max_content_equals_max_seq_len_minus_2_minus_prefix_len() {
    // [T-001] FR-001, FR-002: max_content(10) == 8180, 8180 + 10 + 2 == 8192
    assert_eq!(max_content(10), 8180);
    assert_eq!(8180 + 10 + 2, MAX_SEQ_LEN);
}

#[test]
#[ignore] // requires model download
fn g_001_real_tokenizer_extract_prefix_tokens() {
    let artifacts = download_model(ModelId::default()).expect("download model");
    let tokenizer = load_tokenizer(&artifacts.paths.tokenizer).unwrap();
    let prefix_tokens = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    assert!(!prefix_tokens.is_empty());
}

#[test]
fn t_008_truncate_for_query_shortens_to_max_len() {
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

#[test]
fn t_008b_truncate_for_query_noop_when_short() {
    let input_ids: Vec<u32> = vec![1, 10, 20, 2]; // BOS, text, text, EOS
    let attention_mask = vec![1u32; 4];
    let expected_ids = input_ids.clone();
    let expected_mask = attention_mask.clone();

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 50);
    assert_eq!(seq_len, 4);
    assert_eq!(ids, expected_ids);
    assert_eq!(mask, expected_mask);
}

#[test]
fn t_008c_truncate_for_query_noop_at_exact_boundary() {
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

#[test]
fn t_008e_truncate_for_query_max_len_1_produces_eos_only() {
    // [TC-003] max_len=1: the only slot is overwritten with EOS → output is [EOS]
    let input_ids: Vec<u32> = vec![1, 100, 200, 2]; // BOS, text, text, EOS
    let attention_mask = vec![1u32; 4];

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 1);
    assert_eq!(seq_len, 1);
    assert_eq!(ids, vec![EOS_TOKEN_ID], "single slot must be EOS");
    assert_eq!(mask.len(), 1);
}

#[test]
fn t_008d_truncate_for_query_zero_max_len_returns_unchanged() {
    let input_ids: Vec<u32> = vec![1, 100, 200, 2];
    let attention_mask = vec![1u32; 4];
    let expected_ids = input_ids.clone();
    let expected_mask = attention_mask.clone();

    let (ids, mask, seq_len) = truncate_for_query(input_ids, attention_mask, 0);
    assert_eq!(seq_len, 4);
    assert_eq!(ids, expected_ids, "max_len=0 should return unchanged");
    assert_eq!(mask, expected_mask);
}

#[test]
fn t_014_mock_chunked_embedder_returns_multi_chunk() {
    let embedder = super::MockChunkedEmbedder::new(3);
    let result = embedder.embed_document("some text").unwrap();
    assert_eq!(result.chunks.len(), 3);
    assert_eq!(result.chunks[0].len(), EMBEDDING_DIMS);
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

#[test]
#[ignore] // requires model download
fn regression_prefix_merge_standalone_vs_full_tokenization_diverges() {
    // Verify that the prefix boundary actually diverges for these texts,
    // confirming the need for Approach A.
    let artifacts = download_model(ModelId::default()).expect("download model");
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
#[ignore] // requires model download
fn regression_long_document_sequential_planner_overlap_and_coverage() {
    // Mirrors the production sequential chunk planner. Verifies:
    // 1. Every chunk fits MAX_SEQ_LEN (adaptive shrink)
    // 2. Adjacent chunks overlap >= CHUNK_OVERLAP_TOKENS (overlap contract)
    // 3. First chunk starts at byte 0 (head preserved)
    // 4. Last chunk ends at document end (tail preserved)
    let artifacts = download_model(ModelId::default()).expect("download model");
    let tokenizer = load_tokenizer(&artifacts.paths.tokenizer).unwrap();
    let prefix_tokens = extract_prefix_tokens(&tokenizer, DOCUMENT_PREFIX).unwrap();
    let mc = max_content(prefix_tokens.len());

    let sentence = "apple pie is a traditional dessert enjoyed around the world. ";
    let long_text = sentence.repeat(800);

    let text_enc = tokenizer.encode(long_text.as_str(), false).unwrap();
    let offsets = text_enc.get_offsets();
    let n = text_enc.get_ids().len();
    assert!(n > mc, "text must trigger chunking: {n} <= {mc}");

    // Sequential planner (same logic as production embed_documents_batch_chunked)
    let mut chunk_starts: Vec<usize> = Vec::new();
    let mut accepted_ends: Vec<usize> = Vec::new();
    let mut start = 0usize;

    while start < n {
        chunk_starts.push(start);
        let byte_start = offsets[start].0;
        let mut end = (start + mc).min(n);

        loop {
            assert!(end > start, "adaptive shrink underflow at token {start}");
            let byte_end = offsets[end - 1].1;
            let chunk_str = format!("{DOCUMENT_PREFIX}{}", &long_text[byte_start..byte_end]);
            let enc = tokenizer.encode(chunk_str.as_str(), true).unwrap();
            if enc.get_ids().len() <= MAX_SEQ_LEN {
                assert_eq!(
                    *enc.get_ids().last().unwrap(),
                    EOS_TOKEN_ID,
                    "chunk {}: natural EOS expected",
                    chunk_starts.len() - 1
                );
                break;
            }
            end -= 1;
        }
        accepted_ends.push(end);

        if end >= n {
            break;
        }
        let next_start = end.saturating_sub(CHUNK_OVERLAP_TOKENS);
        if next_start <= start {
            break;
        }
        start = next_start;
    }

    assert!(chunk_starts.len() >= 2, "expected multiple chunks");

    // Head preserved
    assert_eq!(chunk_starts[0], 0);
    assert!(long_text.starts_with("apple"));

    // Tail preserved
    assert_eq!(
        *accepted_ends.last().unwrap(),
        n,
        "last chunk must reach document end"
    );

    // Overlap contract
    for i in 0..chunk_starts.len() - 1 {
        let overlap = accepted_ends[i].saturating_sub(chunk_starts[i + 1]);
        assert!(
            overlap >= CHUNK_OVERLAP_TOKENS,
            "chunks {i} and {}: overlap {overlap} < {CHUNK_OVERLAP_TOKENS}",
            i + 1
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
fn embed_text_returns_correct_dimensionality_for_all_prefixes() {
    let e = super::MockEmbedder::default();
    for prefix in [SEMANTIC_PREFIX, TOPIC_PREFIX, QUERY_PREFIX, DOCUMENT_PREFIX] {
        let vec = e.embed_text("テスト", prefix).unwrap();
        assert_eq!(
            vec.len(),
            EMBEDDING_DIMS,
            "wrong dims for prefix: {prefix:?}"
        );
    }
}

#[test]
fn embed_text_propagates_error() {
    let e = super::FailingEmbedder::all_fail("embed_text error");
    let err = e.embed_text("テスト", SEMANTIC_PREFIX).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("embed_text error")),
        "{err}"
    );
}

#[test]
fn from_probe_error_maps_correctly() {
    use crate::model_probe::ProbeError;

    let err: EmbedInitError = ProbeError::HandlerNotInstalled.into();
    assert!(
        matches!(err, EmbedInitError::Backend(ref m) if m.contains("probe handler not installed")),
        "{err}"
    );

    let err: EmbedInitError = ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    assert!(
        matches!(err, EmbedInitError::ModelCorrupt { ref reason } if reason == "bad weights"),
        "{err}"
    );

    let err: EmbedInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
    assert!(
        matches!(err, EmbedInitError::Backend(ref m) if m == "spawn failed"),
        "{err}"
    );
}
