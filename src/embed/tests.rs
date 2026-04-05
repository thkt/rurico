use super::mlx::{shrink_chunk_to_fit, unpack_batch_output};
use super::pooling::{l2_normalize, mean_pooling};
use super::probe::probe_env_to_paths;
use super::*;
use crate::model_io::load_tokenizer;

// BOS token ID for ruri-v3 (test-only; used in build_token_chunks and chunk structure assertions)
const BOS_TOKEN_ID: u32 = 1;

/// Compute chunk start positions within the text token array.
///
/// Returns a list of start indices. Each chunk reads `max_content` tokens
/// starting from the given position. The last chunk's start is adjusted
/// so that it ends exactly at the end of the text tokens (stretch-to-fill).
///
/// For texts that fit in a single chunk, returns `[0]`.
///
/// NOTE: This function verifies the geometric stride invariant
/// (stride = max_content − overlap). Production chunking in `mlx.rs`
/// uses a sequential planner with tokenizer re-encoding, which may
/// accept fewer tokens per chunk due to prefix boundary merging.
/// The two algorithms share the overlap/coverage contract but differ
/// in how chunk boundaries are determined.
fn compute_chunk_starts(text_token_count: usize, max_content: usize, overlap: usize) -> Vec<usize> {
    if text_token_count <= max_content {
        return vec![0];
    }

    let stride = max_content.saturating_sub(overlap);
    if stride == 0 {
        return vec![0];
    }

    let mut starts = Vec::new();
    let mut pos = 0;
    while pos + max_content < text_token_count {
        starts.push(pos);
        pos += stride;
    }
    // Last chunk: stretch to fill max_content
    starts.push(text_token_count - max_content);
    starts
}

/// Build token chunks from text tokens, adding BOS, prefix, and EOS to each.
///
/// Each chunk has the structure: `[BOS, prefix..., text_slice..., EOS]`.
/// The text slice length is at most `max_content` tokens.
fn build_token_chunks(
    text_tokens: &[u32],
    prefix_tokens: &[u32],
    starts: &[usize],
    max_content: usize,
) -> Vec<Vec<u32>> {
    starts
        .iter()
        .map(|&start| {
            let end = (start + max_content).min(text_tokens.len());
            let mut chunk = Vec::with_capacity(2 + prefix_tokens.len() + (end - start));
            chunk.push(BOS_TOKEN_ID);
            chunk.extend_from_slice(prefix_tokens);
            chunk.extend_from_slice(&text_tokens[start..end]);
            chunk.push(EOS_TOKEN_ID);
            chunk
        })
        .collect()
}

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
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
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

#[test]
fn unpack_batch_output_rejects_nan_embedding() {
    let hidden_size = EMBEDDING_DIMS;
    let max_seq_len = 1;
    let batch_size = 1;
    let mut flat = vec![0.0f32; batch_size * max_seq_len * hidden_size];
    flat[0] = f32::NAN;
    let attention_mask = vec![1u32; batch_size * max_seq_len];
    let err = unpack_batch_output(&flat, batch_size, max_seq_len, &attention_mask).unwrap_err();
    assert!(
        matches!(err, EmbedError::NonFiniteOutput),
        "expected NonFiniteOutput, got: {err}"
    );
}

fn setup_fake_cache_for(hub_dir: &std::path::Path, model: ModelId) {
    crate::test_support::setup_fake_hf_cache(
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
    let result = crate::model_io::artifacts_from_cache(&cache, ModelId::default()).unwrap();
    assert!(result.is_none());
}

#[test]
fn cache_lookup_returns_some_when_all_files_present() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache_for(dir.path(), ModelId::default());
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = crate::model_io::artifacts_from_cache(&cache, ModelId::default()).unwrap();
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
    std::fs::remove_file(snapshot_dir.join("config.json")).unwrap();
    std::fs::remove_file(snapshot_dir.join("tokenizer.json")).unwrap();

    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = crate::model_io::artifacts_from_cache(&cache, ModelId::default()).unwrap();
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
            crate::model_io::artifacts_from_cache(&cache, target)
                .unwrap()
                .is_some(),
            "{:?} should be cached",
            target
        );
        // All other models should not be found
        for &other in &all_models {
            if other == target {
                continue;
            }
            assert!(
                crate::model_io::artifacts_from_cache(&cache, other)
                    .unwrap()
                    .is_none(),
                "{:?} should not be cached when only {:?} is populated",
                other,
                target
            );
        }
    }
}

#[test]
fn model_id_repo_ids_are_distinct() {
    use std::collections::HashSet;
    let all_models = [
        ModelId::RuriV3_30m,
        ModelId::RuriV3_70m,
        ModelId::RuriV3_130m,
        ModelId::RuriV3_310m,
    ];
    let repo_ids: HashSet<_> = all_models.iter().map(|m| m.repo_id()).collect();
    assert_eq!(repo_ids.len(), 4, "all repo IDs must be distinct");

    let revisions: HashSet<_> = all_models.iter().map(|m| m.revision()).collect();
    assert_eq!(revisions.len(), 4, "all revisions must be distinct");
}

#[test]
fn model_id_default_is_310m() {
    assert_eq!(ModelId::default(), ModelId::RuriV3_310m);
    assert_eq!(ModelId::default().repo_id(), "cl-nagoya/ruri-v3-310m");
}

#[test]
fn candidate_verify_returns_invalid_config_for_malformed_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("config.json"), b"not json").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let candidate = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidConfig { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

#[test]
fn probe_env_to_paths_returns_none_when_model_absent() {
    assert!(probe_env_to_paths(None, None, None).is_none());
}

#[test]
fn probe_env_to_paths_returns_err_when_config_missing() {
    let result = probe_env_to_paths(Some("/m".into()), None, Some("/t".into()));
    assert!(matches!(result, Some(Err(3))));
}

#[test]
fn probe_env_to_paths_returns_err_when_tokenizer_missing() {
    let result = probe_env_to_paths(Some("/m".into()), Some("/c".into()), None);
    assert!(matches!(result, Some(Err(3))));
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
fn unpack_batch_output_rejects_indivisible_shape() {
    let flat = vec![0.0f32; 10];
    let mask = vec![1u32; 6];
    let err = unpack_batch_output(&flat, 2, 3, &mask).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::BufferShapeMismatch {
                expected: 6,
                actual: 10
            }
        ),
        "{err}"
    );
}

#[test]
fn unpack_batch_output_rejects_zero_total() {
    let flat = vec![0.0f32; 10];
    let err = unpack_batch_output(&flat, 0, 0, &[]).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::BufferShapeMismatch {
                expected: 0,
                actual: 10
            }
        ),
        "{err}"
    );
}

#[test]
fn unpack_batch_output_happy_path() {
    let hidden = EMBEDDING_DIMS;
    let batch_size = 2;
    let max_seq_len = 1;
    let mut flat = vec![0.0f32; batch_size * max_seq_len * hidden];
    // chunk 0: nonzero at dim 1
    flat[1] = 1.0;
    // chunk 1: nonzero at dim 0
    flat[hidden] = 1.0;
    let mask = vec![1u32; batch_size * max_seq_len];
    let results = unpack_batch_output(&flat, batch_size, max_seq_len, &mask).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), hidden);
    assert_eq!(results[1].len(), hidden);
    // results[0] from chunk 0 (dim 1 nonzero)
    assert!(
        results[0][0].abs() < 1e-6,
        "results[0][0]={}",
        results[0][0]
    );
    assert!(
        (results[0][1] - 1.0).abs() < 1e-6,
        "results[0][1]={}",
        results[0][1]
    );
    // results[1] from chunk 1 (dim 0 nonzero)
    assert!(
        (results[1][0] - 1.0).abs() < 1e-6,
        "results[1][0]={}",
        results[1][0]
    );
    assert!(
        results[1][1].abs() < 1e-6,
        "results[1][1]={}",
        results[1][1]
    );
}

// --- Chunked Embedding (Phase 1) ---

#[test]
fn t_004_short_text_produces_single_chunk_start() {
    // 100 text tokens, max_content=8000, overlap=2048 → single chunk at position 0
    let starts = compute_chunk_starts(100, 8000, 2048);
    assert_eq!(starts, vec![0]);
}

#[test]
fn t_001_long_text_produces_multiple_chunk_starts() {
    // 10000 text tokens, max_content=8000, overlap=2048
    // stride = 8000 - 2048 = 5952
    // starts: [0, 5952], but last must be 10000-8000=2000
    // So: [0, 2000]
    let starts = compute_chunk_starts(10000, 8000, 2048);
    assert_eq!(starts.len(), 2);
    assert!(
        starts
            .iter()
            .all(|&s| s + 8000 <= 10000 || s == 10000 - 8000)
    );
    // Last start fills max_content
    assert_eq!(*starts.last().unwrap(), 10000 - 8000);
}

#[test]
fn t_013_boundary_one_over_produces_two_chunks() {
    // text_token_count = max_content + 1 → 2 chunks
    let max_content = 8000;
    let starts = compute_chunk_starts(max_content + 1, max_content, 2048);
    assert_eq!(starts.len(), 2);
    // Last start = 1 (8001 - 8000), overlap is automatically expanded
    assert_eq!(*starts.last().unwrap(), 1);
}

#[test]
fn t_003_last_chunk_fills_max_content() {
    // 8300 text tokens, max_content=8000
    // Last start = 8300 - 8000 = 300
    let starts = compute_chunk_starts(8300, 8000, 2048);
    assert_eq!(*starts.last().unwrap(), 300);
}

#[test]
fn t_012_empty_text_produces_single_chunk_start() {
    let starts = compute_chunk_starts(0, 8000, 2048);
    assert_eq!(starts, vec![0]);
}

#[test]
fn compute_chunk_starts_zero_stride_returns_single() {
    // overlap >= max_content → stride saturates to 0 → single chunk
    let starts = compute_chunk_starts(10000, 100, 200);
    assert_eq!(starts, vec![0]);
}

#[test]
fn shrink_chunk_to_fit_rejects_empty_range() {
    // end <= start → Inference error without calling tokenizer
    let dir = tempfile::TempDir::new().unwrap();
    let tok_path = dir.path().join("tokenizer.json");
    std::fs::write(
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
    std::fs::write(
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

#[test]
fn t_005_each_chunk_has_bos_and_eos() {
    let prefix_tokens = &[100u32, 200];
    let text_tokens: Vec<u32> = (10..10_010).collect(); // 10000 text tokens
    let max_content = 100 - 2 - 2; // max_seq_len=100, 2 special + 2 prefix
    let starts = compute_chunk_starts(text_tokens.len(), max_content, 20);
    let chunks = build_token_chunks(&text_tokens, prefix_tokens, &starts, max_content);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk[0], BOS_TOKEN_ID, "chunk {i} missing BOS");
        assert_eq!(
            *chunk.last().unwrap(),
            EOS_TOKEN_ID,
            "chunk {i} missing EOS"
        );
    }
}

#[test]
fn t_006_each_chunk_contains_prefix() {
    let prefix_tokens = &[100u32, 200, 300];
    let text_tokens: Vec<u32> = (10..210).collect(); // 200 text tokens
    let max_content = 50 - 2 - 3; // 45
    let starts = compute_chunk_starts(text_tokens.len(), max_content, 10);
    let chunks = build_token_chunks(&text_tokens, prefix_tokens, &starts, max_content);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            &chunk[1..1 + prefix_tokens.len()],
            prefix_tokens,
            "chunk {i} prefix mismatch"
        );
    }
}

#[test]
fn t_002_adjacent_chunks_share_overlap_tokens() {
    let prefix_tokens = &[100u32];
    let text_tokens: Vec<u32> = (0..500).collect();
    let max_content = 100;
    let overlap = 20;
    let starts = compute_chunk_starts(text_tokens.len(), max_content, overlap);
    let chunks = build_token_chunks(&text_tokens, prefix_tokens, &starts, max_content);

    let prefix_len = prefix_tokens.len();
    for i in 0..chunks.len() - 1 {
        let current_text = &chunks[i][1 + prefix_len..chunks[i].len() - 1];
        let next_text = &chunks[i + 1][1 + prefix_len..chunks[i + 1].len() - 1];
        // Count shared tokens
        let current_set: std::collections::HashSet<_> = current_text.iter().collect();
        let shared = next_text.iter().filter(|t| current_set.contains(t)).count();
        assert!(
            shared >= overlap,
            "chunks {i} and {} share only {shared} tokens, expected >= {overlap}",
            i + 1
        );
    }
}

#[test]
fn t_012b_empty_text_chunk_has_bos_prefix_eos() {
    let prefix_tokens = &[100u32, 200];
    let text_tokens: Vec<u32> = vec![];
    let starts = compute_chunk_starts(0, 100, 20);
    let chunks = build_token_chunks(&text_tokens, prefix_tokens, &starts, 100);

    assert_eq!(chunks.len(), 1);
    // [BOS, 100, 200, EOS]
    assert_eq!(chunks[0], vec![BOS_TOKEN_ID, 100, 200, EOS_TOKEN_ID]);
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

// --- Chunked Embedding: build_token_chunks with production constants ---
//
// The tests above verify structural correctness with small constants.
// The following tests use MAX_SEQ_LEN (8192) and CHUNK_OVERLAP_TOKENS (2048)
// to match the spec scenarios exactly.

/// Synthetic prefix tokens (10 tokens, representative of "検索文書: ").
const TEST_PREFIX: &[u32] = &[50, 51, 52, 53, 54, 55, 56, 57, 58, 59];

/// max_content for production constants using the shared helper.
const fn prod_max_content() -> usize {
    max_content(TEST_PREFIX.len())
}

/// Build a synthetic text token array of length `n` (values 1000..1000+n).
/// Uses a high base to avoid collision with special/prefix token IDs.
fn synthetic_text(n: usize) -> Vec<u32> {
    (1000..1000 + n as u32).collect()
}

#[test]
fn t_001_10k_tokens_each_chunk_le_max_seq_len() {
    // [T-001] FR-001: 10000 text tokens, max_seq_len=8192 → 2 chunks, each ≤ 8192
    let mc = prod_max_content();
    let text = synthetic_text(10_000);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    assert_eq!(chunks.len(), 2);
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk.len() <= MAX_SEQ_LEN,
            "chunk {i}: len {} > MAX_SEQ_LEN {MAX_SEQ_LEN}",
            chunk.len()
        );
    }
}

#[test]
fn t_002_20k_tokens_overlap_ge_2048() {
    // [T-002] FR-001: 20000 text tokens, overlap=2048 → adjacent chunks share ≥2048 text tokens
    let mc = prod_max_content();
    let text = synthetic_text(20_000);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    assert!(chunks.len() >= 2, "expected multiple chunks");
    let plen = TEST_PREFIX.len();
    for w in chunks.windows(2) {
        let prev_text = &w[0][1 + plen..w[0].len() - 1]; // strip BOS+prefix / EOS
        let next_text = &w[1][1 + plen..w[1].len() - 1];
        // Synthetic tokens are unique sequential integers, so set intersection == overlap count
        let prev_set: std::collections::HashSet<_> = prev_text.iter().collect();
        let shared = next_text.iter().filter(|t| prev_set.contains(t)).count();
        assert!(
            shared >= CHUNK_OVERLAP_TOKENS,
            "adjacent chunks share {shared} tokens, expected >= {CHUNK_OVERLAP_TOKENS}"
        );
    }
}

#[test]
fn t_003_8300_tokens_last_chunk_len_eq_max_seq_len() {
    // [T-003] FR-003: 8300 text tokens → last chunk length = MAX_SEQ_LEN (8192)
    let mc = prod_max_content();
    let text = synthetic_text(8_300);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    let last = chunks.last().expect("non-empty");
    assert_eq!(last.len(), MAX_SEQ_LEN);
}

#[test]
fn t_004_100_tokens_single_chunk_from_build() {
    // [T-004] FR-004: 100 text tokens → single chunk
    let mc = prod_max_content();
    let text = synthetic_text(100);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    assert_eq!(chunks.len(), 1);
    // Verify structure: BOS + prefix + 100 text tokens + EOS
    assert_eq!(chunks[0].len(), 1 + TEST_PREFIX.len() + 100 + 1);
}

#[test]
fn t_005_bos_eos_with_production_constants() {
    // [T-005] FR-002: BOS/EOS verification across multiple text sizes
    let mc = prod_max_content();
    for &n in &[100, 10_000, 20_000] {
        let text = synthetic_text(n);
        let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
        let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk[0], BOS_TOKEN_ID,
                "n={n} chunk {i}: first token {} != BOS",
                chunk[0]
            );
            assert_eq!(
                *chunk.last().unwrap(),
                EOS_TOKEN_ID,
                "n={n} chunk {i}: last token {} != EOS",
                chunk.last().unwrap()
            );
        }
    }
}

#[test]
fn t_006_prefix_in_each_chunk_with_production_constants() {
    // [T-006] FR-002: chunk[1..1+prefix_len] == prefix_tokens for all chunks
    let mc = prod_max_content();
    let text = synthetic_text(10_000);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            &chunk[1..1 + TEST_PREFIX.len()],
            TEST_PREFIX,
            "chunk {i}: prefix mismatch"
        );
    }
}

#[test]
fn t_007_prefix_tokens_match_regardless_of_text_content() {
    // [T-007] FR-008: structural guarantee that build_token_chunks always places
    // prefix_tokens at chunk[1..1+prefix_len], independent of text content.
    // This is the implementation-side contract of FR-008.
    let prefixes: &[&[u32]] = &[&[7, 42, 99], &[256, 1001, 2002, 3003, 4004], &[]];

    for prefix in prefixes {
        let mc = max_content(prefix.len());
        let text = synthetic_text(500);
        let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
        let chunks = build_token_chunks(&text, prefix, &starts, mc);

        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk[0], BOS_TOKEN_ID);
        assert_eq!(&chunk[1..1 + prefix.len()], *prefix);
        assert_eq!(*chunk.last().unwrap(), EOS_TOKEN_ID);
    }
}

#[test]
fn t_012_empty_text_with_production_prefix() {
    // [T-012] FR-001: empty text → single chunk [BOS, prefix..., EOS]
    let mc = prod_max_content();
    let text: Vec<u32> = vec![];
    let starts = compute_chunk_starts(0, mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert_eq!(chunk.len(), 1 + TEST_PREFIX.len() + 1); // BOS + prefix + EOS
    assert_eq!(chunk[0], BOS_TOKEN_ID);
    assert_eq!(&chunk[1..1 + TEST_PREFIX.len()], TEST_PREFIX);
    assert_eq!(*chunk.last().unwrap(), EOS_TOKEN_ID);
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
fn t_013_mc_plus_1_tokens_last_chunk_fills_max_seq_len() {
    // [T-013] FR-003: max_content+1 text tokens → 2 chunks, last chunk = MAX_SEQ_LEN
    let mc = prod_max_content();
    let text = synthetic_text(mc + 1);
    let starts = compute_chunk_starts(text.len(), mc, CHUNK_OVERLAP_TOKENS);
    let chunks = build_token_chunks(&text, TEST_PREFIX, &starts, mc);

    assert_eq!(chunks.len(), 2);
    let last = chunks.last().unwrap();
    assert_eq!(
        last.len(),
        MAX_SEQ_LEN,
        "last chunk len {} != MAX_SEQ_LEN {MAX_SEQ_LEN}",
        last.len()
    );
}

#[test]
fn t_008c_truncate_for_query_noop_at_exact_boundary() {
    // [TC-003] Boundary: len == max_len → no truncation
    let max_len = 50;
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
fn lock_inner_poison_maps_to_inference_error() {
    // Verify that Embedder::lock_inner's map_err closure produces the expected
    // error variant and message when the mutex is poisoned. We can't construct
    // EmbedderInner without model files, so we test the error mapping directly.
    let mutex: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = mutex.lock().unwrap();
        panic!("poison");
    }));
    assert!(mutex.is_poisoned());
    let err = mutex
        .lock()
        .map_err(|_| EmbedError::inference("embedder lock poisoned"))
        .unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg == "embedder lock poisoned"),
        "unexpected error: {err}"
    );
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

#[test]
fn t_014b_mock_chunked_embedder_batch_preserves_count() {
    let embedder = super::MockChunkedEmbedder::new(2);
    let results = embedder.embed_documents_batch(&["a", "b", "c"]).unwrap();
    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.chunks.len(), 2);
    }
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
fn mock_embedder_with_dims_returns_custom_dimension() {
    for dims in [256, 384, 512] {
        let e = super::MockEmbedder::with_dims(dims);
        assert_eq!(e.embed_query("q").unwrap().len(), dims);
        assert_eq!(e.embed_document("d").unwrap().chunks[0].len(), dims);
        assert_eq!(e.embed_text("t", "").unwrap().len(), dims);
    }
}

#[test]
fn mock_chunked_embedder_with_dims_returns_custom_dimension() {
    let e = super::MockChunkedEmbedder::with_dims(2, 256);
    let doc = e.embed_document("d").unwrap();
    assert_eq!(doc.chunks.len(), 2);
    assert_eq!(doc.chunks[0].len(), 256);
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
fn from_probe_error_handler_not_installed_maps_to_backend() {
    let err: EmbedInitError = crate::model_probe::ProbeError::HandlerNotInstalled.into();
    assert!(
        matches!(err, EmbedInitError::Backend(ref msg) if msg.contains("probe handler not installed")),
        "{err}"
    );
}

#[test]
fn from_probe_error_model_load_failed_maps_to_model_corrupt() {
    let err: EmbedInitError = crate::model_probe::ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    assert!(
        matches!(err, EmbedInitError::ModelCorrupt { ref reason } if reason == "bad weights"),
        "{err}"
    );
}

#[test]
fn from_probe_error_subprocess_failed_maps_to_backend() {
    let err: EmbedInitError =
        crate::model_probe::ProbeError::SubprocessFailed("spawn failed".into()).into();
    assert!(
        matches!(err, EmbedInitError::Backend(ref msg) if msg == "spawn failed"),
        "{err}"
    );
}
