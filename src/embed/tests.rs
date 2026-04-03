use std::path::Path;

use super::mlx::unpack_batch_output;
use super::pooling::{l2_normalize, mean_pooling};
use super::probe::{PROBE_ACK, interpret_probe_output, probe_env_to_paths};
use super::*;

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
    assert!(
        matches!(err, EmbedError::DimensionMismatch { actual: 0, .. }),
        "{err}"
    );
}

#[test]
fn postprocess_embedding_wrong_dims() {
    let data = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1u32];
    let result = postprocess_embedding(&data, 1, &mask);
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::DimensionMismatch { expected, actual: 3 }
            if expected == EMBEDDING_DIMS as usize
        ),
        "{err}"
    );
}

#[test]
fn read_config_missing_file() {
    let err = read_config::<serde_json::Value>(Path::new("/nonexistent/config.json")).unwrap_err();
    assert!(
        matches!(err, EmbedError::Config { ref reason, .. } if reason.contains("No such file")),
        "{err}"
    );
}

#[test]
fn read_config_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, b"not valid json {{{").unwrap();
    let err = read_config::<serde_json::Value>(&path).unwrap_err();
    assert!(
        matches!(err, EmbedError::Config { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

#[test]
fn read_config_missing_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, b"{ \"vocab_size\": 1000 }").unwrap();
    let err = read_config::<crate::modernbert::Config>(&path).unwrap_err();
    assert!(
        matches!(err, EmbedError::Config { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

#[test]
fn validate_partial_download_reports_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = paths.validate().unwrap_err();
    let EmbedError::ModelNotFound { path } = &err else {
        panic!("{err}");
    };
    assert!(path.ends_with("config.json"), "{path:?}");
}

#[test]
fn embedder_new_model_not_found() {
    let paths = ModelPaths::from_dir(Path::new("/nonexistent/path"));
    let err = Embedder::new(&paths).unwrap_err();
    assert!(matches!(err, EmbedError::ModelNotFound { .. }), "{err}");
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
    let hidden_size = EMBEDDING_DIMS as usize;
    let seq_len = 2;
    let data = vec![0.0f32; seq_len * hidden_size + 1];
    let mask = vec![1u32; seq_len];
    let err = postprocess_embedding(&data, seq_len, &mask).unwrap_err();
    assert!(matches!(err, EmbedError::DimensionMismatch { .. }), "{err}");
}

#[test]
fn postprocess_embedding_happy_path() {
    let hidden_size = EMBEDDING_DIMS as usize;
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

fn setup_fake_cache(hub_dir: &std::path::Path) {
    let repo_dir = hub_dir.join("models--cl-nagoya--ruri-v3-310m");
    let refs_dir = repo_dir.join("refs");
    std::fs::create_dir_all(&refs_dir).unwrap();
    let commit_hash = "abc123";
    std::fs::write(refs_dir.join(MODEL_REVISION), commit_hash).unwrap();

    let snapshot_dir = repo_dir.join("snapshots").join(commit_hash);
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    std::fs::write(snapshot_dir.join("model.safetensors"), b"fake").unwrap();
    std::fs::write(snapshot_dir.join("config.json"), b"{}").unwrap();
    std::fs::write(snapshot_dir.join("tokenizer.json"), b"{}").unwrap();
}

#[test]
fn model_paths_from_cache_returns_none_when_empty() {
    let dir = tempfile::tempdir().unwrap();
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = model_paths_from_cache(&cache).unwrap();
    assert!(result.is_none());
}

#[test]
fn model_paths_from_cache_returns_some_when_all_files_present() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache(dir.path());
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = model_paths_from_cache(&cache).unwrap();
    let paths = result.expect("should return Some when all files cached");
    assert!(paths.model.ends_with("model.safetensors"));
    assert!(paths.config.ends_with("config.json"));
    assert!(paths.tokenizer.ends_with("tokenizer.json"));
}

#[test]
fn model_paths_from_cache_returns_none_when_partial() {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_cache(dir.path());
    let snapshot_dir = dir
        .path()
        .join("models--cl-nagoya--ruri-v3-310m/snapshots/abc123");
    std::fs::remove_file(snapshot_dir.join("config.json")).unwrap();
    std::fs::remove_file(snapshot_dir.join("tokenizer.json")).unwrap();

    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = model_paths_from_cache(&cache).unwrap();
    assert!(result.is_none());
}

#[test]
fn probe_rejects_missing_paths() {
    let paths = ModelPaths::from_dir(Path::new("/nonexistent/path"));
    let err = Embedder::probe(&paths).unwrap_err();
    assert!(matches!(err, EmbedError::ModelNotFound { .. }), "{err}");
}

#[test]
fn probe_rejects_invalid_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("config.json"), b"not json").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = Embedder::probe(&paths).unwrap_err();
    assert!(
        matches!(err, EmbedError::Config { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

#[test]
fn probe_rejects_invalid_config_values() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        br#"{
        "vocab_size": 0, "hidden_size": 768, "num_hidden_layers": 2,
        "num_attention_heads": 12, "intermediate_size": 3072,
        "max_position_embeddings": 512, "layer_norm_eps": 1e-5,
        "pad_token_id": 0, "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0, "local_attention": 128,
        "local_rope_theta": 10000.0
    }"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = Embedder::probe(&paths).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("vocab_size")),
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
    let paths = probe_env_to_paths(Some("/m".into()), Some("/c".into()), Some("/t".into()))
        .unwrap()
        .unwrap();
    assert_eq!(paths.model, PathBuf::from("/m"));
    assert_eq!(paths.config, PathBuf::from("/c"));
    assert_eq!(paths.tokenizer, PathBuf::from("/t"));
}

#[test]
fn unpack_batch_output_rejects_indivisible_shape() {
    let flat = vec![0.0f32; 10];
    let sorted = vec![0usize, 1];
    let mask = vec![1u32; 6];
    let err = unpack_batch_output(&flat, &sorted, 3, &mask).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::DimensionMismatch {
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
    let err = unpack_batch_output(&flat, &[], 0, &[]).unwrap_err();
    assert!(
        matches!(
            err,
            EmbedError::DimensionMismatch {
                expected: 0,
                actual: 10
            }
        ),
        "{err}"
    );
}

#[test]
fn unpack_batch_output_happy_path() {
    let hidden = EMBEDDING_DIMS as usize;
    let batch_size = 2;
    let max_seq_len = 1;
    let mut flat = vec![0.0f32; batch_size * max_seq_len * hidden];
    flat[0] = 1.0;
    flat[hidden] = 2.0;
    let sorted = vec![1usize, 0];
    let mask = vec![1u32; batch_size * max_seq_len];
    let results = unpack_batch_output(&flat, &sorted, max_seq_len, &mask).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), hidden);
    assert_eq!(results[1].len(), hidden);
    assert!((results[0][0] - 1.0).abs() < 1e-6);
    assert!((results[1][0] - 1.0).abs() < 1e-6);
}

fn exit_status(code: i32) -> std::process::ExitStatus {
    std::process::Command::new("sh")
        .args(["-c", &format!("exit {code}")])
        .status()
        .unwrap()
}

#[test]
fn interpret_probe_output_available_on_exit_0() {
    let output = std::process::Output {
        status: exit_status(0),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    assert_eq!(
        interpret_probe_output(&output).unwrap(),
        ProbeStatus::Available
    );
}

#[test]
fn interpret_probe_output_model_corrupt_on_nonzero_exit() {
    let output = std::process::Output {
        status: exit_status(1),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"inference error: bad model".to_vec(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, EmbedError::ModelCorrupt { ref reason } if reason.contains("bad model")),
        "{err}"
    );
}

#[test]
fn interpret_probe_output_model_corrupt_empty_stderr() {
    let output = std::process::Output {
        status: exit_status(1),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, EmbedError::ModelCorrupt { ref reason } if reason == "model load failed"),
        "{err}"
    );
}

#[test]
fn interpret_probe_output_missing_ack() {
    let output = std::process::Output {
        status: exit_status(0),
        stdout: b"unexpected output".to_vec(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("handler not installed")),
        "{err}"
    );
}

#[test]
fn interpret_probe_output_backend_unavailable_on_signal() {
    let status = std::process::Command::new("sh")
        .args(["-c", "kill -ABRT $$"])
        .status()
        .unwrap();
    let output = std::process::Output {
        status,
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    assert_eq!(
        interpret_probe_output(&output).unwrap(),
        ProbeStatus::BackendUnavailable
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
    let paths = download_model().expect("download model");
    let tokenizer = load_tokenizer(&paths.tokenizer).unwrap();
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
fn t_014_mock_chunked_embedder_returns_multi_chunk() {
    let embedder = super::MockChunkedEmbedder::new(3);
    let result = embedder.embed_document("some text").unwrap();
    assert_eq!(result.chunks.len(), 3);
    assert_eq!(result.chunks[0].len(), EMBEDDING_DIMS as usize);
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
    let paths = download_model().expect("download model");
    let tokenizer = load_tokenizer(&paths.tokenizer).unwrap();
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
    let paths = download_model().expect("download model");
    let tokenizer = load_tokenizer(&paths.tokenizer).unwrap();
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
