use std::path::Path;

use super::mlx::unpack_batch_output;
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
fn sort_indices_by_len_orders_by_length() {
    let texts = &["long text here", "hi", "medium"];
    let indices = sort_indices_by_len(texts);
    assert_eq!(indices, vec![1, 2, 0]);
}

#[test]
fn sort_indices_by_len_multibyte() {
    // byte lengths: "ab"=2, "あ"=3, "abcde"=5
    let texts = &["ab", "あ", "abcde"];
    let indices = sort_indices_by_len(texts);
    assert_eq!(indices, vec![0, 1, 2]);
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

#[test]
#[ignore] // requires model download
fn embed_query_returns_768_dims() {
    let paths = download_model().expect("download model");
    let embedder = Embedder::new(&paths).expect("load model");
    let embedding = embedder.embed_query("authentication logic").unwrap();
    assert_eq!(embedding.len(), 768);
}

#[test]
#[ignore] // requires model download
fn embed_documents_batch_returns_correct_count() {
    let paths = download_model().expect("download model");
    let embedder = Embedder::new(&paths).expect("load model");
    let texts = vec![
        "function useAuth() { return user; }",
        "function Button() { return <div/>; }",
    ];
    let embeddings = embedder.embed_documents_batch(&texts).unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 768);
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
        matches!(err, EmbedError::DimensionMismatch { expected: 6, actual: 10 }),
        "{err}"
    );
}

#[test]
fn unpack_batch_output_rejects_zero_total() {
    let flat = vec![0.0f32; 10];
    let err = unpack_batch_output(&flat, &[], 0, &[]).unwrap_err();
    assert!(
        matches!(err, EmbedError::DimensionMismatch { expected: 0, actual: 10 }),
        "{err}"
    );
}

#[test]
fn unpack_batch_output_happy_path() {
    let hidden = EMBEDDING_DIMS as usize;
    let batch_size = 2;
    let max_seq_len = 1;
    let mut flat = vec![0.0f32; batch_size * max_seq_len * hidden];
    // Give each item a distinct nonzero value so postprocess produces different unit vectors
    flat[0] = 1.0;
    flat[hidden] = 2.0;
    // sorted_indices: [1, 0] means sorted_pos 0 → orig 1, sorted_pos 1 → orig 0
    let sorted = vec![1usize, 0];
    let mask = vec![1u32; batch_size * max_seq_len];
    let results = unpack_batch_output(&flat, &sorted, max_seq_len, &mask).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), hidden);
    assert_eq!(results[1].len(), hidden);
    // orig 0 came from sorted_pos 1 (flat[hidden]=2.0), orig 1 from sorted_pos 0 (flat[0]=1.0)
    // Both are L2 normalized, so first nonzero element should be 1.0 (single nonzero in 768-dim)
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

#[test]
#[ignore] // requires model download + MLX
fn probe_returns_available_with_real_model() {
    let paths = download_model().expect("download model");
    let status = probe_via_fork(&paths).expect("probe should succeed");
    assert_eq!(status, ProbeStatus::Available);
}
