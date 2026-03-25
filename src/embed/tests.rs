use std::path::Path;

use super::*;

fn reorder_by_indices<T>(sorted: Vec<T>, sorted_indices: &[usize]) -> Vec<T> {
    let mut results: Vec<Option<T>> = (0..sorted_indices.len()).map(|_| None).collect();
    for (sorted_pos, item) in sorted.into_iter().enumerate() {
        results[sorted_indices[sorted_pos]] = Some(item);
    }
    results
        .into_iter()
        .map(|o| o.expect("all positions filled"))
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
    assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
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
    assert!(
        (result[0] - expected_0).abs() < 1e-6,
        "got {}, expected {}",
        result[0],
        expected_0
    );
    assert!(
        (result[1] - expected_1).abs() < 1e-6,
        "got {}, expected {}",
        result[1],
        expected_1
    );
}

#[test]
fn postprocess_embedding_zero_seq_len() {
    let result = postprocess_embedding(&[], 0, &[]);
    let err = result.unwrap_err();
    assert!(
        matches!(err, EmbedError::DimensionMismatch { expected: _, actual: 0 }),
        "expected DimensionMismatch with actual=0, got: {err}"
    );
}

#[test]
fn postprocess_embedding_wrong_dims() {
    let data = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1u32];
    let result = postprocess_embedding(&data, 1, &mask);
    let err = result.unwrap_err();
    assert!(
        matches!(err, EmbedError::DimensionMismatch { expected: 768, actual: 3 }),
        "expected DimensionMismatch{{768, 3}}, got: {err}"
    );
}

#[test]
fn read_config_missing_file() {
    let err = read_config::<serde_json::Value>(Path::new("/nonexistent/config.json")).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("No such file")),
        "expected Inference with fs error, got: {err}"
    );
}

#[test]
fn read_config_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, b"not valid json {{{").unwrap();
    let err = read_config::<serde_json::Value>(&path).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("parse error")),
        "expected Inference with parse error, got: {err}"
    );
}

#[test]
fn read_config_missing_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, b"{ \"vocab_size\": 1000 }").unwrap();
    let err = read_config::<crate::modernbert::Config>(&path).unwrap_err();
    assert!(
        matches!(err, EmbedError::Inference(ref msg) if msg.contains("parse error")),
        "expected Inference with missing field error, got: {err}"
    );
}

#[test]
fn validate_partial_download_reports_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = paths.validate().unwrap_err();
    match &err {
        EmbedError::ModelNotFound { path } => {
            assert!(
                path.ends_with("config.json"),
                "should report config.json as missing, got: {path:?}"
            );
        }
        other => panic!("expected ModelNotFound, got: {other}"),
    }
}

#[test]
fn embedder_new_model_not_found() {
    let paths = ModelPaths::from_dir(Path::new("/nonexistent/path"));
    let err = Embedder::new(&paths).unwrap_err();
    assert!(
        matches!(err, EmbedError::ModelNotFound { .. }),
        "expected ModelNotFound, got: {err}"
    );
}

#[test]
fn mean_pooling_short_mask_truncates_safely() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 tokens, hidden_size=2
    let mask = vec![1u32]; // 1 mask element for 2 tokens
    let result = mean_pooling(&data, 2, 2, &mask);
    assert_eq!(result, vec![1.0, 2.0], "short mask should use available entries only");
}

#[test]
fn sort_indices_by_char_count_orders_by_length() {
    let texts = &["long text here", "hi", "medium"];
    let indices = sort_indices_by_char_count(texts);
    assert_eq!(indices, vec![1, 2, 0]);
}

#[test]
fn sort_indices_by_char_count_single_item() {
    let texts = &["only"];
    let indices = sort_indices_by_char_count(texts);
    assert_eq!(indices, vec![0]);
}

#[test]
fn sort_indices_by_char_count_same_length() {
    let texts = &["abc", "def", "ghi"];
    let indices = sort_indices_by_char_count(texts);
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn sort_indices_by_char_count_multibyte() {
    let texts = &["ab", "あ", "abcde"];
    let indices = sort_indices_by_char_count(texts);
    assert_eq!(indices, vec![1, 0, 2]);
}

#[test]
fn reorder_by_indices_round_trip() {
    let texts = &["long text here", "hi", "medium"];
    let sorted_indices = sort_indices_by_char_count(texts);

    let sorted_results: Vec<&str> = sorted_indices.iter().map(|&i| texts[i]).collect();
    assert_eq!(sorted_results, vec!["hi", "medium", "long text here"]);

    let reordered = reorder_by_indices(sorted_results, &sorted_indices);
    assert_eq!(reordered, vec!["long text here", "hi", "medium"]);
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
    assert_eq!(result.len(), hidden_size, "output should be 768-dim");

    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "output should be L2-normalized, got norm={norm}"
    );
    let ratio = result[0] / result[1];
    let expected_ratio = 2.0 / 3.0;
    assert!(
        (ratio - expected_ratio).abs() < 1e-5,
        "expected ratio {expected_ratio}, got {ratio}"
    );
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
    let texts = vec!["function useAuth() { return user; }", "function Button() { return <div/>; }"];
    let embeddings = embedder.embed_documents_batch(&texts).unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 768);
}
