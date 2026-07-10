use super::*;

#[test]
fn read_config_returns_error_for_missing_file() {
    let err = read_config::<serde_json::Value>(Path::new("/nonexistent/config.json")).unwrap_err();
    assert!(matches!(err, ModelIoError::Config { .. }), "{err}");
}

#[test]
fn read_config_returns_error_for_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(&path, b"not json").unwrap();
    let err = read_config::<serde_json::Value>(&path).unwrap_err();
    assert!(
        matches!(err, ModelIoError::Config { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

#[test]
fn load_tokenizer_returns_error_for_missing_file() {
    let err = load_tokenizer(Path::new("/nonexistent/tokenizer.json")).unwrap_err();
    assert!(matches!(err, ModelIoError::Tokenizer(_)), "{err}");
}

#[test]
fn read_config_returns_error_for_missing_fields() {
    use crate::modernbert::Config;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(&path, b"{ \"vocab_size\": 1000 }").unwrap();
    let err = read_config::<Config>(&path).unwrap_err();
    assert!(
        matches!(err, ModelIoError::Config { ref reason, .. } if reason.contains("parse error")),
        "{err}"
    );
}

// ── pad_sequences tests ──────────────────────────────────────────────

#[test]
fn pad_sequences_no_masks_generates_identity() {
    let ids = vec![vec![1, 2, 3], vec![4, 5]];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None, None);
    assert_eq!(batch, 2);
    assert_eq!(max_len, 3);
    assert_eq!(flat_ids, vec![1, 2, 3, 4, 5, 0]);
    assert_eq!(flat_mask, vec![1, 1, 1, 1, 1, 0]);
}

#[test]
fn pad_sequences_with_masks_copies_mask() {
    let ids = vec![vec![10, 20], vec![30]];
    let masks = vec![vec![1, 1], vec![1]];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, Some(&masks), None);
    assert_eq!(batch, 2);
    assert_eq!(max_len, 2);
    assert_eq!(flat_ids, vec![10, 20, 30, 0]);
    assert_eq!(flat_mask, vec![1, 1, 1, 0]);
}

#[test]
fn pad_sequences_empty_input() {
    let ids: Vec<Vec<u32>> = vec![];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None, None);
    assert_eq!(batch, 0);
    assert_eq!(max_len, 0);
    assert!(flat_ids.is_empty());
    assert!(flat_mask.is_empty());
}

#[test]
fn pad_sequences_single_sequence() {
    let ids = vec![vec![1, 2, 3, 4]];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None, None);
    assert_eq!(batch, 1);
    assert_eq!(max_len, 4);
    assert_eq!(flat_ids, vec![1, 2, 3, 4]);
    assert_eq!(flat_mask, vec![1, 1, 1, 1]);
}

#[test]
fn pad_sequences_target_len_extends_when_larger_than_actual_max() {
    let ids = vec![vec![1, 2], vec![3]];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None, Some(5));
    assert_eq!(batch, 2);
    assert_eq!(max_len, 5, "target_len 5 > actual_max 2 should extend to 5");
    assert_eq!(flat_ids, vec![1, 2, 0, 0, 0, 3, 0, 0, 0, 0]);
    assert_eq!(flat_mask, vec![1, 1, 0, 0, 0, 1, 0, 0, 0, 0]);
}

#[test]
fn pad_sequences_target_len_keeps_actual_max_when_smaller() {
    let ids = vec![vec![1, 2, 3, 4], vec![5, 6]];
    let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None, Some(2));
    assert_eq!(batch, 2);
    assert_eq!(
        max_len, 4,
        "target_len 2 < actual_max 4 must not truncate; actual_max wins"
    );
    assert_eq!(flat_ids, vec![1, 2, 3, 4, 5, 6, 0, 0]);
    assert_eq!(flat_mask, vec![1, 1, 1, 1, 1, 1, 0, 0]);
}

#[test]
#[should_panic(expected = "masks length 1 != ids length 2")]
fn pad_sequences_panics_when_masks_shorter_than_ids() {
    let ids = vec![vec![1, 2], vec![3]];
    let masks = vec![vec![1, 1]];
    pad_sequences(&ids, Some(&masks), None);
}

#[test]
#[should_panic(expected = "masks length 3 != ids length 2")]
fn pad_sequences_panics_when_masks_longer_than_ids() {
    let ids = vec![vec![1, 2], vec![3]];
    let masks = vec![vec![1, 1], vec![1], vec![1]];
    pad_sequences(&ids, Some(&masks), None);
}

// ── compute_sub_batch_size tests ────────────────────────────────────────

// Pin the concrete per-bucket sub-batch sizes, not the formula: asserting
// against `TOKEN_BUDGET / BUCKET_BOUNDS[i]` would track both sides and stay
// green even if TOKEN_BUDGET drifted. 256_000 / (128, 512, 2048, 8192) =
// (2000, 500, 125, 31); embed and reranker callers share this function.
#[test]
fn compute_sub_batch_size_matches_formula_per_bucket() {
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[0], None), 2000);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[1], None), 500);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[2], None), 125);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[3], None), 31);
}

// `max(1)` floor: pathological bucket_len > TOKEN_BUDGET cannot happen in
// production (BUCKET_BOUNDS[3]=8192 < 256_000), but the guard keeps callers
// from looping over zero-sized chunks if BUCKET_BOUNDS is later widened.
#[test]
fn compute_sub_batch_size_returns_one_when_bucket_exceeds_token_budget() {
    assert_eq!(compute_sub_batch_size(TOKEN_BUDGET + 1, None), 1);
    assert_eq!(compute_sub_batch_size(usize::MAX, None), 1);
}

// T-001: compute_sub_batch_size は budget_override=None で現行 const 由来の値を返す
#[test]
fn compute_sub_batch_size_with_none_override_matches_const_token_budget() {
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[0], None), 2000);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[1], None), 500);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[2], None), 125);
    assert_eq!(compute_sub_batch_size(BUCKET_BOUNDS[3], None), 31);
}

// T-002: compute_sub_batch_size は budget_override=Some で override し 1 で floor する
#[test]
fn compute_sub_batch_size_with_some_override_takes_priority_over_const_and_floors_at_one() {
    assert_eq!(compute_sub_batch_size(8192, Some(2048)), 1);
}

// ── ModelArtifact::Kind associated type ─────────────────────────────────

// T-009: <ModelId as ModelArtifact>::Kind == EmbedKind
//        and <RerankerModelId as ModelArtifact>::Kind == RerankerKind
#[test]
fn model_artifact_kind_associated_type_is_fixed_per_id() {
    use crate::artifacts::{EmbedKind, RerankerKind};
    use crate::embed::ModelId;
    use crate::reranker::RerankerModelId;

    // Compile-time type-equality assertion: the `where` clause forces the
    // type checker to confirm `<I as ModelArtifact>::Kind` resolves to `K`.
    fn assert_kind<I, K>()
    where
        I: ModelArtifact<Kind = K>,
    {
    }

    assert_kind::<ModelId, EmbedKind>();
    assert_kind::<RerankerModelId, RerankerKind>();

    // Force at least one runtime assertion so the test body is not empty.
    assert_eq!(ModelId::DEFAULT.repo_id(), "cl-nagoya/ruri-v3-310m");
}

// ── download_artifacts_with seam tests ──────────────────────────────────

use crate::embed::ModelId;

// T-106-004: download_artifacts_with
#[test]
fn download_artifacts_with_routes_success_to_caller() {
    let dir = tempfile::tempdir().unwrap();
    let fake = ModelPaths::from_dir(dir.path());
    let expected = fake.model.clone();
    let result = download_artifacts_with(
        ModelId::DEFAULT,
        DOWNLOAD_TIMEOUT,
        move |_repo_id, _revision| Ok(fake),
    )
    .unwrap();
    assert_eq!(result.model, expected);
}

// T-106-005: download_artifacts_with
#[test]
fn download_artifacts_with_routes_error_to_caller() {
    let err = download_artifacts_with(ModelId::DEFAULT, DOWNLOAD_TIMEOUT, |_repo_id, _revision| {
        Err(ModelIoError::Download("simulated".to_owned()))
    })
    .unwrap_err();
    assert!(
        matches!(err, ModelIoError::Download(ref s) if s == "simulated"),
        "{err}"
    );
}

// T-106-006: download_artifacts_with
#[test]
fn download_artifacts_with_returns_timeout_when_worker_exceeds_budget() {
    let err = download_artifacts_with(
        ModelId::DEFAULT,
        Duration::from_millis(50),
        |_repo_id, _revision| {
            thread::sleep(Duration::from_secs(2));
            Err(ModelIoError::Download("unreachable".to_owned()))
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, ModelIoError::Download(ref s) if s.contains("timeout")),
        "expected timeout message, got: {err}"
    );
}
