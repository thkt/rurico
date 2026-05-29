use super::*;
use crate::test_support::{FAKE_BACKBONE_KEY, write_fake_safetensors};

#[test]
fn read_safetensors_keys_returns_expected_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.safetensors");
    write_fake_safetensors(&path, &["foo.weight", "bar.bias"]);
    let keys = read_safetensors_keys(&path).unwrap();
    assert!(keys.contains(&"foo.weight".to_owned()));
    assert!(keys.contains(&"bar.bias".to_owned()));
    assert!(!keys.contains(&"__metadata__".to_owned()));
}

#[test]
fn read_safetensors_keys_errors_on_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.safetensors");
    // 4 bytes — not enough for the 8-byte header length field
    fs::write(&path, b"fake").unwrap();
    let err = read_safetensors_keys(&path).unwrap_err();
    assert!(matches!(err, ArtifactError::WrongModelKind { .. }), "{err}");
}

#[test]
fn verify_as_embed_returns_missing_file_for_absent_model() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("config.json"), b"{}").unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let paths = ModelPaths::from_dir(dir.path()); // model.safetensors does not exist
    let err = verify_as_embed(paths).unwrap_err();
    assert!(
        matches!(err, ArtifactError::MissingFile { ref path } if path.ends_with("model.safetensors")),
        "{err}"
    );
}

#[test]
fn verify_as_embed_returns_invalid_config_for_malformed_config() {
    let dir = tempfile::tempdir().unwrap();
    // model exists (content irrelevant — config check runs first)
    fs::write(dir.path().join("model.safetensors"), b"placeholder").unwrap();
    fs::write(dir.path().join("config.json"), b"{}").unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = verify_as_embed(paths).unwrap_err();
    assert!(matches!(err, ArtifactError::InvalidConfig { .. }), "{err}");
}

#[test]
fn verify_as_embed_returns_invalid_config_for_zero_hidden_size() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"placeholder").unwrap();
    // Parseable JSON but fails Config::validate() due to hidden_size == 0
    fs::write(
        dir.path().join("config.json"),
        br#"{"vocab_size":1000,"hidden_size":0,"num_hidden_layers":2,
                "num_attention_heads":12,"intermediate_size":3072,
                "max_position_embeddings":512,"layer_norm_eps":1e-5,
                "pad_token_id":0,"global_attn_every_n_layers":3,
                "global_rope_theta":160000.0,"local_attention":128,
                "local_rope_theta":10000.0}"#,
    )
    .unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = verify_as_embed(paths).unwrap_err();
    assert!(
        matches!(err, ArtifactError::InvalidConfig { ref reason, .. } if reason.contains("hidden_size")),
        "{err}"
    );
}

#[test]
fn verify_embed_kind_rejects_reranker_keys() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &["classifier.weight", "head.dense.weight", "head.norm.weight"],
    );
    let paths = ModelPaths::from_dir(dir.path());
    let err = verify_embed_kind(&paths).unwrap_err();
    assert!(
        matches!(
            err,
            ArtifactError::WrongModelKind {
                expected: "embed model",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn verify_reranker_kind_rejects_missing_classifier_key() {
    let dir = tempfile::tempdir().unwrap();
    // embed-only keys — no classifier/head
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    let paths = ModelPaths::from_dir(dir.path());
    let err = verify_reranker_kind(&paths).unwrap_err();
    assert!(
        matches!(
            err,
            ArtifactError::WrongModelKind {
                expected: "reranker model",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn verify_embed_kind_accepts_model_without_reranker_keys() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    let paths = ModelPaths::from_dir(dir.path());
    assert!(verify_embed_kind(&paths).is_ok());
}

#[test]
fn verify_reranker_kind_accepts_model_with_all_required_keys() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &[
            FAKE_BACKBONE_KEY,
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight",
        ],
    );
    let paths = ModelPaths::from_dir(dir.path());
    assert!(verify_reranker_kind(&paths).is_ok());
}

// ── From<ModelIoError> unit tests ──────────────────────────────────────

#[test]
fn from_model_io_error_maps_config_variant() {
    let err = ArtifactError::from(ModelIoError::Config {
        path: "/p".into(),
        reason: "bad".into(),
    });
    assert!(
        matches!(err, ArtifactError::InvalidConfig { ref reason, .. } if reason == "bad"),
        "{err}"
    );
}

#[test]
fn from_model_io_error_maps_tokenizer_variant() {
    let err = ArtifactError::from(ModelIoError::Tokenizer("tok err".into()));
    assert!(
        matches!(err, ArtifactError::InvalidTokenizer(ref m) if m == "tok err"),
        "{err}"
    );
}

#[test]
fn from_model_io_error_maps_download_variant() {
    let err = ArtifactError::from(ModelIoError::Download("dl err".into()));
    assert!(
        matches!(err, ArtifactError::DownloadFailed(ref m) if m == "dl err"),
        "{err}"
    );
}

// ── TC-001/TC-002: safetensors error path tests ─────────────────────────

#[test]
fn read_safetensors_keys_errors_on_oversized_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.safetensors");
    let len: u64 = 200 * 1024 * 1024; // 200 MiB — exceeds 100 MiB limit
    let mut data = Vec::new();
    data.extend_from_slice(&len.to_le_bytes());
    fs::write(&path, data).unwrap();
    let err = read_safetensors_keys(&path).unwrap_err();
    assert!(
        matches!(err, ArtifactError::WrongModelKind { ref keys_hint, .. } if keys_hint.contains("100 MiB")),
        "{err}"
    );
}

#[test]
fn read_safetensors_keys_errors_on_invalid_json_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_json.safetensors");
    let header = b"not valid json";
    let len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&len.to_le_bytes());
    data.extend_from_slice(header);
    fs::write(&path, data).unwrap();
    let err = read_safetensors_keys(&path).unwrap_err();
    assert!(
        matches!(err, ArtifactError::WrongModelKind { ref keys_hint, .. } if keys_hint.contains("JSON parse error")),
        "{err}"
    );
}

// ── TC-002 / TC-003: verify_as_embed / verify_as_reranker happy-path ────

use crate::test_support::{MINIMAL_TOKENIZER_JSON, VALID_CONFIG_JSON};

fn write_valid_config(dir: &Path) {
    fs::write(dir.join("config.json"), VALID_CONFIG_JSON.as_bytes()).unwrap();
}

fn write_valid_tokenizer(dir: &Path) {
    fs::write(
        dir.join("tokenizer.json"),
        MINIMAL_TOKENIZER_JSON.as_bytes(),
    )
    .unwrap();
}

#[test]
fn verify_as_embed_succeeds_with_valid_embed_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    assert!(
        verify_as_embed(paths).is_ok(),
        "verify_as_embed should succeed"
    );
}

#[test]
fn verify_as_embed_rejects_unrelated_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    // No "layers." prefix keys — should fail the positive check.
    write_fake_safetensors(&dir.path().join("model.safetensors"), &["foo.weight"]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    assert!(
        verify_as_embed(paths).is_err(),
        "verify_as_embed should reject unrelated safetensors"
    );
}

#[test]
fn verify_as_reranker_succeeds_with_valid_reranker_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    // Realistic key set: backbone + classifier/head (both required).
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &[
            FAKE_BACKBONE_KEY,
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight",
        ],
    );
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    assert!(
        verify_as_reranker(paths).is_ok(),
        "verify_as_reranker should succeed"
    );
}

#[test]
fn verify_as_reranker_rejects_classifier_only_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    // Classifier/head keys present but backbone absent — should fail positive backbone check.
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &["classifier.weight", "head.dense.weight", "head.norm.weight"],
    );
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    assert!(
        verify_as_reranker(paths).is_err(),
        "verify_as_reranker should reject safetensors without backbone keys"
    );
}

/// Real-world checkpoint: `cl-nagoya/ruri-v3-reranker-310m` ships HF
/// transformers-style backbone keys (`model.layers.*`) alongside flat
/// classifier/head keys. The any-of backbone-prefix check must accept it.
#[test]
fn verify_as_reranker_succeeds_with_hf_wrapped_backbone() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &[
            "model.layers.0.attn.Wo.weight",
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight",
        ],
    );
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    assert!(
        verify_as_reranker(paths).is_ok(),
        "verify_as_reranker should succeed for HF-wrapped backbone keys"
    );
}

// ── delete_files tests ───────────────────────────────────────────────

#[test]
fn delete_files_removes_all_three_artifact_files() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    let artifacts = verify_as_embed(paths).unwrap();

    artifacts.delete_files().unwrap();

    assert!(!dir.path().join("model.safetensors").exists());
    assert!(!dir.path().join("config.json").exists());
    assert!(!dir.path().join("tokenizer.json").exists());
}

#[test]
fn delete_files_returns_error_when_file_already_removed() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    let artifacts = verify_as_embed(paths).unwrap();

    // Pre-remove model so delete_files hits a missing-file error
    fs::remove_file(dir.path().join("model.safetensors")).unwrap();

    assert!(artifacts.delete_files().is_err());
}

// T-105-011: delete_files_continues_after_first_failure
//
// The continues-on-error contract documented on `delete_files`: when the
// first file is already absent, the remaining files MUST still be
// attempted (and removed) so a re-download from a fresh cache cannot be
// blocked by a half-cleaned directory.
#[test]
fn delete_files_continues_after_first_failure() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());
    let paths = ModelPaths::from_dir(dir.path());
    let artifacts = verify_as_embed(paths).unwrap();

    // Pre-remove model so delete_files hits an error on the first file
    // and must continue with the remaining two.
    fs::remove_file(dir.path().join("model.safetensors")).unwrap();

    let err = artifacts
        .delete_files()
        .expect_err("first removal must error");
    assert_eq!(err.kind(), io::ErrorKind::NotFound);

    assert!(
        !dir.path().join("config.json").exists(),
        "config.json must be removed despite earlier failure"
    );
    assert!(
        !dir.path().join("tokenizer.json").exists(),
        "tokenizer.json must be removed despite earlier failure"
    );
}

// ── CandidateArtifacts<K> verify dispatch ───────────────────────────────

use crate::artifacts::CandidateArtifacts;

// T-001: CandidateArtifacts<EmbedKind>::verify happy path
#[test]
fn candidate_verify_embed_kind_succeeds_with_valid_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());

    let candidate: CandidateArtifacts<EmbedKind> = CandidateArtifacts::from_dir(dir.path());
    let result: Result<VerifiedArtifacts<EmbedKind>, ArtifactError> = candidate.verify();
    assert!(
        result.is_ok(),
        "CandidateArtifacts::<EmbedKind>::verify should succeed for backbone-only safetensors + valid config + valid tokenizer; got: {:?}",
        result.err()
    );
}

// T-002: CandidateArtifacts<RerankerKind>::verify happy path
#[test]
fn candidate_verify_reranker_kind_succeeds_with_valid_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &[
            FAKE_BACKBONE_KEY,
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight",
        ],
    );
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());

    let candidate: CandidateArtifacts<RerankerKind> = CandidateArtifacts::from_dir(dir.path());
    let result: Result<VerifiedArtifacts<RerankerKind>, ArtifactError> = candidate.verify();
    assert!(
        result.is_ok(),
        "CandidateArtifacts::<RerankerKind>::verify should succeed for backbone+classifier+head safetensors + valid config + valid tokenizer; got: {:?}",
        result.err()
    );
}

// T-003: CandidateArtifacts<EmbedKind>::verify rejects reranker weights
//        → ArtifactError::WrongModelKind { expected: "embed model", .. }
#[test]
fn candidate_verify_embed_kind_rejects_reranker_weights() {
    let dir = tempfile::tempdir().unwrap();
    // Reranker classifier/head keys without backbone — a reranker-shaped
    // weights file passed to the embed-kind verifier.
    write_fake_safetensors(
        &dir.path().join("model.safetensors"),
        &["classifier.weight", "head.dense.weight", "head.norm.weight"],
    );
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());

    let candidate: CandidateArtifacts<EmbedKind> = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(
            err,
            ArtifactError::WrongModelKind {
                expected: "embed model",
                ..
            }
        ),
        "expected WrongModelKind {{ expected: \"embed model\", .. }}, got: {err}"
    );
}

// T-004: CandidateArtifacts<RerankerKind>::verify rejects backbone-only weights
//        → ArtifactError::WrongModelKind { expected: "reranker model", .. }
#[test]
fn candidate_verify_reranker_kind_rejects_backbone_only_weights() {
    let dir = tempfile::tempdir().unwrap();
    // Backbone-only — an embed-shaped weights file passed to the
    // reranker-kind verifier.
    write_fake_safetensors(&dir.path().join("model.safetensors"), &[FAKE_BACKBONE_KEY]);
    write_valid_config(dir.path());
    write_valid_tokenizer(dir.path());

    let candidate: CandidateArtifacts<RerankerKind> = CandidateArtifacts::from_dir(dir.path());
    let err = candidate.verify().unwrap_err();
    assert!(
        matches!(
            err,
            ArtifactError::WrongModelKind {
                expected: "reranker model",
                ..
            }
        ),
        "expected WrongModelKind {{ expected: \"reranker model\", .. }}, got: {err}"
    );
}
