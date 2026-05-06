//! Shared test helpers available to all modules within the crate.

use std::fs;
use std::path::Path;

use crate::model_init::ModelInitError;
use crate::model_io::{ModelArtifact, artifacts_from_cache};
use crate::model_probe::ProbeError;

/// Valid ModernBERT config JSON with all required fields, for use in tests
/// that need to bypass config parsing errors and reach later verification stages.
///
/// Any change to required [`crate::modernbert::Config`] fields must be reflected here.
#[cfg(any(test, feature = "test-support"))]
pub(crate) const VALID_CONFIG_JSON: &str = r#"{
    "vocab_size": 1000,
    "hidden_size": 768,
    "num_hidden_layers": 2,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "layer_norm_eps": 1e-5,
    "pad_token_id": 0,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_attention": 128,
    "local_rope_theta": 10000.0
}"#;

/// Minimal BPE tokenizer JSON accepted by tokenizers 0.22+.
#[cfg(any(test, feature = "test-support"))]
pub(crate) const MINIMAL_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "model": {"type": "BPE", "vocab": {}, "merges": []},
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "truncation": null,
    "padding": null
}"#;

/// Representative backbone tensor key satisfying the `MODERNBERT_KEY_PREFIXES`
/// any-of check in [`crate::artifacts`].
#[cfg(any(test, feature = "test-support"))]
pub(crate) const FAKE_BACKBONE_KEY: &str = "layers.0.attn.Wo.weight";

/// Write a minimal but structurally valid safetensors file containing the given
/// tensor keys. Each tensor has one `f32` element of weight data.
#[cfg(any(test, feature = "test-support"))]
pub(crate) fn write_fake_safetensors(path: &Path, tensor_keys: &[&str]) {
    let mut header_obj = serde_json::Map::new();
    header_obj.insert("__metadata__".to_owned(), serde_json::json!({}));
    let mut offset = 0usize;
    for &key in tensor_keys {
        let end = offset + 4;
        header_obj.insert(
            key.to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [offset, end]
            }),
        );
        offset = end;
    }
    let header_json = serde_json::to_vec(&header_obj).unwrap();
    let header_len = header_json.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(&header_json);
    for _ in tensor_keys {
        data.extend_from_slice(&0f32.to_le_bytes());
    }
    fs::write(path, data).unwrap();
}

/// Create a fake HuggingFace Hub cache directory structure.
///
/// Builds the `models--{repo_slug}/refs/{revision}` → `snapshots/{hash}/`
/// layout that `hf_hub::Cache` expects, with the given files populated.
pub(crate) fn setup_fake_hf_cache(
    hub_dir: &Path,
    repo_id: &str,
    revision: &str,
    files: &[(&str, &[u8])],
) {
    let repo_slug = repo_id.replace('/', "--");
    let repo_dir = hub_dir.join(format!("models--{repo_slug}"));
    let refs_dir = repo_dir.join("refs");
    fs::create_dir_all(&refs_dir).unwrap();
    let commit_hash = "abc123";
    fs::write(refs_dir.join(revision), commit_hash).unwrap();

    let snapshot_dir = repo_dir.join("snapshots").join(commit_hash);
    fs::create_dir_all(&snapshot_dir).unwrap();
    for &(name, content) in files {
        fs::write(snapshot_dir.join(name), content).unwrap();
    }
}

/// Like [`setup_fake_hf_cache`] but with snapshot files as symlinks to blob
/// files, mirroring the production HF Hub cache layout
/// (`snapshots/<commit>/<file>` -> `../../blobs/<etag>`).
///
/// Used by Issue #107 regression tests to exercise the symlink-preserving
/// invariant: `validate_probe_paths` and `probe_env_to_paths` must NOT
/// substitute the canonicalized blob path for the snapshot path, because
/// MLX `load_safetensors` dispatches on the filename extension.
#[cfg(unix)]
pub(crate) fn setup_fake_hf_cache_with_symlinks(
    hub_dir: &Path,
    repo_id: &str,
    revision: &str,
    files: &[(&str, &[u8])],
) {
    use std::os::unix::fs::symlink;
    let repo_slug = repo_id.replace('/', "--");
    let repo_dir = hub_dir.join(format!("models--{repo_slug}"));
    let refs_dir = repo_dir.join("refs");
    fs::create_dir_all(&refs_dir).unwrap();
    let commit_hash = "abc123";
    fs::write(refs_dir.join(revision), commit_hash).unwrap();

    let blobs_dir = repo_dir.join("blobs");
    fs::create_dir_all(&blobs_dir).unwrap();
    let snapshot_dir = repo_dir.join("snapshots").join(commit_hash);
    fs::create_dir_all(&snapshot_dir).unwrap();
    for &(name, content) in files {
        // Stable etag derived from filename so the blob name is deterministic.
        let etag = format!("etag-{name}");
        let blob = blobs_dir.join(&etag);
        fs::write(&blob, content).unwrap();
        let snapshot_link = snapshot_dir.join(name);
        // Relative symlink target: snapshots/<commit>/X -> ../../blobs/<etag>
        symlink(format!("../../blobs/{etag}"), &snapshot_link).unwrap();
    }
}

// ── Generic lifecycle test helpers (Phase 2 / FR-009 / FR-010) ─────────────

/// Generic helper for `cache_lookup_returns_some_when_all_files_present`.
///
/// Stages a fake HF cache with the three artifact files (placeholder content),
/// invokes [`artifacts_from_cache`](artifacts_from_cache), and
/// asserts that the returned paths end with the standard filenames.
///
/// Callable for any kind whose ID type implements
/// [`ModelArtifact`](crate::model_io::ModelArtifact).
pub(crate) fn assert_cache_lookup_returns_some_when_all_files_present<Id: ModelArtifact>(
    model: Id,
) {
    let dir = tempfile::tempdir().unwrap();
    setup_fake_hf_cache(
        dir.path(),
        model.repo_id(),
        model.revision(),
        &[
            ("model.safetensors", b"fake"),
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
        ],
    );
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, model).unwrap();
    let paths = result.expect("should return Some when all files cached");
    assert!(
        paths.model.ends_with("model.safetensors"),
        "model path: {}",
        paths.model.display()
    );
    assert!(
        paths.config.ends_with("config.json"),
        "config path: {}",
        paths.config.display()
    );
    assert!(
        paths.tokenizer.ends_with("tokenizer.json"),
        "tokenizer path: {}",
        paths.tokenizer.display()
    );
}

/// Generic helper for `cache_lookup_returns_none_when_empty`.
pub(crate) fn assert_cache_lookup_returns_none_when_empty<Id: ModelArtifact>(model: Id) {
    let dir = tempfile::tempdir().unwrap();
    let cache = hf_hub::Cache::new(dir.path().to_path_buf());
    let result = artifacts_from_cache(&cache, model).unwrap();
    assert!(result.is_none());
}

/// Generic helper for `from_probe_error_maps_correctly`.
///
/// Asserts that all four [`ProbeError`](crate::model_probe::ProbeError) variants
/// map to the expected [`ModelInitError`](crate::model_init::ModelInitError)
/// variant. Kind-agnostic because `ModelInitError` is unified across embed and
/// reranker (FR-002).
pub(crate) fn assert_from_probe_error_maps_correctly() {
    let err: ModelInitError = ProbeError::HandlerNotInstalled.into();
    assert!(
        matches!(err, ModelInitError::Backend(ref m) if m.contains("probe handler not installed")),
        "{err}"
    );

    let err: ModelInitError = ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    assert!(
        matches!(err, ModelInitError::ModelCorrupt { ref reason } if reason == "bad weights"),
        "{err}"
    );

    let err: ModelInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
    assert!(
        matches!(err, ModelInitError::Backend(ref m) if m == "spawn failed"),
        "{err}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_fake_hf_cache_creates_expected_structure() {
        let dir = tempfile::tempdir().unwrap();
        setup_fake_hf_cache(
            dir.path(),
            "cl-nagoya/ruri-v3-310m",
            "some-revision-hash",
            &[("model.safetensors", b"model-data"), ("config.json", b"{}")],
        );

        // refs file exists and contains commit hash
        let refs_path = dir
            .path()
            .join("models--cl-nagoya--ruri-v3-310m/refs/some-revision-hash");
        assert!(refs_path.exists());
        let commit = fs::read_to_string(&refs_path).unwrap();
        assert_eq!(commit, "abc123");

        // snapshot files exist with correct content
        let snapshot_dir = dir.path().join(format!(
            "models--cl-nagoya--ruri-v3-310m/snapshots/{commit}"
        ));
        assert_eq!(
            fs::read(snapshot_dir.join("model.safetensors")).unwrap(),
            b"model-data"
        );
        assert_eq!(fs::read(snapshot_dir.join("config.json")).unwrap(), b"{}");
    }

    // T-017: setup_fake_hf_cache_with_symlinks builds the production HF cache layout
    #[cfg(unix)]
    #[test]
    fn t_017_setup_fake_hf_cache_with_symlinks_creates_symlink_structure() {
        let dir = tempfile::tempdir().unwrap();
        setup_fake_hf_cache_with_symlinks(
            dir.path(),
            "org/model",
            "dev",
            &[("model.safetensors", b"weights")],
        );

        let snapshot_link = dir
            .path()
            .join("models--org--model/snapshots/abc123/model.safetensors");
        assert!(
            snapshot_link.is_symlink(),
            "snapshot file must be a symlink"
        );
        let canon = fs::canonicalize(&snapshot_link).unwrap();
        assert!(
            canon.to_string_lossy().contains("blobs/"),
            "canonicalize must resolve into blobs/: {}",
            canon.display()
        );
    }

    #[test]
    fn setup_fake_hf_cache_resolves_via_hf_cache() {
        let dir = tempfile::tempdir().unwrap();
        let repo_id = "org/model";
        let revision = "deadbeef";
        setup_fake_hf_cache(dir.path(), repo_id, revision, &[("weights.bin", b"w")]);

        let cache = hf_hub::Cache::new(dir.path().to_path_buf());
        let repo = cache.repo(hf_hub::Repo::with_revision(
            repo_id.to_owned(),
            hf_hub::RepoType::Model,
            revision.to_owned(),
        ));
        let path = repo.get("weights.bin");
        assert!(path.is_some(), "hf_hub::Cache should resolve the file");
    }
}
