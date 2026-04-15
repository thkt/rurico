//! Shared test helpers available to all modules within the crate.

use std::fs;
use std::path::Path;

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
