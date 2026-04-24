//! Typed artifact verification for ruri-v3 model files.
//!
//! Provides a two-phase pipeline: per-module [`CandidateArtifacts`] (unverified)
//! → [`VerifiedArtifacts<K>`] (verified, kind-checked).
//!
//! Consumers use the domain aliases:
//! - [`embed::Artifacts`](crate::embed::Artifacts) = `VerifiedArtifacts<EmbedKind>`
//! - [`reranker::Artifacts`](crate::reranker::Artifacts) = `VerifiedArtifacts<RerankerKind>`

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::model_io::{ModelIoError, ModelPaths, load_tokenizer, read_config};
use crate::modernbert::Config;

// ── Kind markers ────────────────────────────────────────────────────────────

/// Kind marker for embedding models (ruri-v3 embed variants).
#[derive(Debug)]
pub struct EmbedKind(());

/// Kind marker for reranker models (ruri-v3-reranker variants).
#[derive(Debug)]
pub struct RerankerKind(());

// ── VerifiedArtifacts ────────────────────────────────────────────────────────

/// Verified model artifacts, bound to a specific model kind `K`.
///
/// Constructed via [`embed::CandidateArtifacts::verify`](crate::embed::CandidateArtifacts::verify)
/// or [`reranker::CandidateArtifacts::verify`](crate::reranker::CandidateArtifacts::verify).
///
/// Guarantees:
/// - All three artifact files exist on disk.
/// - `config.json` parses as a valid [`crate::modernbert::Config`].
/// - `tokenizer.json` loads without error.
/// - The safetensors weights contain the expected keys for model kind `K`.
pub struct VerifiedArtifacts<K> {
    pub(crate) paths: ModelPaths,
    pub(crate) config: Config,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    _kind: PhantomData<K>,
}

impl<K> fmt::Debug for VerifiedArtifacts<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedArtifacts")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl<K> VerifiedArtifacts<K> {
    fn new(paths: ModelPaths, config: Config, tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            paths,
            config,
            tokenizer,
            _kind: PhantomData,
        }
    }

    /// Borrow the underlying model paths.
    ///
    /// Scoped to cross-target probe bins (`src/bin/gpu_pool_probe.rs`) that
    /// need to load raw files without going through the full `Embedder`.
    /// Hidden from public docs so it does not become part of the stable
    /// semantic-search surface. Phase 3c removes the Phase 3a probe bin and
    /// with it the only external caller.
    #[doc(hidden)]
    pub fn paths(&self) -> &ModelPaths {
        &self.paths
    }

    /// Borrow the parsed model config. See [`paths`](Self::paths) for scope.
    #[doc(hidden)]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Borrow the loaded tokenizer. See [`paths`](Self::paths) for scope.
    #[doc(hidden)]
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Delete the artifact files from disk.
    ///
    /// Consumes `self` so the artifacts cannot be used after deletion.
    /// Useful for cleaning up corrupt or incompatible files so that a
    /// subsequent [`download_model`](crate::embed::download_model) call
    /// will re-fetch them from the network.
    ///
    /// # Errors
    ///
    /// Returns the first [`std::io::Error`] encountered. Remaining files are
    /// still attempted even if an earlier deletion fails.
    pub fn delete_files(self) -> Result<(), io::Error> {
        let mut first_err: Option<io::Error> = None;
        for path in [&self.paths.model, &self.paths.config, &self.paths.tokenizer] {
            if let Err(e) = fs::remove_file(path)
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// ── ArtifactError ────────────────────────────────────────────────────────────

/// Errors from artifact verification or model download.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// A required model file is missing from disk.
    #[error("missing file: {}", path.display())]
    MissingFile {
        /// Path that was not found.
        path: PathBuf,
    },
    /// `config.json` cannot be read or parsed as a valid model config.
    #[error("invalid config at {}: {reason}", path.display())]
    InvalidConfig {
        /// Config file path.
        path: PathBuf,
        /// Parse or IO failure detail.
        reason: String,
    },
    /// `tokenizer.json` cannot be loaded by the tokenizers library.
    #[error("invalid tokenizer: {0}")]
    InvalidTokenizer(String),
    /// Model weights are for a different model kind than expected.
    ///
    /// Detected by inspecting the safetensors file header (tensor key names).
    #[error("wrong model kind: expected {expected}; {keys_hint}")]
    WrongModelKind {
        /// Expected model kind description (e.g. `"embed model"`).
        expected: &'static str,
        /// Diagnostic hint about what was found or missing.
        keys_hint: String,
    },
    /// Model download from Hugging Face Hub failed.
    #[error("download failed: {0}")]
    DownloadFailed(String),
}

// ── Conversion helpers ───────────────────────────────────────────────────────

impl From<ModelIoError> for ArtifactError {
    fn from(e: ModelIoError) -> Self {
        match e {
            ModelIoError::Config { path, reason } => ArtifactError::InvalidConfig { path, reason },
            ModelIoError::Tokenizer(msg) => ArtifactError::InvalidTokenizer(msg),
            ModelIoError::Download(msg) => ArtifactError::DownloadFailed(msg),
        }
    }
}

// ── pub(crate) entry points ──────────────────────────────────────────────────

/// Verify `paths` as an embedding model artifact.
///
/// Runs all verification steps in order: file existence → config parse →
/// tokenizer load → embed kind check (classifier/head keys must be absent).
pub(crate) fn verify_as_embed(
    paths: ModelPaths,
) -> Result<VerifiedArtifacts<EmbedKind>, ArtifactError> {
    verify_files_exist(&paths)?;
    let config = verify_config(&paths)?;
    let tokenizer = verify_tokenizer(&paths)?;
    verify_embed_kind(&paths)?;
    Ok(VerifiedArtifacts::new(paths, config, tokenizer))
}

/// Verify `paths` as a reranker model artifact.
///
/// Runs all verification steps in order: file existence → config parse →
/// tokenizer load → reranker kind check (classifier/head keys must be present).
pub(crate) fn verify_as_reranker(
    paths: ModelPaths,
) -> Result<VerifiedArtifacts<RerankerKind>, ArtifactError> {
    verify_files_exist(&paths)?;
    let config = verify_config(&paths)?;
    let tokenizer = verify_tokenizer(&paths)?;
    verify_reranker_kind(&paths)?;
    Ok(VerifiedArtifacts::new(paths, config, tokenizer))
}

// ── Verification steps ───────────────────────────────────────────────────────

fn verify_files_exist(paths: &ModelPaths) -> Result<(), ArtifactError> {
    for path in [&paths.model, &paths.config, &paths.tokenizer] {
        if !path.exists() {
            return Err(ArtifactError::MissingFile { path: path.clone() });
        }
    }
    Ok(())
}

fn verify_config(paths: &ModelPaths) -> Result<Config, ArtifactError> {
    let config = read_config::<Config>(&paths.config)?;
    config
        .validate()
        .map_err(|reason| ArtifactError::InvalidConfig {
            path: paths.config.clone(),
            reason,
        })?;
    Ok(config)
}

fn verify_tokenizer(paths: &ModelPaths) -> Result<tokenizers::Tokenizer, ArtifactError> {
    Ok(load_tokenizer(&paths.tokenizer)?)
}

// ── Kind check (safetensors header inspection) ───────────────────────────────

/// Tensor key prefixes that must be present in all ModernBERT-based models.
///
/// The ruri-v3 MLX weight files use a flat naming convention without the `model.`
/// wrapper common in HuggingFace transformers checkpoints. The backbone layers are
/// always present and keyed under the `"layers."` namespace.
const MODERNBERT_KEY_PREFIXES: &[&str] = &["layers."];

/// Tensor key prefixes that are present in reranker models and absent in embed models.
const RERANKER_KEY_PREFIXES: &[&str] = &["classifier.", "head.dense.", "head.norm."];

fn verify_embed_kind(paths: &ModelPaths) -> Result<(), ArtifactError> {
    verify_model_kind(paths, "embed model", false)
}

fn verify_reranker_kind(paths: &ModelPaths) -> Result<(), ArtifactError> {
    verify_model_kind(paths, "reranker model", true)
}

/// Check safetensors key prefixes for a ModernBERT-based model.
///
/// Always requires backbone keys (`MODERNBERT_KEY_PREFIXES`).
/// When `require_reranker_keys` is `true`, classifier/head keys must also be present.
/// When `false`, they must be absent.
fn verify_model_kind(
    paths: &ModelPaths,
    expected: &'static str,
    require_reranker_keys: bool,
) -> Result<(), ArtifactError> {
    let mut backbone_seen = [false; MODERNBERT_KEY_PREFIXES.len()];
    let mut reranker_seen = [false; RERANKER_KEY_PREFIXES.len()];

    scan_safetensors_keys(&paths.model, |k| {
        for (i, prefix) in MODERNBERT_KEY_PREFIXES.iter().enumerate() {
            if !backbone_seen[i] && k.starts_with(prefix) {
                backbone_seen[i] = true;
            }
        }
        for (i, prefix) in RERANKER_KEY_PREFIXES.iter().enumerate() {
            if !reranker_seen[i] && k.starts_with(prefix) {
                reranker_seen[i] = true;
            }
        }
    })?;

    for (i, prefix) in MODERNBERT_KEY_PREFIXES.iter().enumerate() {
        if !backbone_seen[i] {
            return Err(ArtifactError::WrongModelKind {
                expected,
                keys_hint: format!("missing required key with prefix '{prefix}'"),
            });
        }
    }
    for (i, prefix) in RERANKER_KEY_PREFIXES.iter().enumerate() {
        let found = reranker_seen[i];
        if require_reranker_keys && !found {
            return Err(ArtifactError::WrongModelKind {
                expected,
                keys_hint: format!("missing required key with prefix '{prefix}'"),
            });
        } else if !require_reranker_keys && found {
            return Err(ArtifactError::WrongModelKind {
                expected,
                keys_hint: format!("found reranker key with prefix '{prefix}'"),
            });
        }
    }
    Ok(())
}

/// Scan tensor key names from a safetensors file header, calling `f` for each non-metadata key.
///
/// The safetensors format starts with an 8-byte u64 LE value indicating the
/// header JSON length, followed by the JSON object mapping tensor names to
/// metadata. Only the header is read; weight data is not accessed.
fn scan_safetensors_keys(path: &Path, mut f: impl FnMut(&str)) -> Result<(), ArtifactError> {
    let mut file = File::open(path).map_err(|e| ArtifactError::WrongModelKind {
        expected: "valid safetensors file",
        keys_hint: format!("cannot open model file: {e}"),
    })?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| ArtifactError::WrongModelKind {
            expected: "valid safetensors file",
            keys_hint: format!("cannot read header length: {e}"),
        })?;
    let raw_len = u64::from_le_bytes(len_bytes);

    // Guard against corrupt files reporting implausible header sizes.
    const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB
    if raw_len > MAX_HEADER_BYTES {
        return Err(ArtifactError::WrongModelKind {
            expected: "valid safetensors file",
            keys_hint: format!("header length {raw_len} exceeds 100 MiB limit"),
        });
    }
    let header_len = usize::try_from(raw_len).expect("bounded by MAX_HEADER_BYTES");

    let mut header_json = vec![0u8; header_len];
    file.read_exact(&mut header_json)
        .map_err(|e| ArtifactError::WrongModelKind {
            expected: "valid safetensors file",
            keys_hint: format!("cannot read header: {e}"),
        })?;

    let obj: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header_json)
        .map_err(|e| ArtifactError::WrongModelKind {
            expected: "valid safetensors file",
            keys_hint: format!("header JSON parse error: {e}"),
        })?;

    for (k, _) in obj {
        if k != "__metadata__" {
            f(&k);
        }
    }
    Ok(())
}

/// Collect all tensor key names from a safetensors header into a `Vec`.
#[cfg(test)]
fn read_safetensors_keys(path: &Path) -> Result<Vec<String>, ArtifactError> {
    let mut keys = Vec::new();
    scan_safetensors_keys(path, |k| keys.push(k.to_owned()))?;
    Ok(keys)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative backbone tensor key that satisfies `MODERNBERT_KEY_PREFIXES`.
    const FAKE_BACKBONE_KEY: &str = "layers.0.attn.Wo.weight";

    /// Write a minimal but structurally valid safetensors file containing the given
    /// tensor keys. Each tensor has one `f32` element of weight data.
    fn write_fake_safetensors(path: &Path, tensor_keys: &[&str]) {
        let mut header_obj = serde_json::Map::new();
        header_obj.insert("__metadata__".to_owned(), serde_json::json!({}));
        let mut offset = 0usize;
        for &key in tensor_keys {
            let end = offset + 4; // 4 bytes per f32
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

    /// Minimal BPE tokenizer JSON accepted by tokenizers 0.22+.
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
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

    use crate::test_support::VALID_CONFIG_JSON;

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
}
