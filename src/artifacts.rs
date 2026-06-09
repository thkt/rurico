//! Typed artifact verification for ruri-v3 model files.
//!
//! Provides a two-phase pipeline: the crate builds an internal
//! [`CandidateArtifacts`](crate::artifacts::CandidateArtifacts) (unverified) and
//! verifies it into [`VerifiedArtifacts<K>`](crate::artifacts::VerifiedArtifacts)
//! (verified, kind-checked) — the type downstream actually receives.
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
///
/// **Sealed**: the private `()` field prevents external crates from constructing
/// this marker, ensuring the kind ↔ model id binding (via
/// `ModelArtifact::Kind`) is enforced
/// at the type level. Downstream callers cannot construct a kind unrelated to
/// the model id they hold.
#[derive(Debug)]
pub struct EmbedKind(());

/// Kind marker for reranker models (ruri-v3-reranker variants).
///
/// **Sealed**: same sealed-pattern as [`EmbedKind`] — the private `()` field
/// blocks external construction, so kind ↔ model id binding stays type-level.
#[derive(Debug)]
pub struct RerankerKind(());

// ── VerifiedArtifacts ────────────────────────────────────────────────────────

/// Verified model artifacts, bound to a specific model kind `K`.
///
/// Obtained from the model lifecycle entry points
/// [`download_model`](crate::model_lifecycle::download_model) /
/// [`cached_artifacts`](crate::model_lifecycle::cached_artifacts), which build
/// and verify the artifacts internally. Pass the result to the kind's runtime
/// constructor (`Embedder::new` / `Reranker::new`).
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
#[non_exhaustive]
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

// ── ModelKind trait ──────────────────────────────────────────────────────────

/// Per-kind verifier dispatch for [`CandidateArtifacts`].
///
/// Implemented by the kind markers ([`EmbedKind`] / [`RerankerKind`]) so that
/// `CandidateArtifacts<K>::verify` can hand the paths to the kind-specific
/// safetensors header check (`verify_as_embed` / `verify_as_reranker`) without
/// duplicating per-kind glue at the call site.
pub trait ModelKind: Sized {
    /// Verify `paths` against this kind's safetensors signature.
    ///
    /// # Errors
    ///
    /// Returns the same [`ArtifactError`] variants as the underlying
    /// kind-specific verifier (`verify_as_embed` / `verify_as_reranker`).
    fn verify(paths: ModelPaths) -> Result<VerifiedArtifacts<Self>, ArtifactError>;
}

impl ModelKind for EmbedKind {
    fn verify(paths: ModelPaths) -> Result<VerifiedArtifacts<Self>, ArtifactError> {
        verify_as_embed(paths)
    }
}

impl ModelKind for RerankerKind {
    fn verify(paths: ModelPaths) -> Result<VerifiedArtifacts<Self>, ArtifactError> {
        verify_as_reranker(paths)
    }
}

// ── CandidateArtifacts ───────────────────────────────────────────────────────

/// Unverified model artifact paths bound to a kind marker `K`.
///
/// Internal staging type in the verification pipeline. The model lifecycle
/// entry points ([`download_model`](crate::model_lifecycle::download_model) /
/// [`cached_artifacts`](crate::model_lifecycle::cached_artifacts)) build a
/// `CandidateArtifacts` from a download or the local cache, then call
/// [`verify`](Self::verify) to obtain a [`VerifiedArtifacts<K>`] that feeds the
/// kind's runtime constructor (`Embedder::new` / `Reranker::new`).
///
/// Downstream crates receive `VerifiedArtifacts<K>` directly from those entry
/// points and do not construct `CandidateArtifacts` themselves: the
/// constructors are crate-internal by design (the `pub(crate)` visibility of
/// `from_paths` is pinned by `tests/ui/from_paths_pub_crate.rs`).
pub struct CandidateArtifacts<K> {
    pub(crate) paths: ModelPaths,
    _kind: PhantomData<K>,
}

impl<K> fmt::Debug for CandidateArtifacts<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CandidateArtifacts")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl<K: ModelKind> CandidateArtifacts<K> {
    /// Construct from explicit file paths without verification.
    pub(crate) fn from_paths(model: PathBuf, config: PathBuf, tokenizer: PathBuf) -> Self {
        Self {
            paths: ModelPaths {
                model,
                config,
                tokenizer,
            },
            _kind: PhantomData,
        }
    }

    /// Construct from a directory using standard filenames (`model.safetensors`,
    /// `config.json`, `tokenizer.json`). Available for development and test use only.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            paths: ModelPaths::from_dir(dir),
            _kind: PhantomData,
        }
    }

    /// Verify file existence, config integrity, tokenizer validity, and kind dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for missing files, invalid config, invalid tokenizer,
    /// or weights that do not match the bound kind `K`.
    pub fn verify(self) -> Result<VerifiedArtifacts<K>, ArtifactError> {
        K::verify(self.paths)
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

/// Tensor key prefixes that identify ModernBERT-based backbone weights.
///
/// Two naming conventions are accepted (any-of):
/// - `"layers."` — flat MLX convention used by ruri-v3 embed checkpoints
///   (e.g. `cl-nagoya/ruri-v3-310m`).
/// - `"model.layers."` — HuggingFace transformers wrapping convention used by
///   sequence-classification heads such as `ModernBertForSequenceClassification`
///   (e.g. `cl-nagoya/ruri-v3-reranker-310m`).
///
/// At least one prefix must match a key in the safetensors header for the
/// backbone check to succeed.
const MODERNBERT_KEY_PREFIXES: &[&str] = &["layers.", "model.layers."];

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
/// Backbone check: at least one prefix from [`MODERNBERT_KEY_PREFIXES`] must
/// match (any-of). This accepts both flat MLX checkpoints (`"layers."`) and
/// HuggingFace-wrapped checkpoints (`"model.layers."`).
///
/// Reranker head check (`RERANKER_KEY_PREFIXES`): when `require_reranker_keys`
/// is `true`, every classifier/head prefix must match (all-of); when `false`,
/// none of them may match.
fn verify_model_kind(
    paths: &ModelPaths,
    expected: &'static str,
    require_reranker_keys: bool,
) -> Result<(), ArtifactError> {
    let mut backbone_seen = false;
    let mut reranker_seen = [false; RERANKER_KEY_PREFIXES.len()];

    scan_safetensors_keys(&paths.model, |k| {
        if !backbone_seen {
            for prefix in MODERNBERT_KEY_PREFIXES {
                if k.starts_with(prefix) {
                    backbone_seen = true;
                    break;
                }
            }
        }
        for (i, prefix) in RERANKER_KEY_PREFIXES.iter().enumerate() {
            if !reranker_seen[i] && k.starts_with(prefix) {
                reranker_seen[i] = true;
            }
        }
    })?;

    if !backbone_seen {
        return Err(ArtifactError::WrongModelKind {
            expected,
            keys_hint: format!(
                "missing required key with any of prefixes {MODERNBERT_KEY_PREFIXES:?}"
            ),
        });
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
mod tests;
