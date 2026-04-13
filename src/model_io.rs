//! Shared model I/O utilities for the ruri-v3 model family.
//!
//! Provides config/tokenizer loading, model paths, and model constants shared by
//! both [`embed`](crate::embed) and [`reranker`](crate::reranker) modules.

use std::path::{Path, PathBuf};

/// EOS (end of sequence) token ID for ruri-v3 models.
pub(crate) const EOS_TOKEN_ID: u32 = 2;

/// Maximum sequence length for ruri-v3 models (max_position_embeddings).
pub const MAX_SEQ_LEN: usize = 8192;

/// Truncate `ids` and `mask` to `max_len` in place, replacing the last token with EOS.
///
/// Returns `true` if truncation was performed. A `max_len` of 0 is a no-op.
pub(crate) fn truncate_with_eos(ids: &mut Vec<u32>, mask: &mut Vec<u32>, max_len: usize) -> bool {
    if max_len == 0 || ids.len() <= max_len {
        return false;
    }
    ids.truncate(max_len);
    ids[max_len - 1] = EOS_TOKEN_ID;
    mask.truncate(max_len);
    true
}

/// Implemented by model ID enums to provide HuggingFace repository metadata.
pub trait ModelArtifact: Copy {
    /// HuggingFace repository ID (e.g., `"cl-nagoya/ruri-v3-310m"`).
    fn repo_id(self) -> &'static str;
    /// Pinned commit hash for reproducible downloads.
    fn revision(self) -> &'static str;
}

/// Errors from model I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelIoError {
    /// Config file missing or unparseable.
    #[error("config error at {path}: {reason}")]
    Config {
        /// Config file path.
        path: PathBuf,
        /// Parse or IO failure detail.
        reason: String,
    },
    /// Tokenizer load failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Model download failure.
    #[error("download failed: {0}")]
    Download(String),
}

/// Paths to the three model artifacts (weights, config, tokenizer).
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// SafeTensors weights file.
    pub model: PathBuf,
    /// `config.json` (ModernBERT hyperparameters).
    pub config: PathBuf,
    /// `tokenizer.json` (HuggingFace tokenizer).
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// Build paths assuming standard filenames under `dir`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }
}

/// Deserialize a JSON config file into `T`.
///
/// # Errors
///
/// Returns [`ModelIoError::Config`] on IO failure or JSON parse error.
pub fn read_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ModelIoError> {
    let text = std::fs::read_to_string(path).map_err(|e| ModelIoError::Config {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| ModelIoError::Config {
        path: path.to_path_buf(),
        reason: format!("parse error: {e}"),
    })
}

/// Load a HuggingFace tokenizer from a JSON file.
///
/// # Errors
///
/// Returns [`ModelIoError::Tokenizer`] if the tokenizer file cannot be read or
/// parsed by the `tokenizers` crate.
pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer, ModelIoError> {
    tokenizers::Tokenizer::from_file(path).map_err(|e| ModelIoError::Tokenizer(e.to_string()))
}

fn model_repo_for<Id: ModelArtifact>(model: Id) -> hf_hub::Repo {
    hf_hub::Repo::with_revision(
        model.repo_id().to_string(),
        hf_hub::RepoType::Model,
        model.revision().to_string(),
    )
}

/// Download model files from Hugging Face Hub (cached after first download).
///
/// # Errors
///
/// Returns [`ModelIoError::Download`] if the Hugging Face client cannot be
/// initialized or any required artifact download fails.
///
/// # Interruption Safety
///
/// hf-hub downloads each file to a `.part` temporary file and then atomically
/// renames it to the final blob path (`std::fs::rename`). A signal (e.g. SIGINT)
/// received during a download leaves only the `.part` file on disk; the final
/// blob is never partially written.
///
/// On the next invocation, [`artifacts_if_cached`] finds no valid pointer and
/// returns `None`, so the download retries cleanly — no manual cache cleanup needed.
pub fn download_artifacts<Id: ModelArtifact>(model: Id) -> Result<ModelPaths, ModelIoError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| ModelIoError::Download(format!("HF Hub init failed: {e}")))?;
    let repo = api.repo(model_repo_for(model));

    let get = |name: &str| {
        repo.get(name)
            .map_err(|e| ModelIoError::Download(format!("{name} download failed: {e}")))
    };

    Ok(ModelPaths {
        model: get("model.safetensors")?,
        config: get("config.json")?,
        tokenizer: get("tokenizer.json")?,
    })
}

/// Check whether model files exist in the local HF Hub cache.
///
/// Returns `Ok(Some(paths))` if all three files are cached, `Ok(None)` otherwise.
/// Never accesses the network.
pub fn artifacts_if_cached<Id: ModelArtifact>(
    model: Id,
) -> Result<Option<ModelPaths>, ModelIoError> {
    artifacts_from_cache(&hf_hub::Cache::from_env(), model)
}

pub(crate) fn artifacts_from_cache<Id: ModelArtifact>(
    cache: &hf_hub::Cache,
    model: Id,
) -> Result<Option<ModelPaths>, ModelIoError> {
    let repo = cache.repo(model_repo_for(model));
    let Some(model_weights) = repo.get("model.safetensors") else {
        return Ok(None);
    };
    let Some(config) = repo.get("config.json") else {
        return Ok(None);
    };
    let Some(tokenizer) = repo.get("tokenizer.json") else {
        return Ok(None);
    };
    Ok(Some(ModelPaths {
        model: model_weights,
        config,
        tokenizer,
    }))
}

/// Pad variable-length token sequences into contiguous flat arrays for batched inference.
///
/// Zero-pads shorter sequences to `max_len` (the longest sequence in the batch).
/// When `masks` is `None`, generates an identity mask (1 for each token, 0 for padding).
///
/// Returns `(flat_ids, flat_mask, batch_size, max_len)`.
///
/// # Panics
///
/// Panics if `masks` is `Some` and its length differs from `ids`.
pub(crate) fn pad_sequences(
    ids: &[Vec<u32>],
    masks: Option<&[Vec<u32>]>,
) -> (Vec<u32>, Vec<u32>, usize, usize) {
    debug_assert!(
        masks.is_none_or(|m| m.len() == ids.len()),
        "masks length {} != ids length {}",
        masks.map_or(0, |m| m.len()),
        ids.len(),
    );

    let batch_size = ids.len();
    let max_len = ids.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut flat_ids = vec![0u32; batch_size * max_len];
    let mut flat_mask = vec![0u32; batch_size * max_len];

    for (i, id_seq) in ids.iter().enumerate() {
        let offset = i * max_len;
        flat_ids[offset..offset + id_seq.len()].copy_from_slice(id_seq);
        match masks {
            Some(m) => flat_mask[offset..offset + m[i].len()].copy_from_slice(&m[i]),
            None => flat_mask[offset..offset + id_seq.len()].fill(1),
        }
    }

    (flat_ids, flat_mask, batch_size, max_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_config_returns_error_for_missing_file() {
        let err =
            read_config::<serde_json::Value>(Path::new("/nonexistent/config.json")).unwrap_err();
        assert!(matches!(err, ModelIoError::Config { .. }), "{err}");
    }

    #[test]
    fn read_config_returns_error_for_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"not json").unwrap();
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"{ \"vocab_size\": 1000 }").unwrap();
        let err = read_config::<crate::modernbert::Config>(&path).unwrap_err();
        assert!(
            matches!(err, ModelIoError::Config { ref reason, .. } if reason.contains("parse error")),
            "{err}"
        );
    }

    // ── pad_sequences tests ──────────────────────────────────────────────

    #[test]
    fn pad_sequences_no_masks_generates_identity() {
        let ids = vec![vec![1, 2, 3], vec![4, 5]];
        let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None);
        assert_eq!(batch, 2);
        assert_eq!(max_len, 3);
        assert_eq!(flat_ids, vec![1, 2, 3, 4, 5, 0]);
        assert_eq!(flat_mask, vec![1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn pad_sequences_with_masks_copies_mask() {
        let ids = vec![vec![10, 20], vec![30]];
        let masks = vec![vec![1, 1], vec![1]];
        let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, Some(&masks));
        assert_eq!(batch, 2);
        assert_eq!(max_len, 2);
        assert_eq!(flat_ids, vec![10, 20, 30, 0]);
        assert_eq!(flat_mask, vec![1, 1, 1, 0]);
    }

    #[test]
    fn pad_sequences_empty_input() {
        let ids: Vec<Vec<u32>> = vec![];
        let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None);
        assert_eq!(batch, 0);
        assert_eq!(max_len, 0);
        assert!(flat_ids.is_empty());
        assert!(flat_mask.is_empty());
    }

    #[test]
    fn pad_sequences_single_sequence() {
        let ids = vec![vec![1, 2, 3, 4]];
        let (flat_ids, flat_mask, batch, max_len) = pad_sequences(&ids, None);
        assert_eq!(batch, 1);
        assert_eq!(max_len, 4);
        assert_eq!(flat_ids, vec![1, 2, 3, 4]);
        assert_eq!(flat_mask, vec![1, 1, 1, 1]);
    }
}
