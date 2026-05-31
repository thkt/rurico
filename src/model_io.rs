//! Shared model I/O utilities for the ruri-v3 model family.
//!
//! Provides config/tokenizer loading, model paths, and model constants shared by
//! both [`embed`](crate::embed) and [`reranker`](crate::reranker) modules.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use hf_hub::api::sync::Api;
use serde::de::DeserializeOwned;

use crate::artifacts::ModelKind;

/// Hard upper bound for a full `download_model` call (init + 3 file downloads).
///
/// Guards against an unreachable HF Hub endpoint or stalled corporate proxy
/// from blocking the calling thread indefinitely. The download thread itself
/// is detached on timeout and continues until the OS reclaims it on process
/// exit; the user-visible behaviour is a deterministic `ModelIoError::Download`
/// after the deadline.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// EOS (end of sequence) token ID for ruri-v3 models.
pub(crate) const EOS_TOKEN_ID: u32 = 2;

/// Maximum sequence length for ruri-v3 models (max_position_embeddings).
pub const MAX_SEQ_LEN: usize = 8192;

/// Length-bucket upper bounds shared by embed and reranker forward paths.
///
/// All `model.forward(..., seq_len)` callers (chunk encoder, query encoder,
/// reranker pair scorer) round their actual `seq_len` up to one of these four
/// values. This keeps the per-`seq_len` mask cache inside [`ModernBert`] bounded
/// to four entries and turns the global MLX compile cache into a fixed working
/// set of four kernels per model rather than one per observed length.
///
/// [`ModernBert`]: crate::modernbert::ModernBert
pub(crate) const BUCKET_BOUNDS: [usize; 4] = [128, 512, 2048, MAX_SEQ_LEN];

/// Assign `len` to the smallest bucket index `i` with `BUCKET_BOUNDS[i] >= len`.
///
/// # Panics
///
/// Panics if `len > MAX_SEQ_LEN`. Callers must truncate or shrink the sequence
/// before bucket assignment.
pub(crate) fn assign_bucket(len: usize) -> usize {
    BUCKET_BOUNDS
        .iter()
        .position(|&max| len <= max)
        .expect("len exceeds MAX_SEQ_LEN; truncate before bucketing")
}

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
///
/// The `Kind` associated type binds the identifier to a model kind marker so
/// that `download_model::<Id>` and `cached_artifacts::<Id>` can return
/// `VerifiedArtifacts<Id::Kind>` without taking a redundant kind parameter and
/// without admitting wrong-kind combinations at the call site.
pub trait ModelArtifact: Copy {
    /// Kind marker bound to this identifier (e.g. `EmbedKind` for embed model IDs).
    type Kind: ModelKind;
    /// HuggingFace repository ID (e.g., `"cl-nagoya/ruri-v3-310m"`).
    fn repo_id(self) -> &'static str;
    /// Pinned commit hash for reproducible downloads.
    fn revision(self) -> &'static str;
}

/// Errors from model I/O operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
pub(crate) fn read_config<T: DeserializeOwned>(path: &Path) -> Result<T, ModelIoError> {
    let text = fs::read_to_string(path).map_err(|e| ModelIoError::Config {
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

fn make_repo(repo_id: &str, revision: &str) -> hf_hub::Repo {
    hf_hub::Repo::with_revision(
        repo_id.to_owned(),
        hf_hub::RepoType::Model,
        revision.to_owned(),
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
    download_artifacts_with(model, DOWNLOAD_TIMEOUT, |repo_id, revision| {
        let api =
            Api::new().map_err(|e| ModelIoError::Download(format!("HF Hub init failed: {e}")))?;
        let repo = api.repo(make_repo(repo_id, revision));
        let get = |name: &str| {
            repo.get(name)
                .map_err(|e| ModelIoError::Download(format!("{name} download failed: {e}")))
        };
        Ok(ModelPaths {
            model: get("model.safetensors")?,
            config: get("config.json")?,
            tokenizer: get("tokenizer.json")?,
        })
    })
}

/// Generic seam — worker closure decides how to materialize `ModelPaths`.
///
/// The orchestration (thread spawn / mpsc / `recv_timeout`) and the actual
/// download are split here so unit tests can pin worker success / failure /
/// timeout paths without driving real HF Hub I/O. Production calls go
/// through [`download_artifacts`].
pub(crate) fn download_artifacts_with<Id, F>(
    model: Id,
    timeout: Duration,
    download: F,
) -> Result<ModelPaths, ModelIoError>
where
    Id: ModelArtifact,
    F: FnOnce(&str, &str) -> Result<ModelPaths, ModelIoError> + Send + 'static,
{
    let repo_id: &'static str = model.repo_id();
    let revision: &'static str = model.revision();

    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = tx.send(download(repo_id, revision));
    });

    rx.recv_timeout(timeout).map_err(|e| match e {
        mpsc::RecvTimeoutError::Timeout => {
            tracing::error!(
                repo_id,
                revision,
                timeout_secs = timeout.as_secs(),
                "download_artifacts: HF Hub download timed out"
            );
            ModelIoError::Download(format!(
                "download exceeded {} second timeout (HF Hub may be unreachable or proxy stalled)",
                timeout.as_secs()
            ))
        }
        mpsc::RecvTimeoutError::Disconnected => {
            tracing::error!(
                repo_id,
                revision,
                "download_artifacts: worker thread disconnected before sending result"
            );
            ModelIoError::Download("download worker terminated unexpectedly".to_owned())
        }
    })?
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
    let repo = cache.repo(make_repo(model.repo_id(), model.revision()));
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

/// Token-count ceiling for a single forward pass through the bucketed
/// inference path.
///
/// 256K-position budget per sub-batch — chosen empirically to keep padded
/// MLX tensors under safe Apple Silicon GPU memory ceilings and avoid
/// Metal OOM (commit `3c86e90`). Combined with [`BUCKET_BOUNDS`], the
/// derived `(TOKEN_BUDGET / bucket_len)` sub-batch sizes are
/// `(2000, 500, 125, 31)` for bucket lengths `(128, 512, 2048, 8192)`,
/// pinned by `compute_sub_batch_size_matches_formula_per_bucket`.
///
/// Embed and reranker callers both size their sub-batches against this budget
/// so memory consumption is bounded by the same ceiling regardless of which
/// path drives a given call. See `docs/decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md`
/// for the cross-module forward-pass invariant.
pub(crate) const TOKEN_BUDGET: usize = 256_000;

/// Sub-batch size that keeps each forward pass under [`TOKEN_BUDGET`] when
/// every item sits at the bucket boundary. Callers iterate
/// `items.chunks(sub_batch_size)`. Floors at 1 so the loop progresses for
/// `bucket_len > TOKEN_BUDGET`.
pub(crate) fn compute_sub_batch_size(bucket_len: usize) -> usize {
    (TOKEN_BUDGET / bucket_len).max(1)
}

/// Pad variable-length token sequences into contiguous flat arrays for batched inference.
///
/// Zero-pads shorter sequences to `max_len`. When `target_len` is `None`,
/// `max_len` equals the longest sequence in the batch; when `Some(t)`,
/// `max_len = max(t, longest)`. Callers pass `Some(BUCKET_BOUNDS[i])` to keep
/// `model.forward` shapes aligned to the four bucket ceilings.
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
    target_len: Option<usize>,
) -> (Vec<u32>, Vec<u32>, usize, usize) {
    debug_assert!(
        masks.is_none_or(|m| m.len() == ids.len()),
        "masks length {} != ids length {}",
        masks.map_or(0, <[Vec<u32>]>::len),
        ids.len(),
    );

    let batch_size = ids.len();
    let actual_max = ids.iter().map(Vec::len).max().unwrap_or(0);
    let max_len = target_len.map_or(actual_max, |t| t.max(actual_max));

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
mod tests;
