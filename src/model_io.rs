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
pub fn read_config<T: DeserializeOwned>(path: &Path) -> Result<T, ModelIoError> {
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
    // Extract `&'static str` identifiers up-front so the worker closure does
    // not need to carry the generic `Id` across the thread boundary (which
    // would force a `Send + 'static` bound on every caller).
    let repo_id: &'static str = model.repo_id();
    let revision: &'static str = model.revision();

    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = (|| -> Result<ModelPaths, ModelIoError> {
            let api = Api::new()
                .map_err(|e| ModelIoError::Download(format!("HF Hub init failed: {e}")))?;
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
        })();
        let _ = tx.send(result);
    });

    rx.recv_timeout(DOWNLOAD_TIMEOUT).map_err(|e| match e {
        mpsc::RecvTimeoutError::Timeout => ModelIoError::Download(format!(
            "download exceeded {} second timeout (HF Hub may be unreachable or proxy stalled)",
            DOWNLOAD_TIMEOUT.as_secs()
        )),
        mpsc::RecvTimeoutError::Disconnected => {
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
/// Embed and reranker callers both size their sub-batches against this budget
/// so memory consumption is bounded by the same ceiling regardless of which
/// path drives a given call.
pub(crate) const TOKEN_BUDGET: usize = 256_000;

/// Compute the sub-batch size for a bucket of length `bucket_len`.
///
/// Returns `(TOKEN_BUDGET / bucket_len).max(1)`. Callers iterate
/// `items.chunks(sub_batch_size)` so each forward pass stays under the
/// [`TOKEN_BUDGET`] ceiling even when every item sits at the bucket boundary.
/// The `max(1)` floor keeps the loop progressing if a future widening of
/// [`BUCKET_BOUNDS`] makes a bucket exceed [`TOKEN_BUDGET`]; production
/// callers always pass `BUCKET_BOUNDS[i]` (≤ 8192) so that branch never
/// fires today.
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

    // ── compute_sub_batch_size tests ────────────────────────────────────────

    // Pin the embed-side sub-batch formula `(TOKEN_BUDGET / bucket_len).max(1)`
    // so any second caller cannot drift from embed silently.
    #[test]
    fn compute_sub_batch_size_matches_embed_formula_per_bucket() {
        assert_eq!(
            compute_sub_batch_size(BUCKET_BOUNDS[0]),
            TOKEN_BUDGET / BUCKET_BOUNDS[0],
        );
        assert_eq!(
            compute_sub_batch_size(BUCKET_BOUNDS[1]),
            TOKEN_BUDGET / BUCKET_BOUNDS[1],
        );
        assert_eq!(
            compute_sub_batch_size(BUCKET_BOUNDS[2]),
            TOKEN_BUDGET / BUCKET_BOUNDS[2],
        );
        assert_eq!(
            compute_sub_batch_size(BUCKET_BOUNDS[3]),
            TOKEN_BUDGET / BUCKET_BOUNDS[3],
        );
    }

    // `max(1)` floor: pathological bucket_len > TOKEN_BUDGET cannot happen in
    // production (BUCKET_BOUNDS[3]=8192 < 256_000), but the guard keeps callers
    // from looping over zero-sized chunks if BUCKET_BOUNDS is later widened.
    #[test]
    fn compute_sub_batch_size_returns_one_when_bucket_exceeds_token_budget() {
        assert_eq!(compute_sub_batch_size(TOKEN_BUDGET + 1), 1);
        assert_eq!(compute_sub_batch_size(usize::MAX), 1);
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
        assert_eq!(ModelId::default().repo_id(), "cl-nagoya/ruri-v3-310m");
    }
}
