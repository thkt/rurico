mod mlx;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

use self::mlx::RerankerInner;
use crate::artifacts::verify_as_reranker;
use crate::model_io::{ModelArtifact, ModelPaths, artifacts_if_cached, download_artifacts};
use crate::model_probe::{
    ProbeError, RERANKER_PROBE_ENV_CONFIG, RERANKER_PROBE_ENV_MODEL, RERANKER_PROBE_ENV_TOKENIZER,
    probe_paths_via_subprocess, resolve_probe_env,
};
use std::fmt::{self, Debug, Display, Formatter};
#[cfg(any(test, feature = "test-support"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

pub use crate::artifacts::{ArtifactError, RerankerKind, VerifiedArtifacts};
pub use crate::model_probe::ProbeStatus;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::MockReranker;

// ── Domain alias ─────────────────────────────────────────────────────────────

/// Verified reranker model artifacts.
///
/// Produced by [`CandidateArtifacts::verify`] or [`download_model`].
/// Guarantees that all three artifact files exist, that `config.json` parses
/// as a valid [`crate::modernbert::Config`], that `tokenizer.json` loads
/// without error, and that the weights are for a reranker model (not an embed model).
pub type Artifacts = VerifiedArtifacts<RerankerKind>;

// ── CandidateArtifacts ───────────────────────────────────────────────────────

/// Unverified reranker model artifact paths.
///
/// Construct with [`from_paths`](Self::from_paths) (or [`from_dir`](Self::from_dir)
/// in test contexts), then call [`verify`](Self::verify) to obtain
/// [`Artifacts`] that can be passed to [`Reranker::new`].
#[derive(Debug)]
pub struct CandidateArtifacts {
    paths: ModelPaths,
}

impl CandidateArtifacts {
    /// Construct from explicit file paths without any verification.
    ///
    /// Use this when you have raw paths (e.g. from environment variables).
    /// Call [`verify`](Self::verify) before passing the result to [`Reranker::new`].
    pub fn from_paths(model: PathBuf, config: PathBuf, tokenizer: PathBuf) -> Self {
        Self {
            paths: ModelPaths {
                model,
                config,
                tokenizer,
            },
        }
    }

    /// Construct from a directory using standard filenames (`model.safetensors`,
    /// `config.json`, `tokenizer.json`).
    ///
    /// Available for development and test use only.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            paths: ModelPaths::from_dir(dir),
        }
    }

    /// Verify file existence, config integrity, tokenizer validity, and reranker model kind.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for any of the following:
    /// - A required file is missing ([`ArtifactError::MissingFile`])
    /// - `config.json` cannot be parsed ([`ArtifactError::InvalidConfig`])
    /// - `tokenizer.json` cannot be loaded ([`ArtifactError::InvalidTokenizer`])
    /// - Weights are for an embed model, not a reranker ([`ArtifactError::WrongModelKind`])
    pub fn verify(self) -> Result<Artifacts, ArtifactError> {
        verify_as_reranker(self.paths)
    }
}

// ── RerankerModelId ───────────────────────────────────────────────────────────

/// Identifies a ruri-v3 reranker model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankerModelId {
    /// `cl-nagoya/ruri-v3-reranker-310m`.
    #[default]
    RuriV3Reranker310m,
}

impl RerankerModelId {
    /// HuggingFace repository ID for this model.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::RuriV3Reranker310m => "cl-nagoya/ruri-v3-reranker-310m",
        }
    }

    fn revision(self) -> &'static str {
        match self {
            Self::RuriV3Reranker310m => "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3",
        }
    }
}

impl ModelArtifact for RerankerModelId {
    fn repo_id(self) -> &'static str {
        RerankerModelId::repo_id(self)
    }

    fn revision(self) -> &'static str {
        self.revision()
    }
}

// ── RerankerInitError ─────────────────────────────────────────────────────────

/// Errors from initialising the reranker backend ([`Reranker::new`], [`Reranker::probe`]).
///
/// These errors occur after artifact verification has already succeeded. They
/// indicate a failure during MLX backend setup or model weight loading.
#[derive(Debug, thiserror::Error)]
pub enum RerankerInitError {
    /// MLX backend initialisation, weight loading, or subprocess probe failure.
    #[error("reranker init failed: {0}")]
    Backend(String),
    /// Model weights loaded but are corrupt or incompatible with the expected architecture.
    #[error("model load failed: {reason}")]
    ModelCorrupt {
        /// Failure detail from the backend.
        reason: String,
    },
}

impl RerankerInitError {
    pub(crate) fn backend(e: impl Display) -> Self {
        Self::Backend(e.to_string())
    }
}

impl From<ProbeError> for RerankerInitError {
    fn from(e: ProbeError) -> Self {
        match e {
            ProbeError::HandlerNotInstalled => RerankerInitError::Backend(e.to_string()),
            ProbeError::ModelLoadFailed { reason } => RerankerInitError::ModelCorrupt { reason },
            ProbeError::SubprocessFailed(msg) => RerankerInitError::Backend(msg),
        }
    }
}

// ── RerankerError (runtime) ───────────────────────────────────────────────────

/// Errors from reranker operations at runtime.
///
/// These errors occur during calls to [`Rerank`] trait methods after the
/// [`Reranker`] has been successfully initialised.
#[derive(Debug, thiserror::Error)]
pub enum RerankerError {
    /// MLX inference failure during a forward pass.
    #[error("inference error: {0}")]
    Inference(String),
    /// Tokenizer encode failure (e.g. unsupported character sequence).
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Model output contains NaN or infinity.
    #[error("non-finite values in reranker output (NaN or inf)")]
    NonFiniteOutput,
}

impl RerankerError {
    pub(crate) fn inference(e: impl Display) -> Self {
        Self::Inference(e.to_string())
    }
}

// ── RankedResult ─────────────────────────────────────────────────────────────

/// A scored document with its original position in the input slice.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Index in the original `documents` slice.
    pub index: usize,
    /// Relevance score in `[0, 1]`.
    pub score: f32,
}

// ── Rerank trait ──────────────────────────────────────────────────────────────

/// Reranking provider.
///
/// Thread-safe: uses `&self` so implementors can be shared via `Arc<dyn Rerank>`.
/// Implementors that hold mutable state should use interior mutability (e.g. `Mutex`).
pub trait Rerank: Send + Sync {
    /// Score a single (query, document) pair.
    ///
    /// Returns `sigmoid(logit)` as `f32` in `[0, 1]`.
    fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError>;

    /// Score multiple (query, document) pairs in a single batched forward pass.
    ///
    /// Returns one score per pair in input order.
    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError>;

    /// Rerank documents by relevance to a query.
    ///
    /// Returns `Vec<RankedResult>` sorted by score descending.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError>;
}

// ── Reranker ─────────────────────────────────────────────────────────────────

/// Thread-safe cross-encoder reranker backed by MLX.
///
/// Uses ~1.2 GB GPU memory (estimated from ruri-v3-reranker-310m safetensors
/// ~620 MB x FP16 load). Simultaneously holding both a `Reranker` and an
/// [`Embedder`](crate::embed::Embedder) uses approximately 1.2 GB of
/// additional GPU memory.
pub struct Reranker {
    inner: Mutex<RerankerInner>,
}

impl Debug for Reranker {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reranker").finish_non_exhaustive()
    }
}

impl Reranker {
    /// Load reranker weights, config, and tokenizer from verified `artifacts`.
    ///
    /// # Errors
    ///
    /// Returns [`RerankerInitError::Backend`] if MLX model construction or weight loading fails.
    pub fn new(artifacts: &Artifacts) -> Result<Self, RerankerInitError> {
        let inner = RerankerInner::new(artifacts)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Score a single (query, document) pair.
    ///
    /// Returns `sigmoid(logit)` as `f32` in `[0, 1]`.
    ///
    /// Scores are valid only for relative ranking within the same query.
    /// Do not compare scores across different queries or mix with embedding
    /// similarity scores. Absolute score values may change with model updates.
    pub fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError> {
        let scores = self.score_batch(&[(query, document)])?;
        Ok(scores[0])
    }

    /// Score multiple (query, document) pairs in a single batched forward pass.
    ///
    /// Returns one score per pair in input order.
    ///
    /// Scores are valid only for relative ranking within the same query.
    /// Do not compare scores across different queries or mix with embedding
    /// similarity scores. Absolute score values may change with model updates.
    pub fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        self.lock_inner()?.score_batch(pairs)
    }

    /// Rerank documents by relevance to a query.
    ///
    /// Returns `Vec<RankedResult>` sorted by score descending. Each result
    /// contains the original index in `documents` and the relevance score.
    pub fn rerank(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<RankedResult>, RerankerError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let pairs: Vec<(&str, &str)> = documents.iter().map(|&d| (query, d)).collect();
        let scores = self.score_batch(&pairs)?;
        Ok(sort_results(&scores))
    }

    /// Test whether the reranker model can load without aborting the caller.
    ///
    /// Re-execs the current binary as a probe subprocess (via `Command`), so a
    /// crash is contained and reported as [`ProbeStatus::BackendUnavailable`].
    ///
    /// The host binary must call
    /// [`model_probe::handle_probe_if_needed`](crate::model_probe::handle_probe_if_needed)
    /// at the start of `main()`.
    pub fn probe(artifacts: &Artifacts) -> Result<ProbeStatus, RerankerInitError> {
        probe_via_subprocess(artifacts)
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, RerankerInner>, RerankerError> {
        self.inner
            .lock()
            .map_err(|_| RerankerError::inference("reranker lock poisoned"))
    }
}

impl Rerank for Reranker {
    fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError> {
        Reranker::score(self, query, document)
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        Reranker::score_batch(self, pairs)
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
        Reranker::rerank(self, query, documents)
    }
}

fn sort_results(scores: &[f32]) -> Vec<RankedResult> {
    let mut results: Vec<RankedResult> = scores
        .iter()
        .enumerate()
        .map(|(index, &score)| RankedResult { index, score })
        .collect();
    results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    results
}

// ── Public API: download / cache ──────────────────────────────────────────────

/// Download reranker model files from Hugging Face Hub and verify them as reranker artifacts.
///
/// # Errors
///
/// Returns [`ArtifactError::DownloadFailed`] if the Hugging Face client cannot be
/// initialised or any required artifact download fails.
/// Returns other [`ArtifactError`] variants if verification of the downloaded
/// files fails.
pub fn download_model(model: RerankerModelId) -> Result<Artifacts, ArtifactError> {
    let paths = download_artifacts(model)?;
    verify_as_reranker(paths)
}

/// Check whether reranker model files exist in the local HF Hub cache and verify them.
///
/// Returns `Ok(Some(artifacts))` if all three files are cached and pass
/// verification, `Ok(None)` otherwise. Never accesses the network.
///
/// # Errors
///
/// Returns [`ArtifactError`] if cached files fail verification.
/// Cache misses are reported as `Ok(None)`.
pub fn cached_artifacts(model: RerankerModelId) -> Result<Option<Artifacts>, ArtifactError> {
    let Some(paths) = artifacts_if_cached(model)? else {
        return Ok(None);
    };
    verify_as_reranker(paths).map(Some)
}

// ── Probe infrastructure ──────────────────────────────────────────────────────

/// Resolve reranker probe env vars into a [`CandidateArtifacts`].
///
/// Returns `None` if this is not a probe invocation (primary model env var absent).
/// Returns `Some(Ok(candidate))` if all three vars are set.
/// Returns `Some(Err(3))` if the model var is set but config or tokenizer is missing.
pub(crate) fn probe_env_to_paths(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<CandidateArtifacts, i32>> {
    resolve_probe_env(model, config, tokenizer)
        .map(|r| r.map(|(m, c, t)| CandidateArtifacts::from_paths(m, c, t)))
}

fn probe_via_subprocess(artifacts: &Artifacts) -> Result<ProbeStatus, RerankerInitError> {
    probe_paths_via_subprocess(
        &artifacts.paths,
        RERANKER_PROBE_ENV_MODEL,
        RERANKER_PROBE_ENV_CONFIG,
        RERANKER_PROBE_ENV_TOKENIZER,
    )
    .map_err(Into::into)
}
