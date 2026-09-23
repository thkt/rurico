mod lazy;
mod mlx;
mod processing;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

pub use self::lazy::LazyReranker;
use self::mlx::RerankerInner;
use crate::artifacts;
use crate::model_io::ModelArtifact;
use crate::model_probe::{ProbeStatus, probe_paths_via_subprocess};
use std::error::Error;
use std::fmt::{self, Debug, Formatter};
use std::sync::{Arc, Mutex, MutexGuard};

pub use crate::artifacts::{ArtifactError, RerankerKind, VerifiedArtifacts};
pub use crate::model_init::ModelInitError;
pub use crate::model_lifecycle::{cached_artifacts, download_model};
#[cfg(any(test, feature = "test-support"))]
pub use test_support::MockReranker;

/// Probe env-var key for the reranker model weights path.
pub(crate) const PROBE_ENV_MODEL: &str = "__RURICO_RERANKER_PROBE_MODEL";
/// Probe env-var key for the reranker model config path.
pub(crate) const PROBE_ENV_CONFIG: &str = "__RURICO_RERANKER_PROBE_CONFIG";
/// Probe env-var key for the reranker model tokenizer path.
pub(crate) const PROBE_ENV_TOKENIZER: &str = "__RURICO_RERANKER_PROBE_TOKENIZER";

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
/// Type alias for [`crate::artifacts::CandidateArtifacts<RerankerKind>`] so the
/// downstream-visible name `reranker::CandidateArtifacts` keeps working while
/// the underlying definition lives in `artifacts.rs` (single source of truth).
pub type CandidateArtifacts = artifacts::CandidateArtifacts<RerankerKind>;

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
    type Kind = RerankerKind;

    fn repo_id(self) -> &'static str {
        RerankerModelId::repo_id(self)
    }

    fn revision(self) -> &'static str {
        self.revision()
    }
}

// ── RerankerError (runtime) ───────────────────────────────────────────────────

/// Errors from reranker operations at runtime.
///
/// These errors occur during calls to [`Rerank`] trait methods, including
/// initialization when the reranker is wrapped in [`LazyReranker`].
#[derive(Debug)]
#[non_exhaustive]
pub enum RerankerError {
    /// MLX forward/evaluation failure, output shape mismatch, or poisoned lock.
    Inference {
        /// Display rendering at construction.
        message: String,
        /// Originating error; absent for message-only failures such as shape mismatch.
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    /// Tokenizer encode failure (e.g. unsupported character sequence).
    Tokenizer {
        /// Display rendering at construction.
        message: String,
        /// Originating tokenizer error.
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    /// Model logit contains NaN or infinity, before sigmoid conversion.
    NonFiniteOutput,
    /// Lazy initialization failed (model load, cache lookup, or download).
    ///
    /// Returned by [`LazyReranker`] on the first method call when its init
    /// closure returns `Err`. Once observed, this failure is cached for the
    /// lifetime of the wrapper — see [`LazyReranker`] for rationale.
    InitFailed {
        /// Display rendering at construction.
        message: String,
        /// Shared cause, retained across calls without requiring the error to be Clone.
        source: Option<Arc<dyn Error + Send + Sync>>,
    },
}

impl fmt::Display for RerankerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference { message, .. } => write!(f, "inference error: {message}"),
            Self::Tokenizer { message, .. } => write!(f, "tokenizer error: {message}"),
            Self::InitFailed { message, .. } => write!(f, "init failed: {message}"),
            Self::NonFiniteOutput => {
                f.write_str("non-finite values in reranker output (NaN or inf)")
            }
        }
    }
}

impl Error for RerankerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Inference { source, .. } | Self::Tokenizer { source, .. } => {
                source.as_deref().map(|e| e as &(dyn Error + 'static))
            }
            // Expose the error inside Arc directly, so downcasting sees the
            // original init error rather than Arc's Error implementation.
            Self::InitFailed { source, .. } => {
                source.as_deref().map(|e| e as &(dyn Error + 'static))
            }
            Self::NonFiniteOutput => None,
        }
    }
}

impl RerankerError {
    pub(crate) fn inference(e: impl Error + Send + Sync + 'static) -> Self {
        Self::Inference {
            message: e.to_string(),
            source: Some(Box::new(e)),
        }
    }

    pub(crate) fn inference_message(message: impl fmt::Display) -> Self {
        Self::Inference {
            message: message.to_string(),
            source: None,
        }
    }

    pub(crate) fn tokenizer(e: tokenizers::Error) -> Self {
        Self::Tokenizer {
            message: e.to_string(),
            source: Some(e),
        }
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
    /// Finite extreme logits may round to 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns [`RerankerError`] on initialization, tokenization or inference failure,
    /// or [`RerankerError::NonFiniteOutput`] for a non-finite logit before sigmoid.
    /// Backend-generated messages are opaque and not stable API.
    fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError>;

    /// Score multiple (query, document) pairs in a single batched forward pass.
    ///
    /// Returns one score per pair in input order.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Rerank::score`]; one invalid logit fails the batch.
    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError>;

    /// Rerank documents by relevance to a query.
    ///
    /// Returns `Vec<RankedResult>` sorted by score descending. Ties on
    /// `score` are broken by ascending original input index, so repeated
    /// calls with identical scores produce identical ordering.
    /// This includes ties caused by f32 sigmoid saturation; raw logits are not sort keys.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Rerank::score_batch`].
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError>;
}

// ── Reranker ─────────────────────────────────────────────────────────────────

/// Thread-safe cross-encoder reranker backed by MLX.
///
/// Loads F32 weights (about 1.2 GB for the fixed 310m checkpoint).
/// Runtime peak memory also depends on input shape and MLX caches; holding an
/// [`Embedder`](crate::embed::Embedder) at the same time adds its own weights
/// and runtime allocations.
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
    /// Returns [`ModelInitError::Backend`] if MLX model construction or weight loading fails.
    pub fn new(artifacts: &Artifacts) -> Result<Self, ModelInitError> {
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
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Rerank::score`].
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
    ///
    /// # Errors
    ///
    /// Returns [`RerankerError`] for tokenizer, MLX or lock failures, shape mismatch,
    /// or non-finite logits before sigmoid. Backend messages are opaque.
    /// An empty input returns an empty vector without inference.
    pub fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        self.lock_inner()?.score_batch(pairs)
    }

    /// Rerank documents by relevance to a query.
    ///
    /// Returns `Vec<RankedResult>` sorted by score descending, with ties on
    /// `score` broken by ascending original input index. Each result contains
    /// the original index in `documents` and the relevance score.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::score_batch`].
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
    /// [`handle_probe_if_needed`](crate::handle_probe_if_needed)
    /// at the start of `main()`.
    pub fn probe(artifacts: &Artifacts) -> Result<ProbeStatus, ModelInitError> {
        probe_via_subprocess(artifacts)
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, RerankerInner>, RerankerError> {
        self.inner.lock().map_err(|e| {
            tracing::error!(
                error = %e,
                "reranker: mutex poisoned (prior panic in critical section)"
            );
            RerankerError::inference_message("reranker lock poisoned")
        })
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
    results.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.index.cmp(&b.index))
    });
    results
}

// ── Probe infrastructure ──────────────────────────────────────────────────────

fn probe_via_subprocess(artifacts: &Artifacts) -> Result<ProbeStatus, ModelInitError> {
    probe_paths_via_subprocess(
        &artifacts.paths,
        PROBE_ENV_MODEL,
        PROBE_ENV_CONFIG,
        PROBE_ENV_TOKENIZER,
    )
    .map_err(Into::into)
}
