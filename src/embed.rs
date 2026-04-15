mod embedder;
mod mlx;
mod probe;

mod pooling;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{
    AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockChunkedEmbedder, MockEmbedder,
};

pub use crate::artifacts::{ArtifactError, EmbedKind, VerifiedArtifacts};
pub use crate::model_probe::{ProbeStatus, handle_probe_if_needed};
pub use embedder::Embedder;
pub(crate) use pooling::postprocess_embedding;
pub(crate) use probe::probe_env_to_paths;

use crate::artifacts::verify_as_embed;
use crate::model_io::{
    ModelArtifact, ModelPaths, artifacts_if_cached, download_artifacts, truncate_with_eos,
};
use crate::model_probe::ProbeError;
use std::fmt;
#[cfg(any(test, feature = "test-support"))]
use std::path::Path;
use std::path::PathBuf;

// ── Domain alias ─────────────────────────────────────────────────────────────

/// Verified embedding model artifacts.
///
/// Produced by [`CandidateArtifacts::verify`] or [`download_model`].
/// Guarantees that all three artifact files exist, that `config.json` parses
/// as a valid [`crate::modernbert::Config`], that `tokenizer.json` loads
/// without error, and that the weights are for an embed model (not a reranker).
pub type Artifacts = VerifiedArtifacts<EmbedKind>;

// ── CandidateArtifacts ───────────────────────────────────────────────────────

/// Unverified embedding model artifact paths.
///
/// Construct with [`from_paths`](Self::from_paths) (or [`from_dir`](Self::from_dir)
/// in test contexts), then call [`verify`](Self::verify) to obtain
/// [`Artifacts`] that can be passed to [`Embedder::new`].
pub struct CandidateArtifacts {
    paths: ModelPaths,
}

impl CandidateArtifacts {
    /// Construct from explicit file paths without any verification.
    ///
    /// Use this when you have raw paths (e.g. from environment variables).
    /// Call [`verify`](Self::verify) before passing the result to [`Embedder::new`].
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

    /// Verify file existence, config integrity, tokenizer validity, and embed model kind.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for any of the following:
    /// - A required file is missing ([`ArtifactError::MissingFile`])
    /// - `config.json` cannot be parsed ([`ArtifactError::InvalidConfig`])
    /// - `tokenizer.json` cannot be loaded ([`ArtifactError::InvalidTokenizer`])
    /// - Weights are for a reranker, not an embed model ([`ArtifactError::WrongModelKind`])
    pub fn verify(self) -> Result<Artifacts, ArtifactError> {
        verify_as_embed(self.paths)
    }
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Output vector dimensionality for the default model (`cl-nagoya/ruri-v3-310m`).
///
/// Other model variants have different dimensions; use [`Embedder::embedding_dims`]
/// to query the actual dimension of a loaded model at runtime.
pub const EMBEDDING_DIMS: usize = 768;

/// Prefix for encoding semantic meaning (empty string).
pub const SEMANTIC_PREFIX: &str = "";

/// Prefix for classification and clustering tasks.
pub const TOPIC_PREFIX: &str = "トピック: ";

/// Prefix for retrieval queries.
pub const QUERY_PREFIX: &str = "検索クエリ: ";

/// Prefix for retrieval documents.
pub const DOCUMENT_PREFIX: &str = "検索文書: ";

pub use crate::model_io::MAX_SEQ_LEN;
/// Number of overlapping tokens between adjacent chunks.
pub const CHUNK_OVERLAP_TOKENS: usize = 2048;

/// Maximum text content tokens for a given prefix length.
/// Computed as `MAX_SEQ_LEN - 2 (BOS + EOS) - prefix_len`.
pub(crate) const fn max_content(prefix_len: usize) -> usize {
    MAX_SEQ_LEN - 2 - prefix_len
}

// ── ChunkedEmbedding ─────────────────────────────────────────────────────────

/// Embedding result for a single text, potentially split into multiple chunks.
///
/// Each chunk is an embedding vector whose length equals the loaded model's
/// `hidden_size`. Short texts produce a single chunk; texts exceeding
/// [`MAX_SEQ_LEN`] produce multiple overlapping chunks.
#[derive(Debug, Clone)]
pub struct ChunkedEmbedding {
    /// One embedding vector per chunk. Always non-empty.
    pub chunks: Vec<Vec<f32>>,
}

// ── Token helpers ─────────────────────────────────────────────────────────────

/// Extract prefix tokens by encoding the prefix string without special tokens.
pub(crate) fn extract_prefix_tokens(
    tokenizer: &tokenizers::Tokenizer,
    prefix: &str,
) -> Result<Vec<u32>, EmbedError> {
    let encoding = tokenizer
        .encode(prefix, false)
        .map_err(EmbedError::tokenizer)?;
    Ok(encoding.get_ids().to_vec())
}

/// Truncate a query token sequence to fit within `max_len`, preserving BOS and EOS.
///
/// If the sequence already fits, returns the inputs unchanged. Otherwise,
/// truncates to the first `max_len - 1` tokens and replaces the last with EOS.
/// Logs a warning on truncation.
///
/// # Precondition
///
/// `max_len` must be ≥ 1. A zero `max_len` returns the inputs unchanged.
pub(crate) fn truncate_for_query(
    mut input_ids: Vec<u32>,
    mut attention_mask: Vec<u32>,
    max_len: usize,
) -> (Vec<u32>, Vec<u32>, usize) {
    let orig_len = input_ids.len();
    if truncate_with_eos(&mut input_ids, &mut attention_mask, max_len) {
        log::warn!("query exceeds max_seq_len ({orig_len} > {max_len}), truncating");
    }
    let len = input_ids.len();
    (input_ids, attention_mask, len)
}

// ── ModelId ───────────────────────────────────────────────────────────────────

/// Identifies a ruri-v3 embedding model variant.
///
/// All variants share the same tokenizer, prefix scheme, and max sequence length (8192).
/// The embedding dimension varies per model and is read from `config.json.hidden_size`
/// at load time — use [`Embedder::embedding_dims`] to query the actual dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelId {
    /// `cl-nagoya/ruri-v3-30m` — 256-dimensional.
    RuriV3_30m,
    /// `cl-nagoya/ruri-v3-70m` — 384-dimensional.
    RuriV3_70m,
    /// `cl-nagoya/ruri-v3-130m` — 512-dimensional.
    RuriV3_130m,
    /// `cl-nagoya/ruri-v3-310m` — 768-dimensional (default).
    #[default]
    RuriV3_310m,
}

impl ModelId {
    /// HuggingFace repository ID for this model.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::RuriV3_30m => "cl-nagoya/ruri-v3-30m",
            Self::RuriV3_70m => "cl-nagoya/ruri-v3-70m",
            Self::RuriV3_130m => "cl-nagoya/ruri-v3-130m",
            Self::RuriV3_310m => "cl-nagoya/ruri-v3-310m",
        }
    }

    fn revision(self) -> &'static str {
        match self {
            Self::RuriV3_30m => "24899e5de370b56d179604a007c0d727bf144504",
            Self::RuriV3_70m => "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc",
            Self::RuriV3_130m => "e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6",
            Self::RuriV3_310m => "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
        }
    }
}

impl ModelArtifact for ModelId {
    fn repo_id(self) -> &'static str {
        ModelId::repo_id(self)
    }

    fn revision(self) -> &'static str {
        self.revision()
    }
}

// ── EmbedInitError ────────────────────────────────────────────────────────────

/// Errors from initialising the embedding backend ([`Embedder::new`], [`Embedder::probe`]).
///
/// These errors occur after artifact verification has already succeeded. They
/// indicate a failure during MLX backend setup or model weight loading.
#[derive(Debug, thiserror::Error)]
pub enum EmbedInitError {
    /// MLX backend initialisation, weight loading, or subprocess probe failure.
    #[error("embed init failed: {0}")]
    Backend(String),
    /// Model weights loaded but are corrupt or incompatible with the expected architecture.
    #[error("model load failed: {reason}")]
    ModelCorrupt {
        /// Failure detail from the backend.
        reason: String,
    },
}

impl EmbedInitError {
    pub(crate) fn backend(e: impl fmt::Display) -> Self {
        Self::Backend(e.to_string())
    }
}

impl From<ProbeError> for EmbedInitError {
    fn from(e: ProbeError) -> Self {
        match e {
            ProbeError::HandlerNotInstalled => EmbedInitError::Backend(e.to_string()),
            ProbeError::ModelLoadFailed { reason } => EmbedInitError::ModelCorrupt { reason },
            ProbeError::SubprocessFailed(msg) => EmbedInitError::Backend(msg),
        }
    }
}

// ── EmbedError (runtime) ──────────────────────────────────────────────────────

/// Errors from embedding operations at runtime.
///
/// These errors occur during calls to [`Embed`] trait methods after the
/// [`Embedder`] has been successfully initialised.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Model returned an empty sequence (seq_len is 0).
    #[error("empty sequence: seq_len is 0")]
    EmptySequence,
    /// Flat output buffer size is inconsistent with expected batch shape.
    #[error("buffer shape mismatch: expected {expected}, got {actual}")]
    BufferShapeMismatch {
        /// Expected total elements.
        expected: usize,
        /// Actual total elements.
        actual: usize,
    },
    /// MLX inference failure during a forward pass.
    #[error("inference error: {0}")]
    Inference(String),
    /// Tokenizer encode failure (e.g. unsupported character sequence).
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Embedding output contains non-finite values (NaN or infinity).
    #[error("non-finite values in embedding output (NaN or inf)")]
    NonFiniteOutput,
}

impl EmbedError {
    pub(crate) fn inference(e: impl fmt::Display) -> Self {
        Self::Inference(e.to_string())
    }

    pub(crate) fn tokenizer(e: impl fmt::Display) -> Self {
        Self::Tokenizer(e.to_string())
    }
}

// ── Embed trait ───────────────────────────────────────────────────────────────

/// Embedding provider.
///
/// Thread-safe: uses `&self` so implementors can be shared via `Arc<dyn Embed>`.
/// Implementors that hold mutable state should use interior mutability (e.g. `Mutex`).
///
/// # Contract
/// All vectors returned by a single implementor MUST have the same length,
/// determined by the model's `hidden_size`.
pub trait Embed: Send + Sync {
    /// Embed a search query (prepends [`QUERY_PREFIX`]).
    /// Queries are truncated (not chunked) if they exceed [`MAX_SEQ_LEN`].
    ///
    /// # Errors
    ///
    /// Implementations should return [`EmbedError`] for operational failures
    /// such as tokenization, inference, or output post-processing.
    /// Callers should not rely on exact error message text.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed a document for indexing (prepends [`DOCUMENT_PREFIX`]).
    /// Long documents are split into overlapping chunks.
    ///
    /// # Errors
    ///
    /// Implementations should return [`EmbedError`] for operational failures
    /// such as tokenization, inference, or output post-processing.
    /// Callers should not rely on exact error message text.
    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError>;

    /// Batch-embed documents. Returns one [`ChunkedEmbedding`] per input text, in the
    /// same order as `texts`. Default calls [`embed_document`](Self::embed_document) per item.
    ///
    /// # Errors
    ///
    /// The default implementation returns the first error produced by
    /// [`embed_document`](Self::embed_document).
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }

    /// Embed text using the specified prefix (prepended before tokenization).
    /// The text is truncated if it exceeds [`MAX_SEQ_LEN`] (no chunking).
    ///
    /// Use this for explicit prefix control (Ruri v3 1+3 scheme). Available prefixes:
    /// [`SEMANTIC_PREFIX`], [`TOPIC_PREFIX`], [`QUERY_PREFIX`], [`DOCUMENT_PREFIX`].
    /// For the common retrieval use case, prefer
    /// [`embed_query`](Self::embed_query) and [`embed_document`](Self::embed_document).
    ///
    /// # Errors
    ///
    /// Implementations should return [`EmbedError`] for operational failures
    /// such as tokenization, inference, or output post-processing.
    /// Callers should not rely on exact error message text.
    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError>;
}

// ── Public API: download / cache ──────────────────────────────────────────────

/// Download model files from Hugging Face Hub and verify them as embed artifacts.
///
/// # Errors
///
/// Returns [`ArtifactError::DownloadFailed`] if the Hugging Face client cannot be
/// initialised or any required artifact download fails.
/// Returns other [`ArtifactError`] variants if verification of the downloaded
/// files fails.
pub fn download_model(model: ModelId) -> Result<Artifacts, ArtifactError> {
    let paths = download_artifacts(model)?;
    verify_as_embed(paths)
}

/// Check whether embed model files exist in the local HF Hub cache and verify them.
///
/// Returns `Ok(Some(artifacts))` if all three files are cached and pass
/// verification, `Ok(None)` otherwise. Never accesses the network.
///
/// # Errors
///
/// Returns [`ArtifactError`] if cached files fail verification.
/// Cache misses are reported as `Ok(None)`.
pub fn cached_artifacts(model: ModelId) -> Result<Option<Artifacts>, ArtifactError> {
    let Some(paths) = artifacts_if_cached(model)? else {
        return Ok(None);
    };
    verify_as_embed(paths).map(Some)
}

// ── TokenizedInput ────────────────────────────────────────────────────────────

/// Output of [`tokenize_with_prefix`]: token IDs, attention mask, and sequence length.
pub struct TokenizedInput {
    /// Token IDs produced by the tokenizer.
    pub input_ids: Vec<u32>,
    /// 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u32>,
    /// Number of tokens (including special tokens).
    pub seq_len: usize,
}

/// Tokenize `text` with a prefix prepended (e.g. query/document prefix).
///
/// # Errors
///
/// Returns [`EmbedError::Tokenizer`] if encoding the prefixed text fails.
pub fn tokenize_with_prefix(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    prefix: &str,
) -> Result<TokenizedInput, EmbedError> {
    let mut prefixed = String::with_capacity(prefix.len() + text.len());
    prefixed.push_str(prefix);
    prefixed.push_str(text);
    let encoding = tokenizer
        .encode(prefixed, true)
        .map_err(EmbedError::tokenizer)?;
    let input_ids = encoding.get_ids().to_vec();
    let attention_mask = encoding.get_attention_mask().to_vec();
    let seq_len = input_ids.len();
    Ok(TokenizedInput {
        input_ids,
        attention_mask,
        seq_len,
    })
}
