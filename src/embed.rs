mod embedder;
/// Ordinary least squares linear regression. Internal support for the Phase 2
/// smoke harness (`mlx_smoke measure-baseline`'s R² assertion). Compiled only
/// when its sole caller (`mlx_smoke`, gated on the `smoke` feature) or the
/// crate's own test target is built.
#[cfg(any(test, feature = "smoke"))]
#[doc(hidden)]
pub mod linreg;
mod metrics;
mod mlx;
mod probe;
mod processing;

mod pooling;

pub mod fixtures;
pub mod workloads;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{
    AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockChunkedEmbedder, MockEmbedder,
};

pub use crate::artifacts::{ArtifactError, EmbedKind, VerifiedArtifacts};
pub use crate::model_init::ModelInitError;
pub use crate::model_lifecycle::{cached_artifacts, download_model};
pub use embedder::Embedder;
pub use metrics::{BatchMetrics, ForwardShape, InferenceMetrics, MeasuredEmbedding};
pub(crate) use pooling::gpu_pool_and_normalize;

use crate::artifacts;
use crate::model_io::{ModelArtifact, truncate_with_eos};
use std::time::Duration;
use std::{error::Error, fmt};

/// Probe env-var key for the embedding model weights path.
pub(crate) const PROBE_ENV_MODEL: &str = "__RURICO_PROBE_MODEL";
/// Probe env-var key for the embedding model config path.
pub(crate) const PROBE_ENV_CONFIG: &str = "__RURICO_PROBE_CONFIG";
/// Probe env-var key for the embedding model tokenizer path.
pub(crate) const PROBE_ENV_TOKENIZER: &str = "__RURICO_PROBE_TOKENIZER";

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
/// Type alias for [`crate::artifacts::CandidateArtifacts<EmbedKind>`] so the
/// downstream-visible name `embed::CandidateArtifacts` keeps working while the
/// underlying definition lives in `artifacts.rs` (single source of truth).
pub type CandidateArtifacts = artifacts::CandidateArtifacts<EmbedKind>;

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
pub(crate) const CHUNK_OVERLAP_TOKENS: usize = 2048;

/// Maximum text content tokens for a given prefix length.
/// Computed as `MAX_SEQ_LEN - 2 (BOS + EOS) - prefix_len`.
pub(crate) const fn max_content(prefix_len: usize) -> usize {
    MAX_SEQ_LEN - 2 - prefix_len
}

// ── ChunkedEmbedding ─────────────────────────────────────────────────────────

/// Embedding result for a single text, potentially split into multiple chunks.
///
/// The MLX producer returns vectors of the loaded model's `hidden_size`.
/// Short texts produce a single chunk; texts exceeding [`MAX_SEQ_LEN`]
/// produce multiple overlapping chunks.
///
/// `chunk_ids` carries a stable per-chunk identifier (`"c0"`, `"c1"`, …) used
/// by chunk-level retrieval (Issue #76 / ADR 0004 Stage 1) to keep child
/// chunks distinguishable through Stage 1 candidates and Stage 2 fusion.
/// [`Self::try_new_validated`] checks non-empty chunks, non-empty vectors,
/// uniform dimensions within this result, and finite elements.
/// The legacy [`Self::try_new`] only checks that `chunks` is non-empty.
/// Both constructors generate IDs so `chunk_ids.len() == chunks.len()`.
/// Use [`ChunkedEmbedding::chunks`] and [`ChunkedEmbedding::chunk_ids`] for
/// read access.
#[derive(Debug, Clone)]
pub struct ChunkedEmbedding {
    chunks: Vec<Vec<f32>>,
    chunk_ids: Vec<String>,
}

/// Error returned when constructing a [`ChunkedEmbedding`] without any chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("chunked embedding requires at least one chunk")]
pub struct EmptyChunksError;

/// Invalid content passed to [`ChunkedEmbedding::try_new_validated`].
///
/// Positions are zero-based. Chunks are checked in input order; within each
/// chunk, emptiness and dimension are checked before the first non-finite element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EmbeddingValidationError {
    /// No chunks were supplied (the same condition as the legacy constructor).
    #[error(transparent)]
    EmptyChunks(#[from] EmptyChunksError),
    /// A chunk has no vector elements.
    #[error("chunk {chunk} has an empty vector")]
    EmptyVector {
        /// Index of the empty chunk vector.
        chunk: usize,
    },
    /// A vector's dimension differs from the first chunk's dimension.
    #[error("chunk {chunk} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Index of the mismatched chunk.
        chunk: usize,
        /// Dimension of the first chunk.
        expected: usize,
        /// Dimension of this chunk.
        actual: usize,
    },
    /// A vector contains NaN or positive/negative infinity.
    #[error("chunk {chunk} element {element} is non-finite (NaN or inf)")]
    NonFiniteValue {
        /// Index of the chunk containing the non-finite value.
        chunk: usize,
        /// Index of the first non-finite element in that chunk.
        element: usize,
    },
}

// Shared by vector validation and MLX's already-readback flat-buffer check.
fn first_non_finite(values: &[f32]) -> Option<usize> {
    values.iter().position(|value| !value.is_finite())
}

fn validate_vector(
    vector: &[f32],
    chunk: usize,
    expected: usize,
) -> Result<(), EmbeddingValidationError> {
    if vector.is_empty() {
        return Err(EmbeddingValidationError::EmptyVector { chunk });
    }
    if vector.len() != expected {
        return Err(EmbeddingValidationError::DimensionMismatch {
            chunk,
            expected,
            actual: vector.len(),
        });
    }
    if let Some(element) = first_non_finite(vector) {
        return Err(EmbeddingValidationError::NonFiniteValue { chunk, element });
    }
    Ok(())
}

impl ChunkedEmbedding {
    /// Construct from chunk vectors with auto-generated chunk IDs (`"c0"`,
    /// `"c1"`, …). The default label scheme keeps producers (the MLX
    /// embedder, mocks, persistence loaders) consistent without forcing each
    /// caller to mint its own scheme.
    ///
    /// This legacy constructor does not validate vector content. Use
    /// [`Self::try_new_validated`] to check dimensions and finite elements.
    ///
    /// # Errors
    ///
    /// Returns [`EmptyChunksError`] when `chunks` is empty.
    pub fn try_new(chunks: Vec<Vec<f32>>) -> Result<Self, EmptyChunksError> {
        if chunks.is_empty() {
            return Err(EmptyChunksError);
        }
        Ok(Self::new_unchecked(chunks))
    }

    /// Construct from non-empty, equally sized vectors containing only finite
    /// elements. The chunk list must also be non-empty.
    ///
    /// Values, input order, and the `"c0"`, `"c1"`, … ID scheme are preserved.
    /// Zero vectors, non-unit lengths, negative values, and large finite values
    /// are accepted; this does not normalize or otherwise correct values.
    /// Only dimensions within this result are checked. The producer remains
    /// responsible for the cross-document/query/call contract of [`Embed`].
    /// This does not establish model identity or compatibility with an index.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingValidationError`] with the kind and position of the
    /// first invalid input. Empty chunks reuse the legacy [`EmptyChunksError`].
    ///
    /// ```
    /// use rurico::embed::{ChunkedEmbedding, EmbeddingValidationError};
    /// let result: Result<ChunkedEmbedding, EmbeddingValidationError> =
    ///     ChunkedEmbedding::try_new_validated(vec![vec![0.0, -2.0], vec![3.0, 4.0]]);
    /// let embedding = result?;
    /// assert_eq!(embedding.chunk_ids(), ["c0", "c1"]);
    /// # Ok::<(), EmbeddingValidationError>(())
    /// ```
    pub fn try_new_validated(chunks: Vec<Vec<f32>>) -> Result<Self, EmbeddingValidationError> {
        let embedding = Self::try_new(chunks)?;
        let expected = embedding.chunks[0].len();
        for (chunk, vector) in embedding.chunks.iter().enumerate() {
            validate_vector(vector, chunk, expected)?;
        }
        Ok(embedding)
    }

    /// Return the embedding vector for each chunk.
    pub fn chunks(&self) -> &[Vec<f32>] {
        &self.chunks
    }

    /// Return the stable identifier for each chunk.
    ///
    /// The returned slice has the same length and order as [`Self::chunks`].
    pub fn chunk_ids(&self) -> &[String] {
        &self.chunk_ids
    }

    /// Construct without checking non-emptiness.
    ///
    /// Use only after an upstream proof that at least one chunk exists.
    pub(crate) fn new_unchecked(chunks: Vec<Vec<f32>>) -> Self {
        let chunk_ids = (0..chunks.len()).map(|i| format!("c{i}")).collect();
        Self { chunks, chunk_ids }
    }
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
        tracing::warn!(orig_len, max_len, "query exceeds max_seq_len, truncating");
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelId {
    /// `cl-nagoya/ruri-v3-30m` — 256-dimensional.
    RuriV3_30m,
    /// `cl-nagoya/ruri-v3-70m` — 384-dimensional.
    RuriV3_70m,
    /// `cl-nagoya/ruri-v3-130m` — 512-dimensional.
    RuriV3_130m,
    /// `cl-nagoya/ruri-v3-310m` — 768-dimensional.
    RuriV3_310m,
}

impl ModelId {
    /// Product-chosen default embedding model.
    pub const DEFAULT: Self = Self::RuriV3_310m;

    /// Hugging Face repository used by the model lifecycle functions.
    pub fn repo_id(self) -> &'static str {
        <Self as ModelArtifact>::repo_id(self)
    }

    /// Pinned revision used for weights, config and tokenizer cache lookup.
    pub fn revision(self) -> &'static str {
        <Self as ModelArtifact>::revision(self)
    }
}

impl ModelArtifact for ModelId {
    type Kind = EmbedKind;

    fn repo_id(self) -> &'static str {
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

// ── EmbedInitError ────────────────────────────────────────────────────────────

// ── EmbedError (runtime) ──────────────────────────────────────────────────────

/// Errors from embedding operations at runtime.
///
/// These errors occur during calls to [`Embed`] trait methods after the
/// [`Embedder`] has been successfully initialised.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    /// A producer attempted to construct a chunked embedding with no chunks.
    #[error(transparent)]
    EmptyChunks(#[from] EmptyChunksError),
    /// A producer returned invalid vector content through the validated API.
    #[error(transparent)]
    InvalidEmbedding(EmbeddingValidationError),
    /// MLX inference failure during a forward pass.
    #[error("inference error: {message}")]
    Inference {
        /// Display rendering at construction.
        message: String,
        /// Source for chain walking.
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    /// Tokenizer encode failure (e.g. unsupported character sequence).
    #[error("tokenizer error: {message}")]
    Tokenizer {
        /// Display rendering at construction.
        message: String,
        /// Source for chain walking.
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    /// Embedding output contains non-finite values (NaN or infinity).
    #[error("non-finite values in embedding output (NaN or inf)")]
    NonFiniteOutput,
}

impl From<EmbeddingValidationError> for EmbedError {
    fn from(error: EmbeddingValidationError) -> Self {
        match error {
            EmbeddingValidationError::EmptyChunks(error) => Self::EmptyChunks(error),
            other => Self::InvalidEmbedding(other),
        }
    }
}

impl EmbedError {
    pub(crate) fn inference(e: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        let source = e.into();
        let message = source.to_string();
        Self::Inference {
            message,
            source: Some(source),
        }
    }

    pub(crate) fn inference_message(message: impl fmt::Display) -> Self {
        Self::Inference {
            message: message.to_string(),
            source: None,
        }
    }

    pub(crate) fn tokenizer(e: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        let source = e.into();
        let message = source.to_string();
        Self::Tokenizer {
            message,
            source: Some(source),
        }
    }
}

// ── EmbedOptions ──────────────────────────────────────────────────────────────

/// Best-effort performance hints for batch embedding.
///
/// Both fields default to `None`, which preserves the implementation's
/// built-in behavior. Implementations are free to ignore these hints; the
/// trait-level default of
/// [`embed_documents_batch_with_options`](Embed::embed_documents_batch_with_options)
/// does exactly that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EmbedOptions {
    /// Token budget per forward pass, overriding the built-in
    /// `TOKEN_BUDGET` used to size sub-batches. Smaller budgets shorten
    /// each GPU forward, trading throughput for host responsiveness.
    ///
    /// Values above the built-in budget are not clamped here and raise
    /// GPU out-of-memory risk; callers should clamp to the built-in
    /// budget or below.
    pub token_budget: Option<usize>,
    /// Sleep inserted after each completed forward pass, yielding the GPU
    /// to other processes (e.g. WindowServer) between forwards.
    pub forward_pause: Option<Duration>,
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
/// This applies across documents, queries, text prefixes, and calls, as described
/// in ADR-0011. [`ChunkedEmbedding::try_new_validated`] can validate non-empty,
/// finite vectors of equal dimension within one document, but cannot enforce
/// this cross-result producer responsibility or identify the model/index.
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
    /// same order as `texts`.
    ///
    /// Implementations must choose and encode their semantics explicitly. A
    /// per-document fallback may call [`embed_document`](Self::embed_document)
    /// per item, while accelerated implementations may use a fused forward pass.
    ///
    /// # Errors
    ///
    /// Implementations should return the first operational failure they
    /// encounter and preserve input ordering for successful outputs.
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError>;

    /// Batch-embed documents with best-effort performance hints.
    ///
    /// The default implementation ignores `options` and delegates to
    /// [`embed_documents_batch`](Self::embed_documents_batch), so results are
    /// identical either way; `options` only shapes *how* the work is
    /// scheduled (sub-batch sizing, inter-forward pauses) on implementations
    /// that honor it, such as [`Embedder`].
    ///
    /// # Errors
    ///
    /// Same contract as [`embed_documents_batch`](Self::embed_documents_batch).
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        _options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch(texts)
    }

    /// Embed once with best-effort options and return the result with available
    /// metrics. Works through `dyn Embed`. The compatibility default delegates
    /// exactly once to the existing options method and returns `metrics: None`.
    /// It does not require existing providers to honor hints or measure time.
    /// MLX returns `Some` even for empty input (no forwards, measured wall/lock).
    ///
    /// # Errors
    ///
    /// Propagates the same error as the options method without retrying inference.
    fn embed_documents_batch_with_options_and_metrics(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<MeasuredEmbedding, EmbedError> {
        self.embed_documents_batch_with_options(texts, options)
            .map(|embeddings| MeasuredEmbedding {
                embeddings,
                metrics: None,
            })
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

#[cfg(all(test, feature = "smoke"))]
pub(crate) mod research;
