mod mlx;
mod probe;

mod pooling;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

use self::mlx::EmbedderInner;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{
    AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockChunkedEmbedder, MockEmbedder,
};

pub(crate) use pooling::postprocess_embedding;
pub use probe::handle_probe_if_needed;

use std::path::PathBuf;
use std::sync::Mutex;

/// Output vector dimensionality (768-d for ruri-v3-310m).
pub const EMBEDDING_DIMS: u32 = 768;
/// Prefix prepended to query text before tokenization.
pub const QUERY_PREFIX: &str = "検索クエリ: ";
/// Prefix prepended to document text before tokenization.
pub const DOCUMENT_PREFIX: &str = "検索文書: ";

/// Maximum sequence length for the model (ruri-v3 max_position_embeddings).
pub const MAX_SEQ_LEN: usize = 8192;
/// Number of overlapping tokens between adjacent chunks.
pub const CHUNK_OVERLAP_TOKENS: usize = 2048;
/// BOS (beginning of sequence) token ID for ruri-v3.
#[cfg(test)]
const BOS_TOKEN_ID: u32 = 1;
/// EOS (end of sequence) token ID for ruri-v3.
pub(crate) const EOS_TOKEN_ID: u32 = 2;

/// Maximum text content tokens for a given prefix length.
/// Computed as `MAX_SEQ_LEN - 2 (BOS + EOS) - prefix_len`.
pub(crate) const fn max_content(prefix_len: usize) -> usize {
    MAX_SEQ_LEN - 2 - prefix_len
}

/// Embedding result for a single text, potentially split into multiple chunks.
///
/// Each chunk is a 768-dimensional embedding vector. Short texts produce a
/// single chunk; texts exceeding [`MAX_SEQ_LEN`] produce multiple overlapping chunks.
#[derive(Debug, Clone)]
pub struct ChunkedEmbedding {
    /// One embedding vector per chunk. Always non-empty.
    pub chunks: Vec<Vec<f32>>,
}

/// Compute chunk start positions within the text token array.
///
/// Returns a list of start indices. Each chunk reads `max_content` tokens
/// starting from the given position. The last chunk's start is adjusted
/// so that it ends exactly at the end of the text tokens (stretch-to-fill).
///
/// For texts that fit in a single chunk, returns `[0]`.
#[cfg(test)]
pub(crate) fn compute_chunk_starts(
    text_token_count: usize,
    max_content: usize,
    overlap: usize,
) -> Vec<usize> {
    if text_token_count <= max_content {
        return vec![0];
    }

    let stride = max_content.saturating_sub(overlap);
    if stride == 0 {
        return vec![0];
    }

    let mut starts = Vec::new();
    let mut pos = 0;
    while pos + max_content < text_token_count {
        starts.push(pos);
        pos += stride;
    }
    // Last chunk: stretch to fill max_content
    starts.push(text_token_count - max_content);
    starts
}

/// Build token chunks from text tokens, adding BOS, prefix, and EOS to each.
///
/// Each chunk has the structure: `[BOS, prefix..., text_slice..., EOS]`.
/// The text slice length is at most `max_content` tokens.
#[cfg(test)]
pub(crate) fn build_token_chunks(
    text_tokens: &[u32],
    prefix_tokens: &[u32],
    starts: &[usize],
    max_content: usize,
) -> Vec<Vec<u32>> {
    starts
        .iter()
        .map(|&start| {
            let end = (start + max_content).min(text_tokens.len());
            let mut chunk = Vec::with_capacity(2 + prefix_tokens.len() + (end - start));
            chunk.push(BOS_TOKEN_ID);
            chunk.extend_from_slice(prefix_tokens);
            chunk.extend_from_slice(&text_tokens[start..end]);
            chunk.push(EOS_TOKEN_ID);
            chunk
        })
        .collect()
}

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
/// If the sequence already fits, returns a clone. Otherwise, takes the first
/// `max_len - 1` tokens and replaces the last with EOS. Logs a warning on truncation.
pub(crate) fn truncate_for_query(
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    max_len: usize,
) -> (Vec<u32>, Vec<u32>, usize) {
    let len = input_ids.len();
    if len <= max_len {
        return (input_ids, attention_mask, len);
    }
    log::warn!("query exceeds max_seq_len ({len} > {max_len}), truncating");
    let mut ids = input_ids[..max_len].to_vec();
    ids[max_len - 1] = EOS_TOKEN_ID;
    let mask = attention_mask[..max_len].to_vec();
    (ids, mask, max_len)
}

const MODEL_REPO: &str = "cl-nagoya/ruri-v3-310m";
const MODEL_REVISION: &str = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3";

/// Errors from embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Model weights file not found.
    #[error("model not found at {path}")]
    ModelNotFound {
        /// Path that was looked up.
        path: PathBuf,
    },
    /// Output tensor shape does not match expected dimensions.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension count.
        expected: usize,
        /// Actual dimension count.
        actual: usize,
    },
    /// Config file missing or unparseable.
    #[error("config error at {path}: {reason}")]
    Config {
        /// Config file path.
        path: PathBuf,
        /// Parse or IO failure detail.
        reason: String,
    },
    /// MLX inference failure.
    #[error("inference error: {0}")]
    Inference(String),
    /// Tokenizer load or encode failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Model download failure.
    #[error("download failed: {0}")]
    Download(String),
    /// Weights loaded but model is corrupt or incompatible.
    #[error("model load failed: {reason}")]
    ModelCorrupt {
        /// Failure detail from the backend.
        reason: String,
    },
}

impl EmbedError {
    pub(crate) fn config(path: &std::path::Path, e: impl std::fmt::Display) -> Self {
        Self::Config {
            path: path.to_path_buf(),
            reason: e.to_string(),
        }
    }

    pub(crate) fn inference(e: impl std::fmt::Display) -> Self {
        Self::Inference(e.to_string())
    }

    pub(crate) fn tokenizer(e: impl std::fmt::Display) -> Self {
        Self::Tokenizer(e.to_string())
    }

    pub(crate) fn download(e: impl std::fmt::Display) -> Self {
        Self::Download(e.to_string())
    }
}

/// Result of [`Embedder::probe`] — whether the MLX backend can load the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Model loaded successfully.
    Available,
    /// MLX backend crashed or is unsupported on this hardware.
    BackendUnavailable,
}

/// Embedding provider. Returns [`EMBEDDING_DIMS`]-dimensional f32 vectors.
///
/// Thread-safe: uses `&self` so implementors can be shared via `Arc<dyn Embed>`.
/// Implementors that hold mutable state should use interior mutability (e.g. `Mutex`).
///
/// # Contract
/// Implementations MUST return vectors of exactly [`EMBEDDING_DIMS`] elements.
pub trait Embed: Send + Sync {
    /// Embed a search query (prepends [`QUERY_PREFIX`]).
    /// Queries are truncated (not chunked) if they exceed [`MAX_SEQ_LEN`].
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    /// Embed a document for indexing (prepends [`DOCUMENT_PREFIX`]).
    /// Long documents are split into overlapping chunks.
    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError>;
    /// Batch-embed documents. Default calls [`embed_document`](Self::embed_document) per item.
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }
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
    pub fn from_dir(dir: &std::path::Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }

    /// Check that all three files exist, returning [`EmbedError::ModelNotFound`] on the first miss.
    pub fn validate(&self) -> Result<(), EmbedError> {
        for path in [&self.model, &self.config, &self.tokenizer] {
            if !path.exists() {
                return Err(EmbedError::ModelNotFound { path: path.clone() });
            }
        }
        Ok(())
    }
}

/// Download model files from Hugging Face Hub (cached after first download).
pub fn download_model() -> Result<ModelPaths, EmbedError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| EmbedError::download(format!("HF Hub init failed: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        MODEL_REPO.to_string(),
        hf_hub::RepoType::Model,
        MODEL_REVISION.to_string(),
    ));

    let get = |name: &str| {
        repo.get(name)
            .map_err(|e| EmbedError::download(format!("{name} download failed: {e}")))
    };

    let model = get("model.safetensors")?;
    let config = get("config.json")?;
    let tokenizer = get("tokenizer.json")?;

    Ok(ModelPaths {
        model,
        config,
        tokenizer,
    })
}

/// Check whether model files exist in the local HF Hub cache.
///
/// Returns `Ok(Some(paths))` if all three files are cached, `Ok(None)` otherwise.
/// Never accesses the network.
pub fn model_paths_if_cached() -> Result<Option<ModelPaths>, EmbedError> {
    model_paths_from_cache(&hf_hub::Cache::from_env())
}

fn model_paths_from_cache(cache: &hf_hub::Cache) -> Result<Option<ModelPaths>, EmbedError> {
    let repo = cache.repo(hf_hub::Repo::with_revision(
        MODEL_REPO.to_string(),
        hf_hub::RepoType::Model,
        MODEL_REVISION.to_string(),
    ));
    let Some(model) = repo.get("model.safetensors") else {
        return Ok(None);
    };
    let Some(config) = repo.get("config.json") else {
        return Ok(None);
    };
    let Some(tokenizer) = repo.get("tokenizer.json") else {
        return Ok(None);
    };
    Ok(Some(ModelPaths {
        model,
        config,
        tokenizer,
    }))
}

/// Deserialize a JSON config file into `T`.
///
/// Returns [`EmbedError::Config`] on IO failure or JSON parse error.
pub fn read_config<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
) -> Result<T, EmbedError> {
    let text = std::fs::read_to_string(path).map_err(|e| EmbedError::config(path, e))?;
    serde_json::from_str(&text).map_err(|e| EmbedError::config(path, format!("parse error: {e}")))
}

/// Load a HuggingFace tokenizer from a JSON file.
pub fn load_tokenizer(path: &std::path::Path) -> Result<tokenizers::Tokenizer, EmbedError> {
    tokenizers::Tokenizer::from_file(path).map_err(EmbedError::tokenizer)
}

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
pub fn tokenize_with_prefix(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    prefix: &str,
) -> Result<TokenizedInput, EmbedError> {
    let prefixed = format!("{prefix}{text}");
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

/// Thread-safe embedding model backed by MLX. Wraps the backend in a [`Mutex`].
pub struct Embedder {
    inner: Mutex<EmbedderInner>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    /// Load model weights, config, and tokenizer from `paths`.
    ///
    /// # Single-instance expectation
    ///
    /// `mlx_clear_cache()` operates on process-global Metal allocator state.
    /// Concurrent calls are serialized by an internal lock, but holding
    /// multiple `Embedder` instances doubles GPU memory usage (~600 MB).
    /// Prefer one `Embedder` per process.
    pub fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        let inner = EmbedderInner::new(paths)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Test whether the model can load without aborting the caller.
    ///
    /// Re-execs the current binary as a probe subprocess (via `Command`), so a
    /// crash is contained and reported as [`ProbeStatus::BackendUnavailable`].
    ///
    /// The host binary must call [`handle_probe_if_needed`] at the start of `main()`.
    pub fn probe(paths: &ModelPaths) -> Result<ProbeStatus, EmbedError> {
        paths.validate()?;
        let config: crate::modernbert::Config = read_config(&paths.config)?;
        config.validate().map_err(EmbedError::inference)?;
        let _ = load_tokenizer(&paths.tokenizer)?;
        probe::probe_via_subprocess(paths)
    }

    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, EmbedderInner>, EmbedError> {
        self.inner
            .lock()
            .map_err(|_| EmbedError::inference("embedder lock poisoned"))
    }
}

impl Embed for Embedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.lock_inner()?.embed_query_truncated(text, QUERY_PREFIX)
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        self.lock_inner()?.embed_document_chunked(text)
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.lock_inner()?.embed_documents_batch_chunked(texts)
    }
}
