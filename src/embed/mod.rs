#[cfg(not(feature = "mlx"))]
compile_error!("enable the `mlx` feature");

#[cfg(feature = "mlx")]
mod mlx;

mod pooling;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

#[cfg(feature = "mlx")]
use self::mlx::EmbedderInner;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockEmbedder};

pub use pooling::{l2_normalize, mean_pooling, postprocess_embedding};

use std::path::PathBuf;
use std::sync::Mutex;

pub const EMBEDDING_DIMS: u32 = 768;
pub const QUERY_PREFIX: &str = "検索クエリ: ";
pub const DOCUMENT_PREFIX: &str = "検索文書: ";

const MODEL_REPO: &str = "cl-nagoya/ruri-v3-310m";
const MODEL_REVISION: &str = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3";

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding model not available")]
    ModelNotAvailable,
    #[error("model not found at {path}")]
    ModelNotFound { path: PathBuf },
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("inference error: {0}")]
    Inference(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("download failed: {0}")]
    Download(String),
}

impl EmbedError {
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

/// Embedding provider. Returns [`EMBEDDING_DIMS`]-dimensional f32 vectors.
///
/// Thread-safe: uses `&self` so implementors can be shared via `Arc<dyn Embed>`.
/// Implementors that hold mutable state should use interior mutability (e.g. `Mutex`).
///
/// # Contract
/// Implementations MUST return vectors of exactly [`EMBEDDING_DIMS`] elements.
pub trait Embed: Send + Sync {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_document(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_dir(dir: &std::path::Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }

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

pub fn read_config<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
) -> Result<T, EmbedError> {
    let text = std::fs::read_to_string(path).map_err(EmbedError::inference)?;
    serde_json::from_str(&text)
        .map_err(|e| EmbedError::inference(format!("config.json parse error: {e}")))
}

pub fn load_tokenizer(path: &std::path::Path) -> Result<tokenizers::Tokenizer, EmbedError> {
    tokenizers::Tokenizer::from_file(path).map_err(EmbedError::tokenizer)
}

pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub seq_len: usize,
}

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

/// Shorter texts first reduces wasted padding in batched tokenization.
pub(crate) fn sort_indices_by_char_count(texts: &[&str]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_unstable_by_key(|&i| texts[i].chars().count());
    indices
}

pub struct Embedder {
    inner: Mutex<EmbedderInner>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    pub fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        let inner = EmbedderInner::new(paths)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, EmbedderInner>, EmbedError> {
        self.inner
            .lock()
            .map_err(|_| EmbedError::inference("embedder lock poisoned"))
    }
}

impl Embed for Embedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.lock_inner()?.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_document(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.lock_inner()?.embed_with_prefix(text, DOCUMENT_PREFIX)
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.lock_inner()?.embed_batch(texts)
    }
}
