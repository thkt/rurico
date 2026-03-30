mod mlx;

mod pooling;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

use self::mlx::EmbedderInner;

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockEmbedder};

pub(crate) use pooling::postprocess_embedding;

use std::path::PathBuf;
use std::sync::Mutex;

/// Output vector dimensionality (768-d for ruri-v3-310m).
pub const EMBEDDING_DIMS: u32 = 768;
/// Prefix prepended to query text before tokenization.
pub const QUERY_PREFIX: &str = "検索クエリ: ";
/// Prefix prepended to document text before tokenization.
pub const DOCUMENT_PREFIX: &str = "検索文書: ";

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
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    /// Embed a document for indexing (prepends [`DOCUMENT_PREFIX`]).
    fn embed_document(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    /// Batch-embed documents. Default calls [`embed_document`](Self::embed_document) per item.
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
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

/// Return indices into `texts` sorted by ascending byte length.
///
/// Shorter-first ordering minimizes padding waste in batched tokenization.
pub(crate) fn sort_indices_by_len(texts: &[&str]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_unstable_by_key(|&i| texts[i].len());
    indices
}

const PROBE_ENV_MODEL: &str = "__RURICO_PROBE_MODEL";
const PROBE_ENV_CONFIG: &str = "__RURICO_PROBE_CONFIG";
const PROBE_ENV_TOKENIZER: &str = "__RURICO_PROBE_TOKENIZER";
const PROBE_ACK: &str = "RURICO_PROBE_OK";

/// Resolve probe env vars into `ModelPaths`. Returns `None` if this is not a
/// probe invocation, `Some(Ok(paths))` if all vars are set, or `Some(Err(3))`
/// if the model var is set but config or tokenizer is missing.
fn probe_env_to_paths(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<ModelPaths, i32>> {
    let model = model?;
    Some(match (config, tokenizer) {
        (Some(config), Some(tokenizer)) => Ok(ModelPaths {
            model: model.into(),
            config: config.into(),
            tokenizer: tokenizer.into(),
        }),
        _ => Err(3),
    })
}

/// Call at the start of `main()` in binaries that use [`Embedder::probe`].
///
/// When the process was re-invoked as a probe subprocess, this function writes
/// a handshake token to stdout, attempts to load the model, and exits with
/// code 0 (success) or non-zero (failure). Otherwise it returns immediately.
pub fn handle_probe_if_needed() {
    let Some(result) = probe_env_to_paths(
        std::env::var(PROBE_ENV_MODEL).ok(),
        std::env::var(PROBE_ENV_CONFIG).ok(),
        std::env::var(PROBE_ENV_TOKENIZER).ok(),
    ) else {
        return;
    };

    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{PROBE_ACK}");
    let _ = std::io::stdout().flush();

    match result {
        Err(code) => std::process::exit(code),
        Ok(paths) => match Embedder::new(&paths) {
            Ok(_) => std::process::exit(0),
            Err(e) => {
                let _ = write!(std::io::stderr(), "{e}");
                std::process::exit(1);
            }
        },
    }
}

/// Re-exec the current binary as a probe subprocess.
///
/// Uses `Command` (fork+exec internally) so the child gets a clean process
/// state, avoiding the async-signal-safety issues of bare fork().
///
/// - Exit 0 → [`ProbeStatus::Available`]
/// - Exit non-zero → [`EmbedError::ModelCorrupt`] (reason from stderr)
/// - Killed by signal / no exit code → [`ProbeStatus::BackendUnavailable`]
fn probe_via_subprocess(paths: &ModelPaths) -> Result<ProbeStatus, EmbedError> {
    let exe = std::env::current_exe()
        .map_err(|e| EmbedError::inference(format!("cannot locate executable: {e}")))?;

    let output = std::process::Command::new(exe)
        .env(PROBE_ENV_MODEL, &paths.model)
        .env(PROBE_ENV_CONFIG, &paths.config)
        .env(PROBE_ENV_TOKENIZER, &paths.tokenizer)
        .output()
        .map_err(|e| EmbedError::inference(format!("probe spawn failed: {e}")))?;

    interpret_probe_output(&output)
}

fn interpret_probe_output(output: &std::process::Output) -> Result<ProbeStatus, EmbedError> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.starts_with(PROBE_ACK) {
        return Err(EmbedError::inference(
            "probe handler not installed; call rurico::embed::handle_probe_if_needed() \
             at the start of main()",
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if output.status.signal().is_some() {
            return Ok(ProbeStatus::BackendUnavailable);
        }
    }

    match output.status.code() {
        Some(0) => Ok(ProbeStatus::Available),
        Some(_) => {
            let reason = String::from_utf8_lossy(&output.stderr).trim().to_string();
            Err(EmbedError::ModelCorrupt {
                reason: if reason.is_empty() {
                    "model load failed".into()
                } else {
                    reason
                },
            })
        }
        None => Ok(ProbeStatus::BackendUnavailable),
    }
}

/// Fork-based probe for test use only. Tests cannot install `handle_probe_if_needed`
/// in the test harness main, so they use fork directly. Test binaries are single-
/// threaded at the probe call site, making this safe.
#[cfg(test)]
fn probe_via_fork(paths: &ModelPaths) -> Result<ProbeStatus, EmbedError> {
    let pid = unsafe { libc::fork() };
    match pid {
        -1 => Err(EmbedError::inference(std::io::Error::last_os_error())),
        0 => {
            let code = if EmbedderInner::new(paths).is_ok() {
                0
            } else {
                1
            };
            unsafe { libc::_exit(code) }
        }
        child => {
            let mut status: libc::c_int = 0;
            if unsafe { libc::waitpid(child, &mut status, 0) } == -1 {
                return Err(EmbedError::inference(std::io::Error::last_os_error()));
            }
            if libc::WIFEXITED(status) {
                match libc::WEXITSTATUS(status) {
                    0 => Ok(ProbeStatus::Available),
                    _ => Err(EmbedError::ModelCorrupt {
                        reason: "model load failed in child process".into(),
                    }),
                }
            } else {
                Ok(ProbeStatus::BackendUnavailable)
            }
        }
    }
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
        probe_via_subprocess(paths)
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
