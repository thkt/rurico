use std::fmt::{self, Debug, Formatter};
use std::sync::{Mutex, MutexGuard};

use super::metrics::BatchMetrics;
use super::mlx::EmbedderInner;
use super::{
    Artifacts, ChunkedEmbedding, Embed, EmbedError, EmbedInitError, ProbeStatus, QUERY_PREFIX,
};

/// Thread-safe embedding model backed by MLX. Wraps the backend in a [`Mutex`].
pub struct Embedder {
    inner: Mutex<EmbedderInner>,
    embedding_dims: usize,
}

impl Debug for Embedder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    /// Load model weights, config, and tokenizer from verified `artifacts`.
    ///
    /// # Single-instance expectation
    ///
    /// `mlx_clear_cache()` operates on process-global Metal allocator state.
    /// Concurrent calls are serialized by an internal lock, but holding
    /// multiple `Embedder` instances doubles GPU memory usage (~600 MB).
    /// Prefer one `Embedder` per process.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedInitError::Backend`] if MLX model construction or weight loading fails.
    pub fn new(artifacts: &Artifacts) -> Result<Self, EmbedInitError> {
        let inner = EmbedderInner::new(artifacts)?;
        let embedding_dims = inner.embedding_dims();
        Ok(Self {
            inner: Mutex::new(inner),
            embedding_dims,
        })
    }

    /// Embedding vector length for this model (read from `config.json.hidden_size`).
    ///
    /// All vectors returned by [`embed_query`](Embed::embed_query),
    /// [`embed_document`](Embed::embed_document), and
    /// [`embed_text`](Embed::embed_text) have this length.
    pub fn embedding_dims(&self) -> usize {
        self.embedding_dims
    }

    /// Batch-embed documents and return a [`BatchMetrics`] snapshot alongside
    /// the embeddings.
    ///
    /// Equivalent to [`embed_documents_batch`](Embed::embed_documents_batch),
    /// but exposes padding ratio, real/padded token counts, per-phase timings,
    /// and the length-bucket histogram that would otherwise only surface in the
    /// debug log. Intended for the Phase 2 smoke harness (SLA + R² linearity
    /// assertions) and for downstream consumers that want structured
    /// observability.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError`] on tokenization, inference, or post-processing
    /// failure, matching [`embed_documents_batch`](Embed::embed_documents_batch).
    pub fn embed_documents_batch_with_metrics(
        &self,
        texts: &[&str],
    ) -> Result<(Vec<ChunkedEmbedding>, BatchMetrics), EmbedError> {
        self.lock_inner()?
            .embed_documents_batch_chunked_with_metrics(texts)
    }

    /// Test whether the model can load without aborting the caller.
    ///
    /// Re-execs the current binary as a probe subprocess (via `Command`), so a
    /// crash is contained and reported as [`ProbeStatus::BackendUnavailable`].
    ///
    /// The host binary must call [`crate::model_probe::handle_probe_if_needed`] at the start of `main()`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`EmbedInitError::Backend`] if the probe subprocess cannot be spawned
    ///   or the probe handler is not installed in the host binary
    /// - [`EmbedInitError::ModelCorrupt`] if the subprocess exits non-zero after
    ///   starting successfully
    pub fn probe(artifacts: &Artifacts) -> Result<ProbeStatus, EmbedInitError> {
        super::probe::probe_via_subprocess(artifacts)
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, EmbedderInner>, EmbedError> {
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

    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.lock_inner()?.embed_query_truncated(text, prefix)
    }
}
