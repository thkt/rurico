use std::fmt::{self, Debug, Formatter};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use super::metrics::BatchMetrics;
use super::mlx::EmbedderInner;
use super::{
    Artifacts, ChunkedEmbedding, Embed, EmbedError, EmbedOptions, InferenceMetrics,
    MeasuredEmbedding, ModelInitError, QUERY_PREFIX,
};
use crate::model_probe::ProbeStatus;

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
    /// Each instance serializes its calls with an internal lock. Multiple
    /// instances retain a separate set of model weights for each instance.
    /// Prefer one `Embedder` per process.
    ///
    /// # Errors
    ///
    /// Returns [`ModelInitError::Backend`] if MLX model construction or weight loading fails.
    pub fn new(artifacts: &Artifacts) -> Result<Self, ModelInitError> {
        let inner = EmbedderInner::new(artifacts)?;
        let embedding_dims = inner.embedding_dims();
        tracing::info!(
            embedding_dims,
            model_path = ?artifacts.paths.model,
            "embedder: model loaded"
        );
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
            .embed_documents_batch_chunked_with_metrics(texts, &EmbedOptions::default(), false)
            .map(|(results, metrics)| (results, BatchMetrics::from(&metrics)))
    }

    /// Test whether the model can load without aborting the caller.
    ///
    /// Re-execs the current binary as a probe subprocess (via `Command`), so a
    /// crash is contained and reported as [`ProbeStatus::BackendUnavailable`].
    ///
    /// The host binary must call [`crate::handle_probe_if_needed`] at the start of `main()`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`ModelInitError::Backend`] if the probe subprocess cannot be spawned
    ///   or the probe handler is not installed in the host binary
    /// - [`ModelInitError::ModelCorrupt`] if the subprocess exits non-zero after
    ///   starting successfully
    pub fn probe(artifacts: &Artifacts) -> Result<ProbeStatus, ModelInitError> {
        super::probe::probe_via_subprocess(artifacts)
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, EmbedderInner>, EmbedError> {
        self.inner.lock().map_err(|e| {
            tracing::error!(
                error = %e,
                "embedder: mutex poisoned (prior panic in critical section)"
            );
            EmbedError::inference_message("embedder lock poisoned")
        })
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

    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.lock_inner()?
            .embed_documents_batch_chunked_with_options(texts, options)
    }

    fn embed_documents_batch_with_options_and_metrics(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<MeasuredEmbedding, EmbedError> {
        let started = Instant::now();
        let mut inner = self.lock_inner()?;
        let lock_wait = started.elapsed();
        let (embeddings, phases) =
            inner.embed_documents_batch_chunked_with_metrics(texts, options, true)?;
        drop(inner);
        let mut metrics = InferenceMetrics::from(phases);
        metrics.lock_wait = lock_wait;
        metrics.wall = started.elapsed();
        Ok(MeasuredEmbedding {
            embeddings,
            metrics: Some(metrics),
        })
    }

    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.lock_inner()?.embed_query_truncated(text, prefix)
    }
}

#[cfg(all(test, feature = "smoke"))]
#[path = "../../docs/research/issue-317/entry.rs"]
mod query_spike;

// Fixed-model adapter: the measurement protocol itself is model-free and tested in host317.
#[cfg(all(test, feature = "smoke"))]
pub(crate) fn observe317(case: &str, candidate: bool, record: &mut dyn FnMut(&serde_json::Value)) {
    use crate::{
        embed::{ModelId, cached_artifacts, query317::QueryBatch},
        host317,
        research::verify_paths,
    };
    use mlx_rs::memory;
    use serde_json::json;
    let artifacts = cached_artifacts(ModelId::DEFAULT)
        .unwrap()
        .expect("fixed embed cache required");
    verify_paths(&artifacts.paths);
    let texts = host317::query_workload(case);
    let refs: Vec<_> = texts.iter().map(String::as_str).collect();
    let load = Instant::now();
    let model = Embedder::new(&artifacts).unwrap();
    record(
        &json!({"event":"load","ns":load.elapsed().as_nanos(),"repository":ModelId::DEFAULT.repo_id(),"revision":ModelId::DEFAULT.revision()}),
    );
    let legacy = host317::Legacy(&model);
    host317::observe_queries(
        &artifacts.tokenizer,
        &refs,
        candidate,
        model.embedding_dims(),
        |batch, inputs| {
            let provider: &dyn QueryBatch = if batch { &model } else { &legacy };
            provider.queries(inputs)
        },
        (&mut || memory::reset_peak_memory().unwrap(), &mut || {
            memory::peak_memory().unwrap()
        }),
        record,
    );
}

#[cfg(all(test, feature = "smoke"))]
impl super::query317::QueryBatch for Embedder {
    fn queries(&self, texts: &[&str]) -> Result<super::query317::Queries, EmbedError> {
        query_spike::measured_entry(|| self.lock_inner(), |inner| inner.queries317(texts))
    }
}
