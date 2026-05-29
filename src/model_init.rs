//! Unified model initialization error type.
//!
//! Carries the single `ModelInitError` enum used by both embed and reranker
//! constructors (`Embedder::new`/`Reranker::new`) and probe entry points. The
//! variant set is identical for both kinds; call sites distinguish kinds via
//! structured logging fields rather than via the error type.

use std::error::Error;

use crate::model_probe::ProbeError;

/// Errors from initialising a model backend (e.g. `Embedder::new` /
/// `Embedder::probe`, `Reranker::new` / `Reranker::probe`).
///
/// These errors occur after artifact verification has already succeeded. They
/// indicate a failure during MLX backend setup or model weight loading.
#[derive(Debug, thiserror::Error)]
pub enum ModelInitError {
    /// MLX backend initialisation, weight loading, or subprocess probe failure.
    ///
    /// `source` preserves the originating typed error so callers can walk the
    /// chain via `std::error::Error::source()`. `None` when the failure path
    /// carries only a `String` (e.g. `ProbeError::SubprocessFailed`) with no
    /// upstream typed error to box.
    #[error("model init failed: {message}")]
    Backend {
        /// Display rendering at construction.
        message: String,
        /// Source for chain walking.
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    /// Model weights loaded but are corrupt or incompatible with the expected architecture.
    #[error("model load failed: {reason}")]
    ModelCorrupt {
        /// Failure detail from the backend.
        reason: String,
    },
}

impl ModelInitError {
    pub(crate) fn backend(e: impl Error + Send + Sync + 'static) -> Self {
        Self::Backend {
            message: e.to_string(),
            source: Some(Box::new(e)),
        }
    }
}

impl From<ProbeError> for ModelInitError {
    fn from(e: ProbeError) -> Self {
        match e {
            ProbeError::ModelLoadFailed { reason } => ModelInitError::ModelCorrupt { reason },
            ProbeError::SubprocessFailed(msg) => ModelInitError::Backend {
                message: msg,
                source: None,
            },
            ProbeError::HandlerNotInstalled => ModelInitError::backend(e),
            ProbeError::SetupRejected { .. } => ModelInitError::backend(e),
        }
    }
}

#[cfg(test)]
mod tests;
