//! Unified model initialization error type.
//!
//! Carries the single `ModelInitError` enum used by both embed and reranker
//! constructors (`Embedder::new`/`Reranker::new`) and probe entry points. The
//! variant set is identical for both kinds; call sites distinguish kinds via
//! structured logging fields rather than via the error type.

use std::fmt::Display;

use crate::model_probe::ProbeError;

/// Errors from initialising a model backend (e.g. `Embedder::new` /
/// `Embedder::probe`, `Reranker::new` / `Reranker::probe`).
///
/// These errors occur after artifact verification has already succeeded. They
/// indicate a failure during MLX backend setup or model weight loading.
#[derive(Debug, thiserror::Error)]
pub enum ModelInitError {
    /// MLX backend initialisation, weight loading, or subprocess probe failure.
    #[error("model init failed: {0}")]
    Backend(String),
    /// Model weights loaded but are corrupt or incompatible with the expected architecture.
    #[error("model load failed: {reason}")]
    ModelCorrupt {
        /// Failure detail from the backend.
        reason: String,
    },
}

impl ModelInitError {
    #[allow(dead_code)] // wired up by Phase 1 Unit 1.2 (embed) / 1.3 (reranker)
    pub(crate) fn backend(e: impl Display) -> Self {
        Self::Backend(e.to_string())
    }
}

impl From<ProbeError> for ModelInitError {
    fn from(e: ProbeError) -> Self {
        match e {
            ProbeError::HandlerNotInstalled => ModelInitError::Backend(e.to_string()),
            ProbeError::ModelLoadFailed { reason } => ModelInitError::ModelCorrupt { reason },
            ProbeError::SetupRejected { .. } => ModelInitError::Backend(e.to_string()),
            ProbeError::SubprocessFailed(msg) => ModelInitError::Backend(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests reference items that do not exist yet; compile failure here is the
    // Red signal. Green phase (`/code` next pass) implements `ModelInitError`
    // and the `From<ProbeError>` mapping inside this module's parent scope.
    use crate::model_init::ModelInitError;
    use crate::model_probe::{PROBE_EXIT_PATH_OUTSIDE_CACHE, ProbeError};

    // T-006: From<ProbeError::HandlerNotInstalled> → Backend(s) where s == Display verbatim
    #[test]
    fn from_probe_error_handler_not_installed_maps_to_backend_with_verbatim_display() {
        let probe_err = ProbeError::HandlerNotInstalled;
        let display = probe_err.to_string();
        let init_err: ModelInitError = probe_err.into();
        let ModelInitError::Backend(s) = init_err else {
            panic!("expected ModelInitError::Backend, got: {init_err:?}");
        };
        assert_eq!(
            s, display,
            "Backend payload must equal ProbeError::HandlerNotInstalled Display output verbatim"
        );
    }

    // T-007: From<ProbeError::ModelLoadFailed { reason }> → ModelCorrupt { reason } (verbatim)
    #[test]
    fn from_probe_error_model_load_failed_maps_to_model_corrupt_verbatim_reason() {
        let probe_err = ProbeError::ModelLoadFailed { reason: "x".into() };
        let init_err: ModelInitError = probe_err.into();
        let ModelInitError::ModelCorrupt { reason } = init_err else {
            panic!("expected ModelInitError::ModelCorrupt, got: {init_err:?}");
        };
        assert_eq!(
            reason, "x",
            "ModelCorrupt.reason must preserve the source ProbeError::ModelLoadFailed reason verbatim"
        );
    }

    // T-008: From<ProbeError::SetupRejected { .. }> and other backend-failure variants → Backend(_)
    #[test]
    fn from_probe_error_setup_rejected_and_subprocess_failed_both_map_to_backend() {
        let setup_err: ModelInitError = ProbeError::SetupRejected {
            code: PROBE_EXIT_PATH_OUTSIDE_CACHE,
        }
        .into();
        assert!(
            matches!(setup_err, ModelInitError::Backend(_)),
            "ProbeError::SetupRejected must map to ModelInitError::Backend, got: {setup_err:?}"
        );

        let subprocess_err: ModelInitError = ProbeError::SubprocessFailed("y".into()).into();
        let ModelInitError::Backend(s) = subprocess_err else {
            panic!("expected ModelInitError::Backend for SubprocessFailed");
        };
        assert_eq!(
            s, "y",
            "SubprocessFailed payload must round-trip through Backend"
        );
    }
}
