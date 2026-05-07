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
mod tests {
    use std::error::Error as _;

    use crate::model_init::ModelInitError;
    use crate::model_probe::{ProbeError, SetupReason};

    // T-006: From<ProbeError::HandlerNotInstalled> → Backend { message } where message == Display verbatim
    #[test]
    fn from_probe_error_handler_not_installed_maps_to_backend_with_verbatim_display() {
        let probe_err = ProbeError::HandlerNotInstalled;
        let display = probe_err.to_string();
        let init_err: ModelInitError = probe_err.into();
        let ModelInitError::Backend { message, .. } = &init_err else {
            panic!("expected ModelInitError::Backend, got: {init_err:?}");
        };
        assert_eq!(
            message, &display,
            "Backend.message must equal ProbeError::HandlerNotInstalled Display output verbatim"
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

    // T-008: From<ProbeError::SetupRejected { .. }> and other backend-failure variants → Backend { .. }
    #[test]
    fn from_probe_error_setup_rejected_and_subprocess_failed_both_map_to_backend() {
        let setup_err: ModelInitError = ProbeError::SetupRejected {
            reason: SetupReason::PathOutsideCache,
        }
        .into();
        assert!(
            matches!(setup_err, ModelInitError::Backend { .. }),
            "ProbeError::SetupRejected must map to ModelInitError::Backend, got: {setup_err:?}"
        );

        let subprocess_err: ModelInitError = ProbeError::SubprocessFailed("y".into()).into();
        let ModelInitError::Backend { message, .. } = subprocess_err else {
            panic!("expected ModelInitError::Backend for SubprocessFailed");
        };
        assert_eq!(
            message, "y",
            "SubprocessFailed payload must round-trip through Backend.message"
        );
    }

    // T-125-A: backend(e) preserves the source error so std::error::Error::source()
    // walks the chain back to the original typed error.
    #[test]
    fn backend_helper_preserves_source_chain() {
        let probe_err = ProbeError::HandlerNotInstalled;
        let probe_display = probe_err.to_string();
        let init_err = ModelInitError::backend(probe_err);

        let source = init_err
            .source()
            .expect("backend() must populate Backend.source so the chain is walkable");
        assert_eq!(
            source.to_string(),
            probe_display,
            "Backend.source must Display verbatim as the originating ProbeError"
        );
    }

    // T-125-B: From<ProbeError::SetupRejected> routes through backend() and preserves
    // the source chain. Guards against a regression that special-cases SetupRejected
    // to construct Backend { source: None } directly and silently drops the typed error.
    #[test]
    fn from_probe_error_setup_rejected_preserves_source_chain() {
        let init_err: ModelInitError = ProbeError::SetupRejected {
            reason: SetupReason::PathOutsideCache,
        }
        .into();
        let source = init_err
            .source()
            .expect("SetupRejected must populate Backend.source via backend()");
        assert!(
            source.downcast_ref::<ProbeError>().is_some(),
            "Backend.source must downcast back to ProbeError, got: {source}"
        );
    }

    // T-125-C: SubprocessFailed has only a `String` payload (no upstream typed
    // error to box), so Backend.source is None — documented contract distinct
    // from HandlerNotInstalled / SetupRejected.
    #[test]
    fn from_probe_error_subprocess_failed_has_no_source() {
        let init_err: ModelInitError = ProbeError::SubprocessFailed("y".into()).into();
        assert!(
            init_err.source().is_none(),
            "SubprocessFailed maps to Backend with source: None — got source: {:?}",
            init_err.source().map(ToString::to_string)
        );
    }
}
