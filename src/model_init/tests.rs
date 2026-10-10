use std::error::Error as _;

use crate::model_init::ModelInitError;
use crate::model_probe::{ProbeError, SetupReason};

#[test]
fn probe_backend_errors_preserve_verbatim_message_and_typed_source() {
    for probe in [
        ProbeError::HandlerNotInstalled,
        ProbeError::SetupRejected {
            reason: SetupReason::EnvIncomplete,
        },
        ProbeError::SetupRejected {
            reason: SetupReason::CanonicalizeFailed,
        },
        ProbeError::SetupRejected {
            reason: SetupReason::PathOutsideCache,
        },
        ProbeError::SetupRejected {
            reason: SetupReason::CacheRootInvalid,
        },
    ] {
        let display = probe.to_string();
        let init: ModelInitError = probe.into();
        let ModelInitError::Backend { message, .. } = &init else {
            panic!("expected Backend: {init:?}");
        };
        assert_eq!(message, &display);
        let source = init
            .source()
            .expect("typed probe cause must survive conversion");
        let source = source
            .downcast_ref::<ProbeError>()
            .expect("original ProbeError");
        assert_eq!(source.to_string(), display);
    }
}

#[test]
fn probe_string_errors_preserve_payload_without_inventing_source() {
    let init: ModelInitError = ProbeError::ModelLoadFailed {
        reason: "bad weights".into(),
    }
    .into();
    let ModelInitError::ModelCorrupt { reason } = &init else {
        panic!("expected ModelCorrupt: {init:?}");
    };
    assert_eq!(reason, "bad weights");
    assert!(init.source().is_none());

    let init: ModelInitError = ProbeError::SubprocessFailed("spawn failed".into()).into();
    let ModelInitError::Backend { message, .. } = &init else {
        panic!("expected Backend: {init:?}");
    };
    assert_eq!(message, "spawn failed");
    assert!(init.source().is_none());
}
