use super::{Artifacts, ModelInitError, PROBE_ENV_CONFIG, PROBE_ENV_MODEL, PROBE_ENV_TOKENIZER};
use crate::model_probe::{self, ProbeStatus};

/// Re-exec the current binary as a probe subprocess for the embedding model.
pub(super) fn probe_via_subprocess(artifacts: &Artifacts) -> Result<ProbeStatus, ModelInitError> {
    let paths = &artifacts.paths;
    model_probe::probe_paths_via_subprocess(
        paths,
        PROBE_ENV_MODEL,
        PROBE_ENV_CONFIG,
        PROBE_ENV_TOKENIZER,
    )
    .map_err(Into::into)
}
