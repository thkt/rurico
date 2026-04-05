use super::{Artifacts, CandidateArtifacts, EmbedInitError};
use crate::model_probe::{
    self, EMBED_PROBE_ENV_CONFIG, EMBED_PROBE_ENV_MODEL, EMBED_PROBE_ENV_TOKENIZER,
};

/// Resolve probe env vars into a [`CandidateArtifacts`].
///
/// Returns `None` if this is not a probe invocation (primary model env var absent).
/// Returns `Some(Ok(candidate))` if all three vars are set.
/// Returns `Some(Err(3))` if the model var is set but config or tokenizer is missing.
pub(crate) fn probe_env_to_paths(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<CandidateArtifacts, i32>> {
    crate::model_probe::resolve_probe_env(model, config, tokenizer)
        .map(|r| r.map(|(m, c, t)| CandidateArtifacts::from_paths(m, c, t)))
}

/// Re-exec the current binary as a probe subprocess for the embedding model.
pub(super) fn probe_via_subprocess(
    artifacts: &Artifacts,
) -> Result<crate::model_probe::ProbeStatus, EmbedInitError> {
    let paths = &artifacts.paths;
    model_probe::probe_paths_via_subprocess(
        paths,
        EMBED_PROBE_ENV_MODEL,
        EMBED_PROBE_ENV_CONFIG,
        EMBED_PROBE_ENV_TOKENIZER,
    )
    .map_err(Into::into)
}
