use super::{
    Artifacts, CandidateArtifacts, EmbedInitError, PROBE_ENV_CONFIG, PROBE_ENV_MODEL,
    PROBE_ENV_TOKENIZER,
};
use crate::model_probe::{self, ProbeStatus, resolve_probe_env, validate_probe_paths};

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
    resolve_probe_env(model, config, tokenizer).map(|r| {
        r.and_then(|(m, c, t)| {
            validate_probe_paths(&m, &c, &t)?;
            Ok((m, c, t))
        })
        .map(|(m, c, t)| CandidateArtifacts::from_paths(m, c, t))
    })
}

/// Re-exec the current binary as a probe subprocess for the embedding model.
pub(super) fn probe_via_subprocess(artifacts: &Artifacts) -> Result<ProbeStatus, EmbedInitError> {
    let paths = &artifacts.paths;
    model_probe::probe_paths_via_subprocess(
        paths,
        PROBE_ENV_MODEL,
        PROBE_ENV_CONFIG,
        PROBE_ENV_TOKENIZER,
    )
    .map_err(Into::into)
}
