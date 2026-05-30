//! Generic model lifecycle: download / cache lookup / probe-env resolution.
//!
//! Provides the three kind-generic entry points that replace the per-module
//! duplicates that previously lived in `embed.rs` / `reranker.rs`:
//!
//! - [`download_model`] -- fetch HF Hub artifacts and verify them as `Id::Kind`.
//! - [`cached_artifacts`] -- inspect the local HF cache for `Id::Kind` artifacts.
//! - [`probe_env_to_paths`] -- resolve probe env vars into a kind-bound candidate.
//!
//! The single `Id::Kind` associated type binds the model identifier to its
//! kind marker so wrong-kind combinations (e.g. an embed identifier returning
//! reranker artifacts) cannot be expressed at the call site.

use crate::artifacts::{ArtifactError, CandidateArtifacts, ModelKind, VerifiedArtifacts};
use crate::model_io::{ModelArtifact, artifacts_if_cached, download_artifacts};
use crate::model_probe::{SetupReason, resolve_probe_env, validate_probe_paths};

/// Download model files from Hugging Face Hub and verify them as `Id::Kind` artifacts.
///
/// # Errors
///
/// Returns [`ArtifactError::DownloadFailed`] if the Hugging Face client cannot be
/// initialised or any required artifact download fails. Returns other
/// [`ArtifactError`] variants if verification of the downloaded files fails.
pub fn download_model<Id: ModelArtifact>(
    model: Id,
) -> Result<VerifiedArtifacts<Id::Kind>, ArtifactError> {
    let repo_id = model.repo_id();
    let revision = model.revision();
    tracing::info!(repo_id, revision, "model_lifecycle: download start");
    let paths = download_artifacts(model).inspect_err(|e| {
        tracing::error!(repo_id, revision, error = %e, "model_lifecycle: download failed");
    })?;
    let verified = Id::Kind::verify(paths).inspect_err(|e| {
        tracing::error!(repo_id, revision, error = %e, "model_lifecycle: verification failed");
    })?;
    tracing::info!(repo_id, revision, "model_lifecycle: download complete");
    Ok(verified)
}

/// Check whether model files exist in the local HF Hub cache and verify them.
///
/// Returns `Ok(Some(_))` if all three files are cached and pass verification,
/// `Ok(None)` if any file is missing from the cache. Never accesses the network.
///
/// # Errors
///
/// Returns [`ArtifactError`] if cached files exist but fail verification.
/// Cache misses are reported as `Ok(None)`.
pub fn cached_artifacts<Id: ModelArtifact>(
    model: Id,
) -> Result<Option<VerifiedArtifacts<Id::Kind>>, ArtifactError> {
    let repo_id = model.repo_id();
    let Some(paths) = artifacts_if_cached(model)? else {
        tracing::debug!(repo_id, "model_lifecycle: cache miss");
        return Ok(None);
    };
    let verified = Id::Kind::verify(paths)?;
    tracing::info!(repo_id, "model_lifecycle: loaded from cache");
    Ok(Some(verified))
}

/// Resolve probe env vars into a [`CandidateArtifacts<K>`].
///
/// Returns `None` if this is not a probe invocation (primary model env var absent).
/// Returns `Some(Ok(_))` if all three vars are set and pass path validation.
/// Returns `Some(Err(_))` if the model var is set but config or tokenizer is
/// missing or the paths fail validation ([`SetupReason`] from
/// [`validate_probe_paths`]).
pub(crate) fn probe_env_to_paths<K: ModelKind>(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<CandidateArtifacts<K>, SetupReason>> {
    resolve_probe_env(model, config, tokenizer).map(|r| {
        r.and_then(|(m, c, t)| {
            validate_probe_paths(&m, &c, &t)?;
            Ok((m, c, t))
        })
        .map(|(m, c, t)| CandidateArtifacts::from_paths(m, c, t))
    })
}

#[cfg(test)]
mod tests;
