//! Hugging Face Hub I/O backend — the thin glue that drives real hf-hub 1.0
//! network downloads and local-cache resolution.
//!
//! Isolated in its own file so the diff-coverage gate can exclude it (see the
//! `ignore-filename-regex` in `.github/workflows/ci.yml`). These functions
//! either require a live HF endpoint ([`download_artifacts`]) or map an
//! hf-hub-internal error variant ([`cached_file`]'s non-`LocalEntryNotFound`
//! arm) that unit tests cannot reach without depending on hf-hub's private
//! resolution internals. The pure orchestration and branching that *can* be
//! tested (thread/timeout in [`super::download_artifacts_with`], the
//! three-file fan-out in [`super::artifacts_from_cache`]) stays in the parent
//! module and remains measured.

use std::path::PathBuf;

use hf_hub::{HFRepositorySync, RepoTypeModel};

use super::{
    DOWNLOAD_TIMEOUT, ModelArtifact, ModelIoError, ModelPaths, download_artifacts_with,
    split_repo_id,
};

/// Download model files from Hugging Face Hub (cached after first download).
///
/// # Errors
///
/// Returns [`ModelIoError::Download`] if the Hugging Face client cannot be
/// initialized or any required artifact download fails.
///
/// # Interruption Safety
///
/// hf-hub downloads each file to a `.part` temporary file and then atomically
/// renames it to the final blob path (`std::fs::rename`). A signal (e.g. SIGINT)
/// received during a download leaves only the `.part` file on disk; the final
/// blob is never partially written.
///
/// On the next invocation, [`super::artifacts_if_cached`] finds no valid
/// pointer and returns `None`, so the download retries cleanly — no manual
/// cache cleanup needed.
pub fn download_artifacts<Id: ModelArtifact>(model: Id) -> Result<ModelPaths, ModelIoError> {
    download_artifacts_with(model, DOWNLOAD_TIMEOUT, |repo_id, revision| {
        let client = hf_hub::HFClientSync::new()
            .map_err(|e| ModelIoError::Download(format!("HF Hub init failed: {e}")))?;
        let (owner, name) = split_repo_id(repo_id);
        let repo = client.model(owner, name);
        let get = |file: &str| {
            repo.download_file()
                .filename(file)
                .revision(revision)
                .send()
                .map_err(|e| ModelIoError::Download(format!("{file} download failed: {e}")))
        };
        Ok(ModelPaths {
            model: get("model.safetensors")?,
            config: get("config.json")?,
            tokenizer: get("tokenizer.json")?,
        })
    })
}

/// Resolve one artifact from the local HF Hub cache without network access.
///
/// `local_files_only(true)` guarantees no network access; a cache miss surfaces
/// as `LocalEntryNotFound`, which maps to `Ok(None)` (retry the download). Any
/// other error is a real local-filesystem/config fault worth propagating.
pub(crate) fn cached_file(
    repo: &HFRepositorySync<RepoTypeModel>,
    file: &str,
    revision: &str,
) -> Result<Option<PathBuf>, ModelIoError> {
    match repo
        .download_file()
        .filename(file)
        .revision(revision)
        .local_files_only(true)
        .send()
    {
        Ok(path) => Ok(Some(path)),
        Err(hf_hub::HFError::LocalEntryNotFound { .. }) => Ok(None),
        Err(e) => Err(ModelIoError::Download(format!(
            "{file} cache lookup failed: {e}"
        ))),
    }
}
