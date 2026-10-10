//! Hugging Face Hub I/O backend — the thin glue that drives real hf-hub 1.0
//! network downloads and local-cache resolution.
//!
//! Network operations remain separate from the subprocess lifetime manager;
//! synthetic workers exercise that manager without a live HF endpoint.
//! The vendored 1.0.0 cache publisher never unlinks an existing Unix snapshot.

use std::path::PathBuf;

use hf_hub::{HFRepositorySync, RepoTypeModel};

use super::{ModelArtifact, ModelIoError, ModelPaths, split_repo_id};

/// Download model files from Hugging Face Hub (cached after first download).
///
/// # Errors
///
/// Returns [`ModelIoError::Download`] if the Hugging Face client cannot be
/// initialized or any required artifact download fails.
///
/// Requires `crate::handle_probe_if_needed()` at the start of the host's main.
/// Timeout stops the owned worker; incomplete cache files are left for retry.
pub fn download_artifacts<Id: ModelArtifact>(model: Id) -> Result<ModelPaths, ModelIoError> {
    super::download_process::download(model)
}

pub(super) fn download_in_worker(
    repo_id: &str,
    revision: &str,
) -> Result<ModelPaths, ModelIoError> {
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
