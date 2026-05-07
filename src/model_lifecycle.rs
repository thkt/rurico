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
use crate::model_probe::{resolve_probe_env, validate_probe_paths};

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
    let paths = match download_artifacts(model) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(
                repo_id,
                revision,
                error = %e,
                "model_lifecycle: download failed"
            );
            return Err(e.into());
        }
    };
    let verified = match Id::Kind::verify(paths) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                repo_id,
                revision,
                error = %e,
                "model_lifecycle: verification failed"
            );
            return Err(e);
        }
    };
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
/// missing or the paths fail validation (exit code from
/// [`validate_probe_paths`]).
pub(crate) fn probe_env_to_paths<K: ModelKind>(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<CandidateArtifacts<K>, i32>> {
    resolve_probe_env(model, config, tokenizer).map(|r| {
        r.and_then(|(m, c, t)| {
            validate_probe_paths(&m, &c, &t)?;
            Ok((m, c, t))
        })
        .map(|(m, c, t)| CandidateArtifacts::from_paths(m, c, t))
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use crate::artifacts::{
        ArtifactError, CandidateArtifacts, EmbedKind, RerankerKind, VerifiedArtifacts,
    };
    use crate::embed::ModelId;
    use crate::model_io::ModelArtifact;
    use crate::model_lifecycle::{cached_artifacts, download_model, probe_env_to_paths};
    use crate::reranker::RerankerModelId;
    use crate::test_support::{
        FAKE_BACKBONE_KEY, MINIMAL_TOKENIZER_JSON, VALID_CONFIG_JSON, setup_fake_hf_cache,
        write_fake_safetensors,
    };

    /// Build an embed-kind safetensors file (backbone-only keys).
    fn write_embed_safetensors(path: &Path) {
        write_fake_safetensors(path, &[FAKE_BACKBONE_KEY]);
    }

    /// Compile-time signature assertion helper. The function body is unused;
    /// the `where` clause forces the type checker to confirm `F` returns `T`.
    fn assert_returns<F, T>(_f: F)
    where
        F: FnOnce() -> T,
    {
    }

    // T-010: download_model(ModelId::default()) returns Result<VerifiedArtifacts<EmbedKind>, ArtifactError>
    #[test]
    fn download_model_for_embed_id_returns_verified_artifacts_of_embed_kind() {
        // Compile-time signature check only; do not invoke (would hit network).
        assert_returns::<_, Result<VerifiedArtifacts<EmbedKind>, ArtifactError>>(|| {
            download_model(ModelId::default())
        });
    }

    // T-011: download_model(RerankerModelId::default()) returns Result<VerifiedArtifacts<RerankerKind>, ArtifactError>
    #[test]
    fn download_model_for_reranker_id_returns_verified_artifacts_of_reranker_kind() {
        assert_returns::<_, Result<VerifiedArtifacts<RerankerKind>, ArtifactError>>(|| {
            download_model(RerankerModelId::default())
        });
    }

    // T-012: cached_artifacts(ModelId::default()) with all artifacts present → Ok(Some(VerifiedArtifacts<EmbedKind>))
    #[test]
    fn cached_artifacts_for_embed_id_returns_some_when_cache_populated() {
        let hf_home = tempfile::tempdir().unwrap();
        let cache_root = hf_home.path().join("hub");
        fs::create_dir_all(&cache_root).unwrap();

        // Stage real backbone-keyed safetensors + valid config + valid tokenizer
        // outside the cache, then point the fake HF cache at them via direct
        // file content so verify() can succeed.
        let model_id = ModelId::default();
        let scratch = tempfile::tempdir().unwrap();
        let model_path = scratch.path().join("model.safetensors");
        write_embed_safetensors(&model_path);
        let model_bytes = fs::read(&model_path).unwrap();

        setup_fake_hf_cache(
            &cache_root,
            model_id.repo_id(),
            model_id.revision(),
            &[
                ("model.safetensors", model_bytes.as_slice()),
                ("config.json", VALID_CONFIG_JSON.as_bytes()),
                ("tokenizer.json", MINIMAL_TOKENIZER_JSON.as_bytes()),
            ],
        );

        let hf_home_path = hf_home.path().to_path_buf();
        temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
            let result: Result<Option<VerifiedArtifacts<EmbedKind>>, ArtifactError> =
                cached_artifacts(model_id);
            let opt = result.expect("cached_artifacts should succeed when cache is populated");
            assert!(
                opt.is_some(),
                "expected Some(VerifiedArtifacts<EmbedKind>) when all three files are cached"
            );
        });
    }

    // T-013: cached_artifacts(RerankerModelId::default()) missing tokenizer → Ok(None)
    #[test]
    fn cached_artifacts_for_reranker_id_returns_none_when_tokenizer_missing() {
        let hf_home = tempfile::tempdir().unwrap();
        let cache_root = hf_home.path().join("hub");
        fs::create_dir_all(&cache_root).unwrap();

        let model_id = RerankerModelId::default();
        // Omit tokenizer.json so cache lookup short-circuits to None before verify.
        setup_fake_hf_cache(
            &cache_root,
            model_id.repo_id(),
            model_id.revision(),
            &[
                ("model.safetensors", b"placeholder"),
                ("config.json", b"{}"),
            ],
        );

        let hf_home_path = hf_home.path().to_path_buf();
        temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
            let result: Result<Option<VerifiedArtifacts<RerankerKind>>, ArtifactError> =
                cached_artifacts(model_id);
            let opt = result.expect("cached_artifacts should not error on cache miss");
            assert!(
                opt.is_none(),
                "expected Ok(None) when tokenizer.json is absent from cache"
            );
        });
    }

    // T-014: probe_env_to_paths::<EmbedKind>(...) with three valid paths → Some(Ok(CandidateArtifacts<EmbedKind>))
    #[test]
    fn probe_env_to_paths_for_embed_kind_returns_candidate_when_all_paths_valid() {
        let hf_home = tempfile::tempdir().unwrap();
        let cache_root = hf_home.path().join("hub");
        fs::create_dir_all(&cache_root).unwrap();
        let m = cache_root.join("model.safetensors");
        let c = cache_root.join("config.json");
        let t = cache_root.join("tokenizer.json");
        fs::write(&m, b"").unwrap();
        fs::write(&c, b"{}").unwrap();
        fs::write(&t, b"{}").unwrap();

        let hf_home_path = hf_home.path().to_path_buf();
        temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
            let result: Option<Result<CandidateArtifacts<EmbedKind>, i32>> =
                probe_env_to_paths::<EmbedKind>(
                    Some(m.to_string_lossy().into_owned()),
                    Some(c.to_string_lossy().into_owned()),
                    Some(t.to_string_lossy().into_owned()),
                );
            let inner =
                result.expect("probe_env_to_paths should return Some when model var is set");
            assert!(
                inner.is_ok(),
                "expected Ok(CandidateArtifacts<EmbedKind>) for three valid paths"
            );
        });
    }

    // T-015: probe_env_to_paths::<RerankerKind>(...) with three valid paths → Some(Ok(CandidateArtifacts<RerankerKind>))
    #[test]
    fn probe_env_to_paths_for_reranker_kind_returns_candidate_when_all_paths_valid() {
        let hf_home = tempfile::tempdir().unwrap();
        let cache_root = hf_home.path().join("hub");
        fs::create_dir_all(&cache_root).unwrap();
        let m = cache_root.join("model.safetensors");
        let c = cache_root.join("config.json");
        let t = cache_root.join("tokenizer.json");
        fs::write(&m, b"").unwrap();
        fs::write(&c, b"{}").unwrap();
        fs::write(&t, b"{}").unwrap();

        let hf_home_path = hf_home.path().to_path_buf();
        temp_env::with_vars([("HF_HOME", Some(hf_home_path.to_str().unwrap()))], || {
            let result: Option<Result<CandidateArtifacts<RerankerKind>, i32>> =
                probe_env_to_paths::<RerankerKind>(
                    Some(m.to_string_lossy().into_owned()),
                    Some(c.to_string_lossy().into_owned()),
                    Some(t.to_string_lossy().into_owned()),
                );
            let inner =
                result.expect("probe_env_to_paths should return Some when model var is set");
            assert!(
                inner.is_ok(),
                "expected Ok(CandidateArtifacts<RerankerKind>) for three valid paths"
            );
        });
    }
}
