//! Top-level probe dispatcher that wires the embed and reranker domains into
//! a single `main()` entry point.
//!
//! Lives above `model_probe`, `embed`, and `reranker` to break the dependency
//! cycle that previously had `model_probe` importing both domains while
//! `embed`/`reranker` re-exported `model_probe` symbols.

use std::env;

use crate::{embed, model_probe, reranker};

/// Single entry point for probe subprocess dispatch in host binaries.
///
/// Call at the start of `main()`. When invoked as a probe subprocess (detected
/// via env vars), this function loads the appropriate model and exits via
/// [`std::process::exit`]. Otherwise returns immediately.
///
/// # Process Behavior
///
/// When the current process is a probe subprocess:
///
/// - Exit 0: model loaded successfully
/// - Exit 1: model load failed (reason written to stderr)
/// - Exit 3: primary env var set but config or tokenizer env var missing
pub fn handle_probe_if_needed() {
    handle_probe_if_needed_with(|key| env::var(key).ok())
}

/// Like [`handle_probe_if_needed`] but the env-var lookup is provided
/// explicitly. Used by tests to drive both probe paths without mutating the
/// process environment (forbidden under Rust 2024).
pub fn handle_probe_if_needed_with<F>(get: F)
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(result) = embed::probe_env_to_paths(
        get(embed::PROBE_ENV_MODEL),
        get(embed::PROBE_ENV_CONFIG),
        get(embed::PROBE_ENV_TOKENIZER),
    ) {
        model_probe::dispatch_probe(result, |candidate| {
            let artifacts = candidate.verify().map_err(|e| e.to_string())?;
            embed::Embedder::new(&artifacts)
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
    }

    if let Some(result) = reranker::probe_env_to_paths(
        get(reranker::PROBE_ENV_MODEL),
        get(reranker::PROBE_ENV_CONFIG),
        get(reranker::PROBE_ENV_TOKENIZER),
    ) {
        model_probe::dispatch_probe(result, |candidate| {
            let artifacts = candidate.verify().map_err(|e| e.to_string())?;
            reranker::Reranker::new(&artifacts)
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup_from(map: &HashMap<&'static str, &'static str>) -> impl Fn(&str) -> Option<String> {
        let owned: HashMap<String, String> = map
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |key: &str| owned.get(key).cloned()
    }

    #[test]
    fn handle_probe_if_needed_with_returns_when_no_env_set() {
        let map: HashMap<&'static str, &'static str> = HashMap::new();
        handle_probe_if_needed_with(lookup_from(&map));
    }

    #[test]
    fn handle_probe_if_needed_with_skips_paths_without_primary_keys() {
        let mut map = HashMap::new();
        map.insert(embed::PROBE_ENV_CONFIG, "/tmp/c");
        map.insert(reranker::PROBE_ENV_TOKENIZER, "/tmp/t");
        handle_probe_if_needed_with(lookup_from(&map));
    }
}
