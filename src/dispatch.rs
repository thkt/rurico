//! Top-level probe dispatcher that wires the embed and reranker domains into
//! a single `main()` entry point.
//!
//! Lives above `model_probe`, `embed`, and `reranker` to break the dependency
//! cycle that previously had `model_probe` importing both domains while
//! `embed`/`reranker` re-exported `model_probe` symbols.

use std::env;

use crate::artifacts::{EmbedKind, RerankerKind};
use crate::model_lifecycle;
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
    if let Some(result) = model_lifecycle::probe_env_to_paths::<EmbedKind>(
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

    if let Some(result) = model_lifecycle::probe_env_to_paths::<RerankerKind>(
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
    use std::cell::Cell;
    use std::collections::HashMap;

    /// Per-kind query count from `handle_probe_if_needed_with`: every kind
    /// reads `model`, `config`, and `tokenizer` regardless of branch outcome.
    /// With embed + reranker dispatched in sequence, the lookup is called
    /// `2 * 3 = 6` times when neither kind triggers a probe.
    const NO_DISPATCH_LOOKUP_COUNT: usize = 6;

    #[test]
    fn handle_probe_if_needed_with_returns_when_no_env_set() {
        let calls = Cell::new(0_usize);
        handle_probe_if_needed_with(|_key| {
            calls.set(calls.get() + 1);
            None
        });
        // The function returned (no `process::exit` from `dispatch_probe`)
        // and still consulted every PROBE_ENV_* var — the empty env path
        // short-circuits at `probe_env_to_paths` returning None for both kinds.
        assert_eq!(
            calls.get(),
            NO_DISPATCH_LOOKUP_COUNT,
            "lookup must be queried for embed + reranker × {{model, config, tokenizer}}"
        );
    }

    #[test]
    fn handle_probe_if_needed_with_skips_paths_without_primary_keys() {
        let calls = Cell::new(0_usize);
        let map: HashMap<&'static str, &'static str> = [
            (embed::PROBE_ENV_CONFIG, "/tmp/c"),
            (reranker::PROBE_ENV_TOKENIZER, "/tmp/t"),
        ]
        .into_iter()
        .collect();
        handle_probe_if_needed_with(|key| {
            calls.set(calls.get() + 1);
            map.get(key).map(|v| (*v).to_owned())
        });
        // PROBE_ENV_MODEL is absent for both kinds, so `resolve_probe_env`
        // returns None and dispatch is skipped — guards against a regression
        // where setting only secondary keys tricks the function into spawning
        // a probe.
        assert_eq!(
            calls.get(),
            NO_DISPATCH_LOOKUP_COUNT,
            "lookup must be queried for all three vars per kind even when only secondary keys are set"
        );
    }
}
