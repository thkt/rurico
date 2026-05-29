use super::*;
use std::collections::HashMap;

// Both tests rely on a side-effect oracle: `dispatch_probe` would call
// `process::exit` and kill the test runner, so reaching the end of the
// test proves the no-dispatch path was taken for both kinds.

#[test]
fn handle_probe_if_needed_with_returns_when_no_env_set() {
    handle_probe_if_needed_with(|_key| None);
}

#[test]
fn handle_probe_if_needed_with_skips_paths_without_primary_keys() {
    let map: HashMap<&'static str, &'static str> = [
        (embed::PROBE_ENV_CONFIG, "/tmp/c"),
        (reranker::PROBE_ENV_TOKENIZER, "/tmp/t"),
    ]
    .into_iter()
    .collect();
    // PROBE_ENV_MODEL is absent for both kinds, so dispatch is skipped:
    // setting only secondary keys must NOT trick the function into
    // spawning a probe.
    handle_probe_if_needed_with(|key| map.get(key).map(|v| (*v).to_owned()));
}
