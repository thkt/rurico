//! Reranker probe smoke-test binary — validates the reranker subprocess probe contract.
//!
//! Requires ruri-v3-reranker-310m to be cached locally. Exits non-zero if the
//! model is not cached.
//!
//! Invoked by integration tests via `Command` so the re-exec target is a real
//! binary with `handle_probe_if_needed()` wired in `main()`. When
//! `Reranker::probe()` re-execs `current_exe()`, it re-execs this binary —
//! so the full probe cycle is exercised end-to-end.

use std::env;
use std::process;

use rurico::model_probe;
use rurico::reranker::{ProbeStatus, Reranker, RerankerModelId, cached_artifacts};

const SEATBELT_SKIP_EXIT: i32 = 78;

fn codex_seatbelt_sandbox_active() -> bool {
    env::var("CODEX_SANDBOX").is_ok_and(|v| v == "seatbelt")
}

fn main() {
    // Must be first: handles re-exec when called as a probe subprocess.
    model_probe::handle_probe_if_needed();

    if codex_seatbelt_sandbox_active() {
        eprintln!(
            "probe_reranker_smoke: skipped in Codex seatbelt sandbox; \
             run outside sandbox for MLX probe verification"
        );
        process::exit(SEATBELT_SKIP_EXIT);
    }

    let artifacts = cached_artifacts(RerankerModelId::default())
        .expect("reranker cache lookup failed")
        .expect("reranker model not cached — download ruri-v3-reranker-310m before running");

    let status = Reranker::probe(&artifacts).expect("reranker probe subprocess should not error");
    assert_eq!(
        status,
        ProbeStatus::Available,
        "reranker model should be available"
    );
    eprintln!("probe_reranker_smoke: reranker OK");
}
