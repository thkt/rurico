//! Reranker probe smoke-test binary — validates the reranker subprocess probe contract.
//!
//! Requires ruri-v3-reranker-310m to be cached locally. Exits non-zero if the
//! model is not cached.
//!
//! Invoked by integration tests via `Command` so the re-exec target is a real
//! binary with `handle_probe_if_needed()` wired in `main()`. When
//! `Reranker::probe()` re-execs `current_exe()`, it re-execs this binary —
//! so the full probe cycle is exercised end-to-end.

fn main() {
    // Must be first: handles re-exec when called as a probe subprocess.
    rurico::model_probe::handle_probe_if_needed();

    let artifacts =
        rurico::reranker::cached_artifacts(rurico::reranker::RerankerModelId::default())
            .expect("reranker cache lookup failed")
            .expect("reranker model not cached — download ruri-v3-reranker-310m before running");

    let status = rurico::reranker::Reranker::probe(&artifacts)
        .expect("reranker probe subprocess should not error");
    assert_eq!(
        status,
        rurico::reranker::ProbeStatus::Available,
        "reranker model should be available"
    );
    eprintln!("probe_reranker_smoke: reranker OK");
}
