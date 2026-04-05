//! Probe smoke-test binary — validates the subprocess probe contract end-to-end.
//!
//! Invoked by integration tests via `Command` so the re-exec target is a real
//! binary with `handle_probe_if_needed()` wired in `main()`.
//!
//! `Embedder::probe()` and `Reranker::probe()` internally re-exec `current_exe()`.
//! When this binary is the host, `current_exe()` resolves to `probe_smoke` — a
//! binary that actually calls `handle_probe_if_needed()` — so the probe contract
//! is tested end-to-end rather than against a test harness that ignores probe env vars.
//!
//! Both models must be downloaded before running probe smoke tests.

use rurico::reranker;

fn main() {
    // Must be first: handles re-exec when called as a probe subprocess.
    rurico::model_probe::handle_probe_if_needed();

    let mut ran = 0u32;

    // ── Embed probe ───────────────────────────────────────────────────────────

    match rurico::embed::cached_artifacts(rurico::embed::ModelId::default())
        .expect("embed cache lookup failed")
    {
        None => eprintln!("probe smoke: embed model not cached — skipping"),
        Some(artifacts) => {
            let status = rurico::embed::Embedder::probe(&artifacts)
                .expect("embed probe subprocess should not error");
            assert_eq!(
                status,
                rurico::embed::ProbeStatus::Available,
                "embed model should be available"
            );
            eprintln!("probe smoke: embed OK");
            ran += 1;
        }
    }

    // ── Reranker probe ────────────────────────────────────────────────────────

    match reranker::cached_artifacts(reranker::RerankerModelId::default())
        .expect("reranker cache lookup failed")
    {
        None => eprintln!("probe smoke: reranker model not cached — skipping"),
        Some(artifacts) => {
            let status = reranker::Reranker::probe(&artifacts)
                .expect("reranker probe subprocess should not error");
            assert_eq!(
                status,
                reranker::ProbeStatus::Available,
                "reranker model should be available"
            );
            eprintln!("probe smoke: reranker OK");
            ran += 1;
        }
    }

    assert!(
        ran > 0,
        "no models cached — download at least one before running probe_smoke"
    );
    eprintln!("probe smoke: {ran}/2 model(s) verified");
}
