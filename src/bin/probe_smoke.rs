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

    // ── Embed probe ───────────────────────────────────────────────────────────

    let embed_artifacts = rurico::embed::cached_artifacts(rurico::embed::ModelId::default())
        .expect("embed cache lookup failed")
        .expect("embed model not cached; run download first");

    let embed_status = rurico::embed::Embedder::probe(&embed_artifacts)
        .expect("embed probe subprocess should not error");
    assert_eq!(
        embed_status,
        rurico::embed::ProbeStatus::Available,
        "embed model should be available"
    );

    // ── Reranker probe ────────────────────────────────────────────────────────

    let reranker_artifacts = reranker::cached_artifacts(reranker::RerankerModelId::default())
        .expect("reranker cache lookup failed")
        .expect("reranker model not cached; run download first");

    let reranker_status = reranker::Reranker::probe(&reranker_artifacts)
        .expect("reranker probe subprocess should not error");
    assert_eq!(
        reranker_status,
        reranker::ProbeStatus::Available,
        "reranker model should be available"
    );

    eprintln!("probe smoke: embed + reranker available");
}
