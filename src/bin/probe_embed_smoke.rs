//! Embed probe smoke-test binary — validates the embed subprocess probe contract.
//!
//! Requires ruri-v3-310m to be cached locally. Exits non-zero if the model is
//! not cached.
//!
//! Invoked by integration tests via `Command` so the re-exec target is a real
//! binary with `handle_probe_if_needed()` wired in `main()`. When
//! `Embedder::probe()` re-execs `current_exe()`, it re-execs this binary —
//! so the full probe cycle is exercised end-to-end.

fn main() {
    // Must be first: handles re-exec when called as a probe subprocess.
    rurico::model_probe::handle_probe_if_needed();

    let artifacts = rurico::embed::cached_artifacts(rurico::embed::ModelId::default())
        .expect("embed cache lookup failed")
        .expect("embed model not cached — download ruri-v3-310m before running");

    let status = rurico::embed::Embedder::probe(&artifacts)
        .expect("embed probe subprocess should not error");
    assert_eq!(
        status,
        rurico::embed::ProbeStatus::Available,
        "embed model should be available"
    );
    eprintln!("probe_embed_smoke: embed OK");
}
