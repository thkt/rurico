//! Integration tests that run smoke binaries in subprocesses.
//!
//! All MLX-dependent verification is isolated here. A SIGABRT from MLX FFI
//! kills only the subprocess, not the test runner.
//!
//! - `smoke_full`: embed model functionality end-to-end (via `mlx_smoke`)
//! - `smoke_verify_fixture`: Phase 2 numerical equivalence (T-BIT-001〜003)
//!   — invokes `mlx_smoke verify-fixture` which compares W1/W2/W3 output
//!   against `tests/fixtures/phase2_baseline/w{1,2,3}.bin` at Spec NFR-001
//!   tolerances.
//! - `probe_embed_smoke_binary`: embed subprocess probe contract (via `probe_embed_smoke`)
//! - `probe_reranker_smoke_binary`: reranker subprocess probe contract (via `probe_reranker_smoke`)

use std::process::{Command, Output};

use rurico::sandbox::SEATBELT_SKIP_EXIT;

/// Run the embed smoke binary and check it succeeds.
///
/// Covers: query embedding, document embedding (short + long + batch),
/// consistency, prefix-merge texts.
///
/// The smoke binary loads the model via `cached_artifacts` internally,
/// so the model must be downloaded before running this test.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn smoke_full() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .output()
        .expect("spawn smoke binary");
    assert_smoke_success(&output);
}

/// T-BIT-001〜003: Phase 2 numerical equivalence check.
///
/// Runs the bucket-batched `embed_documents_batch` on W1/W2/W3, loads the
/// committed Phase 1 fixtures, and verifies `cosine_similarity ≥ 0.99999`
/// AND `max_abs_diff ≤ 1e-5` on every chunk pair (Spec NFR-001).
///
/// Re-generate the fixtures via `mlx_smoke capture-fixture` when the expected
/// output genuinely changes (model upgrade, intentional algorithm change).
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn smoke_verify_fixture() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("verify-fixture")
        .output()
        .expect("spawn mlx_smoke verify-fixture");
    assert_smoke_success(&output);
}

/// Validate the embed subprocess probe contract end-to-end.
///
/// Spawns `probe_embed_smoke`, which has `handle_probe_if_needed()` wired in
/// its `main()`. When `Embedder::probe()` re-execs `current_exe()`, it
/// re-execs `probe_embed_smoke` — so the full probe cycle is exercised rather
/// than a test harness that ignores probe env vars.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn probe_embed_smoke_binary() {
    let output = Command::new(env!("CARGO_BIN_EXE_probe_embed_smoke"))
        .output()
        .expect("spawn probe_embed_smoke binary");
    assert_smoke_success(&output);
}

/// Validate the reranker subprocess probe contract end-to-end.
///
/// Spawns `probe_reranker_smoke`, which has `handle_probe_if_needed()` wired
/// in its `main()`. When `Reranker::probe()` re-execs `current_exe()`, it
/// re-execs `probe_reranker_smoke` — so the full probe cycle is exercised
/// rather than a test harness that ignores probe env vars.
#[test]
#[ignore] // requires ruri-v3-reranker-310m cached + MLX (Apple Silicon)
fn probe_reranker_smoke_binary() {
    let output = Command::new(env!("CARGO_BIN_EXE_probe_reranker_smoke"))
        .output()
        .expect("spawn probe_reranker_smoke binary");
    assert_smoke_success(&output);
}

fn assert_smoke_success(output: &Output) {
    if output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.code() == Some(SEATBELT_SKIP_EXIT) {
        panic!(
            "smoke binary skipped in Codex seatbelt sandbox; \
             run this verification outside the sandbox\nstderr: {stderr}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = output.status.signal() {
            panic!("smoke binary killed by signal {sig} (MLX FFI crash)\nstderr: {stderr}");
        }
    }
    panic!(
        "smoke binary failed with {:?}\nstderr: {stderr}",
        output.status.code()
    );
}
