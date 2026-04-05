//! Integration tests that run smoke binaries in subprocesses.
//!
//! All MLX-dependent verification is isolated here. A SIGABRT from MLX FFI
//! kills only the subprocess, not the test runner.
//!
//! - `smoke_full`: embed model functionality end-to-end (via `mlx_smoke`)
//! - `probe_embed_smoke_binary`: embed subprocess probe contract (via `probe_embed_smoke`)
//! - `probe_reranker_smoke_binary`: reranker subprocess probe contract (via `probe_reranker_smoke`)

use std::process::Command;

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

fn assert_smoke_success(output: &std::process::Output) {
    if output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
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
