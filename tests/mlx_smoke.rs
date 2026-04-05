//! Integration tests that run smoke binaries in subprocesses.
//!
//! All MLX-dependent verification is isolated here. A SIGABRT from MLX FFI
//! kills only the subprocess, not the test runner.
//!
//! - `smoke_full`: embed model functionality end-to-end (via `mlx_smoke`)
//! - `probe_smoke_binary`: subprocess probe contract end-to-end (via `probe_smoke`)

use std::process::Command;

/// Run the smoke binary and check it succeeds.
///
/// Covers: query embedding, document embedding (short + long + batch),
/// consistency, prefix-merge texts.
///
/// The smoke binary loads the model via `cached_artifacts` internally,
/// so the model must be downloaded before running this test.
#[test]
#[ignore] // requires model download + MLX (Apple Silicon)
fn smoke_full() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .output()
        .expect("spawn smoke binary");
    assert_smoke_success(&output);
}

/// Validate the subprocess probe contract end-to-end for both models.
///
/// Spawns the `probe_smoke` binary, which has `handle_probe_if_needed()` wired
/// in its `main()`. When `Embedder::probe()` / `Reranker::probe()` re-exec
/// `current_exe()`, they re-exec `probe_smoke` — the correct probe host — so the
/// full probe cycle is exercised rather than a test harness that ignores probe env vars.
///
/// Each model section is skipped if not cached; at least one model must be cached.
#[test]
#[ignore] // requires model download + MLX (Apple Silicon)
fn probe_smoke_binary() {
    let output = Command::new(env!("CARGO_BIN_EXE_probe_smoke"))
        .output()
        .expect("spawn probe_smoke binary");
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
