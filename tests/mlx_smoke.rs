//! Integration tests that run the `mlx_smoke` binary in a subprocess.
//!
//! All MLX-dependent verification is isolated here. A SIGABRT from MLX FFI
//! kills only the subprocess, not the test runner.

use std::process::Command;

/// Run the smoke binary and check it succeeds.
///
/// Covers: query embedding, document embedding (short + long + batch),
/// consistency, prefix-merge texts.
#[test]
#[ignore] // requires model download + MLX (Apple Silicon)
fn smoke_full() {
    let paths =
        rurico::embed::download_model(rurico::embed::ModelId::default()).expect("download model");
    let output = smoke_command(&paths).output().expect("spawn smoke binary");
    assert_smoke_success(&output);
}

/// Run the smoke binary in probe mode (env vars from `handle_probe_if_needed`).
#[test]
#[ignore] // requires model download + MLX (Apple Silicon)
fn smoke_probe() {
    let paths =
        rurico::embed::download_model(rurico::embed::ModelId::default()).expect("download model");
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .env("__RURICO_PROBE_MODEL", &paths.model)
        .env("__RURICO_PROBE_CONFIG", &paths.config)
        .env("__RURICO_PROBE_TOKENIZER", &paths.tokenizer)
        .output()
        .expect("spawn probe");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("RURICO_PROBE_OK"),
        "probe handshake missing in stdout: {stdout:?}"
    );
    assert!(
        output.status.success(),
        "probe exited {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn smoke_command(paths: &rurico::embed::ModelPaths) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"));
    cmd.env("RURICO_SMOKE_MODEL", &paths.model)
        .env("RURICO_SMOKE_CONFIG", &paths.config)
        .env("RURICO_SMOKE_TOKENIZER", &paths.tokenizer);
    cmd
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
