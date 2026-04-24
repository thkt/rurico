//! Integration tests for `src/bin/gpu_pool_probe.rs` (Phase 3a).
//!
//! Spawns the probe subprocess and asserts on stderr banners and exit code.
//! Mirrors the pattern in `tests/mlx_smoke.rs` (subprocess isolation so an
//! MLX SIGABRT cannot kill the test runner).
//!
//! All tests are `#[ignore]` because they require the ruri-v3-310m model
//! cached locally and an MLX-capable runtime (Apple Silicon, unsandboxed).

use std::process::{Command, Output};

use rurico::sandbox::SEATBELT_SKIP_EXIT;

fn run_probe(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gpu_pool_probe"))
        .args(args)
        .output()
        .expect("spawn gpu_pool_probe")
}

fn skip_if_seatbelt(output: &Output) -> bool {
    output.status.code() == Some(SEATBELT_SKIP_EXIT)
}

// T-008 / FR-004 / AC-4
//
// [T-008] On a real W1 single-chunk input (seq_len = 8192), the probe must
// complete successfully and emit the three banner fields on stderr. The
// implementation is expected to pass the 10x margin on real weights; if the
// gate actually fails, Phase 3b is blocked per ADR 0002 sub-decision 3.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon) unsandboxed
fn t_008_probe_emits_max_abs_diff_cosine_sim_and_margin_banner() {
    let output = run_probe(&[]);
    if skip_if_seatbelt(&output) {
        return;
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "[T-008] probe must exit 0 on passing margin; got status={:?}, stderr={}",
        output.status,
        stderr
    );
    assert!(
        stderr.contains("max_abs_diff="),
        "[T-008] stderr must contain `max_abs_diff=` banner; got: {stderr}"
    );
    assert!(
        stderr.contains("cosine_sim="),
        "[T-008] stderr must contain `cosine_sim=` banner; got: {stderr}"
    );
    assert!(
        stderr.contains("margin_10x: PASS"),
        "[T-008] stderr must contain `margin_10x: PASS` on real-weight W1 \
         single chunk; got: {stderr}"
    );
}

// T-013 / FR-004a / AC-4
//
// [T-013] When the probe is run with `--force-fail`, the CPU reference is
// deliberately offset so the reported `max_abs_diff` exceeds the 10x margin.
// Exit code must be non-zero and stderr must contain `margin_10x: FAIL`.
// `--force-fail` is a synthetic perturbation path, not a Scope Cut Candidate
// — the 10x margin is designed to absorb honest f32 reduction order drift,
// so injecting drift into the real backend would be unreliable.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon) unsandboxed
fn t_013_probe_exits_nonzero_and_emits_fail_banner_on_margin_violation() {
    let output = run_probe(&["--force-fail"]);
    if skip_if_seatbelt(&output) {
        return;
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "[T-013] probe must exit non-zero on margin violation; got status={:?}, stderr={}",
        output.status,
        stderr
    );
    assert_ne!(
        output.status.code(),
        Some(0),
        "[T-013] exit code must be ≠ 0 on margin violation"
    );
    assert!(
        stderr.contains("margin_10x: FAIL"),
        "[T-013] stderr must contain `margin_10x: FAIL` banner; got: {stderr}"
    );
}
