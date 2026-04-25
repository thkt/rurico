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
//! - `smoke_measure_baseline`: shape check for `mlx_smoke measure-baseline`
//!   output that PR #6's SLA + linearity parser will consume.
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

/// End-to-end Phase 2E gate: T-WLD-001..006 SLA + padding + T-MET-003 R²
/// linearity are all enforced inside `mlx_smoke measure-baseline`, which
/// panics (non-zero exit) on any violation. This integration test therefore:
///
/// 1. Asserts the binary completed successfully (→ every threshold passed).
/// 2. Guards the stderr shape so downstream consumers (`phase2_result.md`
///    paste, future parsers) catch format drift before number drift.
///
/// Run time is ~4 minutes on Apple Silicon because each workload is timed
/// `MEASURE_REPEATS = 3` times to harden the median against single-run noise.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon); ~4 minute runtime
fn smoke_measure_baseline() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("measure-baseline")
        .output()
        .expect("spawn mlx_smoke measure-baseline");
    assert_smoke_success(&output);
    let stderr = String::from_utf8_lossy(&output.stderr);

    for name in ["baseline[w1]", "baseline[w2]", "baseline[w3]"] {
        assert!(
            stderr.contains(name),
            "measure-baseline stderr must contain {name} line (got: {stderr})"
        );
    }
    for field in [
        "padding_ratio=",
        "real_tokens=",
        "padded_tokens=",
        "forward_eval_ms=",
        "tokenize_ms=",
        "chunk_plan_ms=",
        "num_chunks=",
        "bucket_hist=[",
    ] {
        assert!(
            stderr.contains(field),
            "measure-baseline baseline[wN] line must carry `{field}` (got: {stderr})"
        );
    }
    for row in ["mdrow[w1]", "mdrow[w2]", "mdrow[w3]"] {
        assert!(
            stderr.contains(row),
            "measure-baseline must emit {row} line for phase2_result.md paste \
             (got: {stderr})"
        );
    }
    for tag in ["linearity ", "r_squared=", "slope=", "intercept="] {
        assert!(
            stderr.contains(tag),
            "measure-baseline linearity summary must carry `{tag}` (got: {stderr})"
        );
    }
    for res in ["residual[w1]", "residual[w2]", "residual[w3]"] {
        assert!(
            stderr.contains(res),
            "measure-baseline must emit per-workload {res} line (got: {stderr})"
        );
    }
    // T-006 / FR-002 / NFR-002 / AC-1
    //
    // [T-006] Phase 3b GPU pool emits per-workload `readback_shape[wN]:`
    // banner so this integration test confirms readback volume reduced
    // from `O(seq * hidden)` to `O(batch * hidden)`. Two-stage check:
    // (a) banner exists per workload (presence guard, mirrors the
    // `baseline[wN]` / `mdrow[wN]` / `residual[wN]` style), then
    // (b) the banner's `total_flat == total_rows * hidden_size` arithmetic
    // identity holds — this is the actual NFR-002 guarantee. The arithmetic
    // check survives format reordering and would catch a regression that
    // the prefix-only check would miss.
    for name in ["w1", "w2", "w3"] {
        let prefix = format!("readback_shape[{name}]:");
        let line = stderr
            .lines()
            .find(|l| l.contains(&prefix))
            .unwrap_or_else(|| {
                panic!("[T-006] missing {prefix} banner in measure-baseline stderr: {stderr}")
            });
        let parse_field = |key: &str| -> usize {
            line.split_whitespace()
                .find_map(|tok| tok.strip_prefix(key)?.parse::<usize>().ok())
                .unwrap_or_else(|| panic!("[T-006] {prefix} banner missing field '{key}': {line}"))
        };
        let hidden_size = parse_field("hidden_size=");
        let total_rows = parse_field("total_rows=");
        let total_flat = parse_field("total_flat=");
        assert_eq!(
            total_flat,
            total_rows * hidden_size,
            "[T-006] {prefix} NFR-002 invariant: total_flat must equal total_rows * hidden_size \
             (got total_flat={total_flat}, total_rows={total_rows}, hidden_size={hidden_size})"
        );
    }
    assert!(
        stderr.contains("saturated:"),
        "measure-baseline should surface W1 bucket-saturated diagnostics \
         per spec NFR-bucket-saturated; got: {stderr}"
    );
    assert!(
        stderr.contains("aspirational:"),
        "measure-baseline should surface aspirational-target diagnostics \
         (spec NFR-003/004-aspirational, Phase 3/5a gap); got: {stderr}"
    );
    assert!(
        stderr.contains("measure-baseline: primary thresholds passed"),
        "measure-baseline should end with the primary-thresholds-passed \
         banner (got: {stderr})"
    );
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
