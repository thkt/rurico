//! Integration tests that run smoke binaries in subprocesses.
//!
//! A SIGABRT from MLX FFI kills only the subprocess, not the test runner.

use std::process::{Command, Output};

use rurico::sandbox::SEATBELT_SKIP_EXIT;

#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn smoke_full() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .output()
        .expect("spawn smoke binary");
    assert_smoke_success(&output);
}

#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn smoke_verify_fixture() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("verify-fixture")
        .output()
        .expect("spawn mlx_smoke verify-fixture");
    assert_smoke_success(&output);
}

/// Success requires valid measurements and applicable primary gates, not diagnostic misses.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
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
    // Observe production readback telemetry, not a copy of output-shape arithmetic.
    for name in ["w1", "w2", "w3"] {
        let prefix = format!("readback_shape[{name}]:");
        let lines: Vec<_> = stderr
            .lines()
            .filter(|line| line.starts_with(&prefix))
            .collect();
        assert_eq!(
            lines.len(),
            3,
            "each warm trial must report readback: {stderr}"
        );
        for line in lines {
            let field = |key: &str| -> usize {
                line.split_whitespace()
                    .find_map(|token| token.strip_prefix(key)?.parse().ok())
                    .unwrap()
            };
            assert_eq!(field("total_flat="), field("expected_flat="), "{line}");
            assert_eq!(field("readback_count="), field("expected_count="), "{line}");
        }
    }
    // Diagnostics are conditional: an improvement meeting all goals must pass.
    // Synthetic binary tests cover tier routing without requiring a real miss.
    assert!(stderr.contains("W1/W3_speed=not_guaranteed"));
    assert!(
        stderr.contains("measure-baseline: primary thresholds passed"),
        "measure-baseline should end with the primary-thresholds-passed \
         banner (got: {stderr})"
    );
}

/// Re-exec the probe binary so probe environment handling is exercised.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn probe_embed_smoke_binary() {
    let output = Command::new(env!("CARGO_BIN_EXE_probe_embed_smoke"))
        .output()
        .expect("spawn probe_embed_smoke binary");
    assert_smoke_success(&output);
}

/// Re-exec the probe binary so probe environment handling is exercised.
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

/// Offline record replay must bypass model lookup and Metal initialization.
#[test]
fn summarize_records_cli_preserves_sub_ms_and_excludes_warmup() {
    use serde_json::{Value, json};
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw.jsonl");
    let records: Vec<_> = [("warmup", 99_999), ("warm", 100), ("warm", 300)]
        .into_iter()
        .enumerate()
        .map(|(sequence, (state, nanos))| {
            json!({
                "schema": 1, "context": {"synthetic": true}, "sequence": sequence,
                "workload": "synthetic", "input_sha256": null,
                "options": {"token_budget": null, "forward_pause": null},
                "method": "batch", "measured": false, "state": state,
                "repeat": null, "wall": {"secs": 0, "nanos": nanos}, "calls": [null]
            })
            .to_string()
        })
        .collect();
    fs::write(&path, records.join("\n")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("summarize-records")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["wall"]["n"], 2);
    assert_eq!(summary["wall"]["median"]["nanos"], 200);
    assert_eq!(summary["sequences"], json!([1, 2]));
    let bad = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("summarize-records")
        .arg(dir.path().join("absent.jsonl"))
        .output()
        .unwrap();
    assert!(!bad.status.success());
}

/// Reuse the short workload, options, parity and alternating order.
/// Production checks reject extra host accesses without a new fixture or latency target.
#[test]
#[ignore] // cached default model and unsandboxed Metal
fn smoke_measure_overhead_observes_readbacks() {
    use serde_json::Value;
    let output = Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
        .arg("measure-overhead")
        .output()
        .expect("spawn measure-overhead");
    assert_smoke_success(&output);
    let records: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|value: &Value| value.get("event").is_none())
        .collect();
    let measured: Vec<_> = records
        .iter()
        .filter(|r| r["measured"] == true && r["state"] == "warm")
        .collect();
    assert!(!measured.is_empty());
    for r in measured {
        for call in r["calls"].as_array().unwrap() {
            assert_eq!(
                call["readback_elements"].as_array().unwrap().len(),
                call["forwards"].as_array().unwrap().len()
            );
        }
    }
    let empty = records.iter().find(|r| r["workload"] == "empty").unwrap();
    assert_eq!(
        empty["calls"][0]["readback_elements"],
        serde_json::json!([])
    );
}

/// Exercise the offline dispatch and refusal without model lookup or a Metal
/// device. Numerical comparison cases stay in the smaller pure unit layer.
#[test]
fn compare_records_cli_reports_latency_separately_and_refuses_unknown_build() {
    use serde_json::{Value, json};
    use std::fs;
    use std::path::Path;
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.jsonl");
    let current = dir.path().join("current.jsonl");
    let mut context = json!({
        "commit": "base", "executable_sha256": "base-executable", "tracked_diff_sha256": "diff",
        "untracked_sha256": "untracked", "model": "model", "model_revision": "model-rev",
        "tokenizer_revision": "model-rev", "machine": "machine", "chip": "chip", "os": "os",
        "rustc": "rustc", "cargo": "cargo", "xcode": "xcode", "metal": "metal",
        "debug_assertions": false, "build_flags": "actual-Cargo-build-conditions", "build_invocation": "locked-release-smoke-v1",
        "lockfile_sha256": "lock", "os_cache_cleared": false, "inference_cache_policy": "cleanup"
    });
    let write = |path: &Path, context: &Value, scale: u64| {
        let lines: Vec<_> = [("batch", 100), ("sequential", 200)]
            .into_iter()
            .enumerate()
            .map(|(sequence, (method, ns))| {
                json!({"schema": 1, "context": context, "sequence": sequence, "workload": "w1",
                "input_sha256": "input", "options": {"token_budget": null, "forward_pause": null},
                "method": method, "measured": false, "state": "warm", "repeat": 0,
                "wall": {"secs": 0, "nanos": ns * scale}, "calls": [null]})
                .to_string()
            })
            .collect();
        fs::write(path, lines.join("\n")).unwrap();
    };
    write(&base, &context, 1);
    context["commit"] = json!("current");
    context["executable_sha256"] = json!("current-executable");
    write(&current, &context, 100);
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_mlx_smoke"))
            .arg("compare-records")
            .arg(&base)
            .arg(&current)
            .output()
            .unwrap()
    };
    let output = run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let values: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(values[0]["event"], "revision_latency");
    assert_eq!(values[0]["current_over_baseline"], 100.0);
    assert_eq!(values[2]["event"], "batch_sequential_efficiency");
    assert_eq!(values[2]["batch_over_sequential"], 0.5);
    assert_eq!(values[3]["batch_over_sequential"], 0.5);
    context["build_flags"] = Value::Null;
    write(&current, &context, 100);
    let output = run();
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "refused comparison must emit no partial success"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("build_flags unavailable"));
}
