//! Offline revision comparison of existing raw records; no model or Metal startup.
use serde_json::{Value, json};
use std::process;

use super::records::{Method, Record, distribution, read_file};

// Revision/executable/diff hashes intentionally differ. Everything determining
// input, model, device and build conditions must instead be available and equal.
const CONDITIONS: &[&str] = &[
    "model",
    "model_revision",
    "tokenizer_revision",
    "machine",
    "chip",
    "os",
    "rustc",
    "cargo",
    "xcode",
    "metal",
    "debug_assertions",
    "build_flags",
    "build_invocation",
    "lockfile_sha256",
    "os_cache_cleared",
    "inference_cache_policy",
];

fn same_group(a: &Record, b: &Record) -> bool {
    a.workload == b.workload
        && a.input_sha256 == b.input_sha256
        && a.options == b.options
        && a.method == b.method
        && a.measured == b.measured
}

fn groups(records: &[Record]) -> Result<Vec<Vec<&Record>>, String> {
    let mut groups: Vec<Vec<&Record>> = Vec::new();
    for r in records.iter().filter(|r| r.state == "warm") {
        if r.wall.is_zero()
            || r.input_sha256.as_ref().is_none_or(String::is_empty)
            || r.calls.is_empty()
            || (r.measured && r.calls.iter().any(Option::is_none))
        {
            return Err(format!(
                "{} sequence {}: missing or zero-resolution measurement/input",
                r.workload, r.sequence
            ));
        }
        if let Some(group) = groups.iter_mut().find(|g| same_group(g[0], r)) {
            // The first record's complete context has already been validated.
            // Equality lets later trials share that result, never their timings.
            if group[0].context != r.context {
                return Err(format!(
                    "{}: mixed contexts within one comparison group",
                    r.workload
                ));
            }
            group.push(r);
        } else {
            for key in CONDITIONS.iter().copied().chain([
                "commit",
                "executable_sha256",
                "tracked_diff_sha256",
                "untracked_sha256",
            ]) {
                let value = &r.context[key];
                if value.is_null() || value.as_str().is_some_and(str::is_empty) {
                    return Err(format!(
                        "{} sequence {}: comparison condition {key} unavailable",
                        r.workload, r.sequence
                    ));
                }
            }
            groups.push(vec![r]);
        }
    }
    if groups.is_empty() {
        return Err("no warm measurements".to_owned());
    }
    Ok(groups)
}

pub fn compare(base: &[Record], current: &[Record]) -> Result<Vec<Value>, String> {
    let base = groups(base)?;
    let current = groups(current)?;
    if base.len() != current.len() {
        return Err("missing comparison groups".to_owned());
    }
    let base_walls: Vec<_> = base
        .iter()
        .map(|g| distribution(g.iter().map(|r| r.wall).collect()))
        .collect();
    let current_walls: Vec<_> = current
        .iter()
        .map(|g| distribution(g.iter().map(|r| r.wall).collect()))
        .collect();
    let mut output = Vec::new();
    for (current_index, g) in current.iter().enumerate() {
        let c = g[0];
        let (base_index, b) = base.iter().enumerate().find(|(_, b)| same_group(b[0], c)).ok_or_else(|| {
            format!(
                "{}: input/workload/options/method/instrumentation mismatch or missing baseline",
                c.workload
            )
        })?;
        for key in CONDITIONS {
            if b[0].context[key] != c.context[key] {
                return Err(format!(
                    "{}: incompatible comparison condition {key}",
                    c.workload
                ));
            }
        }
        let base_wall = &base_walls[base_index];
        let current_wall = &current_walls[current_index];
        let ratio = current_wall.median.as_secs_f64() / base_wall.median.as_secs_f64();
        output.push(json!({
            "schema": 1, "event": "revision_latency", "workload": c.workload,
            "input_sha256": c.input_sha256, "options": c.options, "method": c.method,
            "measured": c.measured, "baseline_context": b[0].context, "current_context": c.context,
            "baseline_sequences": b.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            "current_sequences": g.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            "baseline_wall": base_wall, "current_wall": current_wall,
            "current_over_baseline": ratio, "median_increase": ratio > 1.0,
            "scope": "observed same-condition median; no absolute SLA or statistical speed guarantee",
            "limits": "model/tokenizer identified by pinned revisions; local artifact contents and background load not certified by raw records"
        }));
    }
    // Efficiency is separate from revision latency. Both can worsen while their
    // quotient stays unchanged; never use batch/sequential ratio as regression.
    for (revision, all, walls) in [
        ("baseline", &base, &base_walls),
        ("current", &current, &current_walls),
    ] {
        for (index, group) in all.iter().enumerate() {
            let r = group[0];
            let (other_index, _) = all
                .iter()
                .enumerate()
                .find(|(_, g)| {
                    let s = g[0];
                    s.method != r.method
                        && s.context == r.context
                        && s.workload == r.workload
                        && s.input_sha256 == r.input_sha256
                        && s.options == r.options
                        && s.measured == r.measured
                })
                .ok_or_else(|| {
                    let missing = match r.method {
                        Method::Batch => "sequential",
                        Method::Sequential => "batch",
                    };
                    format!("{}: missing comparable {missing} measurements", r.workload)
                })?;
            // Check both directions so a missing batch group cannot escape validation.
            if r.method != Method::Batch {
                continue;
            }
            let batch_wall = &walls[index];
            let sequential_wall = &walls[other_index];
            output.push(json!({"schema": 1, "event": "batch_sequential_efficiency", "revision": revision,
                "context": r.context, "workload": r.workload, "input_sha256": r.input_sha256,
                "options": r.options, "measured": r.measured,
                "batch_over_sequential": batch_wall.median.as_secs_f64() / sequential_wall.median.as_secs_f64(),
                "batch_wall": batch_wall, "sequential_wall": sequential_wall,
                "scope": "relative API efficiency; separate from revision latency; no absolute SLA"}));
        }
    }
    Ok(output)
}

pub fn compare_files(base: &str, current: &str) {
    let result = read_file(base).and_then(|b| read_file(current).and_then(|c| compare(&b, &c)));
    match result {
        Ok(values) => {
            for value in values {
                super::records::emit(&value);
            }
        }
        Err(reason) => {
            eprintln!("compare-records: indeterminate: {reason}; comparison refused");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rurico::embed::EmbedOptions;
    use std::time::Duration;

    fn records(commit: &str, batch: u64, sequential: u64) -> Vec<Record> {
        let mut context = json!({});
        for key in CONDITIONS.iter().copied().chain([
            "executable_sha256",
            "tracked_diff_sha256",
            "untracked_sha256",
        ]) {
            context[key] = json!("fixed");
        }
        context["commit"] = json!(commit);
        [Method::Batch, Method::Sequential]
            .into_iter()
            .map(|method| Record {
                schema: 1,
                context: context.clone(),
                sequence: if method == Method::Batch { 0 } else { 1 },
                workload: "w1".to_owned(),
                input_sha256: Some("input".to_owned()),
                options: EmbedOptions::default(),
                method,
                measured: false,
                state: "warm".to_owned(),
                repeat: Some(0),
                wall: Duration::from_nanos(if method == Method::Batch {
                    batch
                } else {
                    sequential
                }),
                calls: vec![None],
            })
            .collect()
    }

    #[test]
    fn slowdown_is_reported_even_when_efficiency_is_unchanged() {
        let base = records("base", 100, 200);
        let current = records("current", 10_000, 20_000);
        let values = compare(&base, &current).unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0]["current_over_baseline"].as_f64().unwrap() - 100.0).abs() < 1e-12);
        assert_eq!(values[1]["median_increase"], true);
        assert_eq!(values[2]["batch_over_sequential"], 0.5);
        assert_eq!(values[3]["batch_over_sequential"], 0.5);
        assert_eq!(base[0].wall.as_nanos(), 100);
    }

    #[test]
    fn incompatible_or_unavailable_conditions_refuse_comparison() {
        let base = records("base", 100, 200);
        for key in CONDITIONS {
            for replacement in [Value::Null, json!("different")] {
                let mut current = records("current", 100, 200);
                for r in &mut current {
                    r.context[key] = replacement.clone();
                }
                assert!(compare(&base, &current).unwrap_err().contains(key));
            }
        }
        let mut current = records("current", 100, 200);
        current[0].input_sha256 = Some("different".to_owned());
        assert!(
            compare(&base, &current)
                .unwrap_err()
                .contains("input/workload/options")
        );
        let mut current = records("current", 100, 200);
        current[0].options.token_budget = Some(256);
        assert!(
            compare(&base, &current)
                .unwrap_err()
                .contains("input/workload/options")
        );
        let mut current = records("current", 100, 200);
        current[0].wall = Duration::ZERO;
        assert!(
            compare(&base, &current)
                .unwrap_err()
                .contains("zero-resolution")
        );
        let mut current = records("current", 100, 200);
        current[0].measured = true;
        assert!(compare(&base, &current).unwrap_err().contains("missing"));
        assert!(compare(&base, &[]).unwrap_err().contains("no warm"));
    }

    #[test]
    fn repeated_trials_require_equal_contexts_and_valid_measurements() {
        let mut base = records("base", 100, 200);
        let mut later = records("base", 300, 400);
        for r in &mut later {
            r.sequence += 2;
            r.repeat = Some(1);
        }
        base.extend(later);
        let values = compare(&base, &base).unwrap();
        assert_eq!(values[0]["baseline_sequences"], json!([0, 2]));
        assert_eq!(values[0]["baseline_wall"]["n"], 2);
        assert_eq!(values[0]["baseline_wall"]["median"]["nanos"], 200);
        assert_eq!(values[1]["baseline_wall"]["median"]["nanos"], 300);

        for index in [2, 3] {
            for (key, value) in [
                ("build_flags", Value::Null),
                ("build_flags", json!("")),
                ("commit", json!("other")),
                ("extra_condition", json!("other")),
            ] {
                let mut current = base.clone();
                current[index].context[key] = value;
                assert!(
                    compare(&base, &current)
                        .unwrap_err()
                        .contains("mixed contexts")
                );
            }
            let mut current = base.clone();
            current[index].wall = Duration::ZERO;
            assert!(
                compare(&base, &current)
                    .unwrap_err()
                    .contains("zero-resolution")
            );
            let mut current = base.clone();
            current[index].input_sha256 = None;
            assert!(
                compare(&base, &current)
                    .unwrap_err()
                    .contains("measurement/input")
            );
            let mut current = base.clone();
            current[index].calls.clear();
            assert!(
                compare(&base, &current)
                    .unwrap_err()
                    .contains("measurement/input")
            );
        }
    }
    #[test]
    fn missing_either_method_refuses_comparison() {
        for (present, missing) in [(Method::Batch, "sequential"), (Method::Sequential, "batch")] {
            // Start each case with valid measurements; an unrelated telemetry
            // error must not masquerade as detection of the missing method.
            let mut base = records("base", 100, 200);
            let mut current = records("current", 100, 200);
            base.retain(|r| r.method == present);
            current.retain(|r| r.method == present);
            assert_eq!(
                compare(&base, &current).unwrap_err(),
                format!("w1: missing comparable {missing} measurements")
            );
        }
    }
}
