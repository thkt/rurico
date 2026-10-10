//! Host-only measurement of the fixed 50-pair public API workload.
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::time::Instant;

use rurico::reranker::{Reranker, RerankerModelId, cached_artifacts};
use rurico::sandbox::require_unsandboxed_mlx_runtime;
use serde_json::{Value, json};

fn percentile(samples: &[f64], index: usize) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[index]
}

fn main() {
    require_unsandboxed_mlx_runtime();
    let args: Vec<_> = env::args().skip(1).filter(|arg| arg != "--bench").collect();
    assert!(
        (2..=3).contains(&args.len()),
        "usage: reranker_latency CONTEXT.json OUTPUT.json [BASELINE.json]"
    );
    let context: Value = serde_json::from_slice(&fs::read(&args[0]).unwrap()).unwrap();
    for key in [
        "source",
        "machine",
        "chip",
        "os",
        "rust",
        "cargo",
        "xcode",
        "metal",
        "lockfile_sha256",
        "build",
        "load_conditions",
    ] {
        assert!(
            context[key].as_str().is_some_and(|s| !s.trim().is_empty()),
            "missing context: {key}"
        );
    }
    if cfg!(debug_assertions) {
        panic!("benchmark requires release build");
    }
    let model = RerankerModelId::default();
    let workload = json!({
        "repo": model.repo_id(), "revision": "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3",
        "pairs": 50, "query": "query", "document": "doc", "bucket_len": 128,
        "warmup": 3, "runs": 30, "method": "public score_batch wall microseconds excluding load",
    });
    let baseline = args.get(2).map(|path| {
        let base: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(base["format"], 1);
        assert_eq!(base["workload"], workload, "workload/model mismatch");
        let mut old = base["context"].clone();
        let mut current = context.clone();
        assert_ne!(
            old["source"], current["source"],
            "comparison needs distinct source versions"
        );
        old.as_object_mut().unwrap().remove("source");
        current.as_object_mut().unwrap().remove("source");
        assert_eq!(
            old, current,
            "comparison requires identical device/build/load conditions"
        );
        let samples: Vec<f64> = base["samples_us"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(samples.len(), 30);
        assert!(samples.iter().all(|s| s.is_finite() && *s > 0.0));
        (base["context"]["source"].clone(), samples)
    });
    // Refuse to replace prior evidence, before loading or measuring a model.
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])
        .unwrap();
    let start = Instant::now();
    let artifacts = cached_artifacts(model)
        .unwrap()
        .expect("fixed model must be cached; no download");
    let reranker = Reranker::new(&artifacts).unwrap();
    let load_us = start.elapsed().as_secs_f64() * 1e6;
    let pairs = vec![("query", "doc"); 50];
    let mut samples = Vec::with_capacity(30);
    let mut score_batch_api_calls = 0;
    for trial in 0..33 {
        let start = Instant::now();
        let scores = reranker.score_batch(&pairs).unwrap();
        score_batch_api_calls += 1;
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(scores.len(), 50);
        assert!(
            scores
                .iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s))
        );
        if trial >= 3 {
            samples.push(elapsed);
        }
    }
    assert!(samples.iter().all(|s| s.is_finite() && *s > 0.0));
    let p50 = percentile(&samples, 15);
    let p95 = percentile(&samples, 28);
    let comparison = baseline.map(|(source, old)| {
        json!({"baseline_source": source,
            "p50_current_over_baseline": p50 / percentile(&old, 15),
            "p95_current_over_baseline": p95 / percentile(&old, 28)})
    });
    let result = json!({"format": 1, "context": context, "workload": workload,
        "constructor_calls": 1, "score_batch_api_calls": score_batch_api_calls, "load_us": load_us,
        "samples_us": samples, "p50_us": p50, "p95_us": p95, "comparison": comparison});
    writeln!(output, "{}", serde_json::to_string_pretty(&result).unwrap()).unwrap();
}
