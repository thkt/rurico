//! Eval harness binary (Phase 1d, Issue #65).
//!
//! Subprocess-isolated evaluator: capture / verify baselines, run known-answer
//! fixtures, and shuffle-mutate ranking for wiring sanity. Loaded behind the
//! `eval-harness` feature flag (added in Phase 1d GREEN by leader; until then
//! the binary still compiles via `src/bin/` auto-discovery, but every mode
//! body panics with `unimplemented!`, so subprocess tests fail — that is the
//! intended RED state).
//!
//! Argv shape (key=value, no `clap` dependency):
//!
//! ```text
//! eval_harness evaluate [kind=full|identity|reverse|single_doc|shuffled]
//! eval_harness capture-baseline output=<path>
//! eval_harness capture-reverse-baseline output=<path>
//! eval_harness verify-baseline baseline=<path>
//! ```
//!
//! `evaluate` defaults to `kind=full`. `kind=shuffled` re-uses the full
//! evaluation and shuffles the ranking with a fixed RNG seed before metric
//! computation (FR-014). Required keys (`output=`, `baseline=`) panic if
//! absent so leader's GREEN code can rely on them being present.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use rurico::eval::baseline::{BaselineSnapshot, build_metric_result, write_json};
use rurico::eval::fixture::{
    EvalDocument, EvalQuery, load_documents, load_known_answers, load_queries,
};
use rurico::eval::metrics::{MetricResult, bootstrap_ci, mrr_at_k, ndcg_at_k, recall_at_k};
use rurico::eval::pipeline::{PipelineConfig, QueryResult, evaluate as run_pipeline};
use rurico::sandbox::exit_if_seatbelt;
use rurico::{embed, model_probe, reranker};

const PIPELINE_K: usize = 10;
const SHUFFLE_SEED: u64 = 42;
const VERIFY_TOLERANCE: f64 = 1e-5;
const BOOTSTRAP_RESAMPLES: usize = 1000;
const BOOTSTRAP_SEED: u64 = 42;
const MLX_RS_VERSION: &str = "0.25";
/// Pinned ruri-v3-310m revision string. The HF commit hash lives in
/// `src/embed.rs::ModelId::revision` (private), so the harness pins a stable
/// label string here; Phase 1f may align with the upstream hash if needed.
const RURI_V3_310M_REVISION: &str = "pinned-via-rurico-embed-cache";

/// IR metric function signature shared by [`build_global_metrics`] and
/// [`build_one_metric`]; aliased to silence `clippy::type_complexity`.
type MetricFn = fn(&[String], &HashMap<String, u8>, usize) -> f64;

fn main() -> ExitCode {
    model_probe::handle_probe_if_needed();
    exit_if_seatbelt(env!("CARGO_BIN_NAME"));

    let args: Vec<String> = env::args().skip(1).collect();
    let Some(mode) = args.first() else {
        eprintln!(
            "usage: eval_harness <evaluate|capture-baseline|capture-reverse-baseline|\
             verify-baseline> [key=value...]"
        );
        return ExitCode::from(2);
    };
    let kvs: HashMap<String, String> = args[1..]
        .iter()
        .filter_map(|s| s.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
        .collect();

    match mode.as_str() {
        "evaluate" => run_evaluate(&kvs),
        "capture-baseline" => run_capture_baseline(&kvs),
        "capture-reverse-baseline" => run_capture_reverse_baseline(&kvs),
        "verify-baseline" => run_verify_baseline(&kvs),
        other => {
            eprintln!("unknown mode: {other}");
            ExitCode::from(2)
        }
    }
}

/// `evaluate kind=...` — run the reference pipeline against the chosen
/// fixture slice and print metric JSON to stdout.
fn run_evaluate(kvs: &HashMap<String, String>) -> ExitCode {
    let kind = kvs.get("kind").map_or("full", String::as_str);
    let (corpus, queries) = match load_fixture_for_kind(kind) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("evaluate({kind}): {msg}");
            return ExitCode::from(1);
        }
    };
    let embedder = match init_embedder() {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("evaluate({kind}): {msg}");
            return ExitCode::from(1);
        }
    };
    let reranker = match init_reranker() {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("evaluate({kind}): {msg}");
            return ExitCode::from(1);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let mut results = match run_pipeline(&corpus, &queries, &embedder, Some(&reranker), &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("evaluate({kind}): pipeline failed: {e}");
            return ExitCode::from(1);
        }
    };
    if kind == "shuffled" {
        shuffle_each_ranking(&mut results);
    }
    let summary = serde_json::json!({
        "kind": kind,
        "recall_at_1": global_metric(&results, &queries, recall_at_k, 1),
        "mrr": global_metric(&results, &queries, mrr_at_k, PIPELINE_K),
        "ndcg_at_10": global_metric(&results, &queries, ndcg_at_k, 10),
    });
    match serde_json::to_string_pretty(&summary) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("evaluate({kind}): serialise failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// `capture-baseline output=<path>` — run full evaluation + bootstrap CI and
/// write `BaselineSnapshot` to `output=`.
fn run_capture_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path) = kvs.get("output") else {
        eprintln!("capture-baseline: output= argument required");
        return ExitCode::from(2);
    };
    let (corpus, queries) = match load_fixture_for_kind("full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let embedder = match init_embedder() {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let reranker = match init_reranker() {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let results = match run_pipeline(&corpus, &queries, &embedder, Some(&reranker), &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture-baseline: pipeline failed: {e}");
            return ExitCode::from(1);
        }
    };

    let global = build_global_metrics(&results, &queries);
    let per_category = build_per_category_metrics(&results, &queries);
    let (latency_p50_ms, latency_p95_ms) = compute_latency_percentiles(&results);
    let snapshot = BaselineSnapshot {
        timestamp: iso_timestamp(),
        model_id: embed::ModelId::default().repo_id().to_owned(),
        model_revision: RURI_V3_310M_REVISION.to_owned(),
        mlx_rs_version: MLX_RS_VERSION.to_owned(),
        fixture_hash: hash_fixture_dir(),
        global,
        per_category,
        latency_p50_ms,
        latency_p95_ms,
    };

    if let Err(e) = write_json(&snapshot, Path::new(output_path)) {
        eprintln!("capture-baseline: write failed: {e}");
        return ExitCode::from(1);
    }
    eprintln!("capture-baseline: wrote {output_path}");
    ExitCode::SUCCESS
}

/// `capture-reverse-baseline output=<path>` — measure the reverse-ranker
/// `nDCG@10` lower bound and persist to `output=` so T-014 can pin it.
fn run_capture_reverse_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path) = kvs.get("output") else {
        eprintln!("capture-reverse-baseline: output= argument required");
        return ExitCode::from(2);
    };
    let (corpus, queries) = match load_fixture_for_kind("reverse") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let embedder = match init_embedder() {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let reranker = match init_reranker() {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let mut results = match run_pipeline(&corpus, &queries, &embedder, Some(&reranker), &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture-reverse-baseline: pipeline failed: {e}");
            return ExitCode::from(1);
        }
    };
    for r in &mut results {
        r.ranked_hits.reverse();
    }
    let observed_lower_bound = global_metric(&results, &queries, ndcg_at_k, 10);

    let payload = serde_json::json!({
        "kind": "reverse",
        "observed_lower_bound": observed_lower_bound,
        "k": 10,
        "captured_with": "eval_harness capture-reverse-baseline",
    });
    let json = match serde_json::to_string_pretty(&payload) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("capture-reverse-baseline: serialise failed: {e}");
            return ExitCode::from(1);
        }
    };
    if let Err(e) = fs::write(output_path, format!("{json}\n")) {
        eprintln!("capture-reverse-baseline: write failed: {e}");
        return ExitCode::from(1);
    }
    eprintln!(
        "capture-reverse-baseline: wrote {output_path} (observed_lower_bound={observed_lower_bound:.4})"
    );
    ExitCode::SUCCESS
}

/// `verify-baseline baseline=<path>` — re-run evaluation, compare against the
/// committed baseline.json under ADR 0002 tolerance, exit 0 + stderr banner
/// `verify-baseline: passed` on success (FR-017 / AC-5.3).
fn run_verify_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(baseline_path) = kvs.get("baseline") else {
        eprintln!("verify-baseline: baseline= argument required");
        return ExitCode::from(2);
    };
    let json = match fs::read_to_string(baseline_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify-baseline: read failed: {e}");
            return ExitCode::from(1);
        }
    };
    let committed: BaselineSnapshot = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify-baseline: parse failed: {e}");
            return ExitCode::from(1);
        }
    };
    let (corpus, queries) = match load_fixture_for_kind("full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let embedder = match init_embedder() {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let reranker = match init_reranker() {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(1);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let results = match run_pipeline(&corpus, &queries, &embedder, Some(&reranker), &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify-baseline: pipeline failed: {e}");
            return ExitCode::from(1);
        }
    };
    let current_global = build_global_metrics(&results, &queries);
    let by_name: HashMap<&str, &MetricResult> = current_global
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();

    for committed_m in &committed.global {
        let Some(current_m) = by_name.get(committed_m.name.as_str()) else {
            eprintln!(
                "verify-baseline: failed — committed metric {} missing in current run",
                committed_m.name
            );
            return ExitCode::from(1);
        };
        let diff = (committed_m.point_estimate - current_m.point_estimate).abs();
        if diff > VERIFY_TOLERANCE {
            eprintln!(
                "verify-baseline: failed — {} drifted by {diff:.6} (committed {:.6} vs current {:.6})",
                committed_m.name, committed_m.point_estimate, current_m.point_estimate
            );
            return ExitCode::from(1);
        }
    }
    eprintln!("verify-baseline: passed");
    ExitCode::SUCCESS
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval")
}

/// Load corpus + queries for `kind` from the fixture directory. `full` /
/// `shuffled` use `documents.jsonl` + `queries.jsonl`; the known-answer kinds
/// pull a sub-fixture from `known_answers.jsonl`.
fn load_fixture_for_kind(kind: &str) -> Result<(Vec<EvalDocument>, Vec<EvalQuery>), String> {
    let dir = fixture_dir();
    match kind {
        "full" | "shuffled" => {
            let docs = load_documents(&dir.join("documents.jsonl"))
                .map_err(|e| format!("load_documents: {e}"))?;
            let queries = load_queries(&dir.join("queries.jsonl"))
                .map_err(|e| format!("load_queries: {e}"))?;
            Ok((docs, queries))
        }
        "identity" => {
            let known = load_known_answers(&dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.identity.corpus, known.identity.queries))
        }
        "reverse" => {
            let known = load_known_answers(&dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.reverse.corpus, known.reverse.queries))
        }
        "single_doc" => {
            let known = load_known_answers(&dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.single_doc.corpus, known.single_doc.queries))
        }
        other => Err(format!("unknown kind: {other}")),
    }
}

fn init_embedder() -> Result<embed::Embedder, String> {
    let artifacts = embed::cached_artifacts(embed::ModelId::default())
        .map_err(|e| format!("embed cache lookup: {e}"))?
        .ok_or_else(|| "embed model not cached; run download first".to_owned())?;
    embed::Embedder::new(&artifacts).map_err(|e| format!("embedder load: {e}"))
}

fn init_reranker() -> Result<reranker::Reranker, String> {
    let artifacts = reranker::cached_artifacts(reranker::RerankerModelId::default())
        .map_err(|e| format!("reranker cache lookup: {e}"))?
        .ok_or_else(|| "reranker model not cached; run download first".to_owned())?;
    reranker::Reranker::new(&artifacts).map_err(|e| format!("reranker load: {e}"))
}

/// Shuffle every per-query ranking with a fixed seed so T-016 is deterministic.
fn shuffle_each_ranking(results: &mut [QueryResult]) {
    let mut rng = ChaCha8Rng::seed_from_u64(SHUFFLE_SEED);
    for r in results.iter_mut() {
        r.ranked_hits.shuffle(&mut rng);
    }
}

/// Mean of `metric_fn` across every (result, query) pair.
fn global_metric<F>(results: &[QueryResult], queries: &[EvalQuery], metric_fn: F, k: usize) -> f64
where
    F: Fn(&[String], &HashMap<String, u8>, usize) -> f64,
{
    let scores: Vec<f64> = results
        .iter()
        .zip(queries.iter())
        .map(|(r, q)| {
            let ranked: Vec<String> = r.ranked_hits.iter().map(|h| h.doc_id.clone()).collect();
            metric_fn(&ranked, &q.relevance_map, k)
        })
        .collect();
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// `[recall@5, recall@10, mrr@10, ndcg@10]` with bootstrap CI applied per metric.
fn build_global_metrics(results: &[QueryResult], queries: &[EvalQuery]) -> Vec<MetricResult> {
    let configs: &[(&str, usize, MetricFn)] = &[
        ("recall@5", 5, recall_at_k),
        ("recall@10", 10, recall_at_k),
        ("mrr@10", 10, mrr_at_k),
        ("ndcg@10", 10, ndcg_at_k),
    ];
    configs
        .iter()
        .map(|(name, k, metric)| build_one_metric(results, queries, name, *k, *metric))
        .collect()
}

fn build_one_metric(
    results: &[QueryResult],
    queries: &[EvalQuery],
    name: &str,
    k: usize,
    metric: MetricFn,
) -> MetricResult {
    let scores: Vec<f64> = results
        .iter()
        .zip(queries.iter())
        .map(|(r, q)| {
            let ranked: Vec<String> = r.ranked_hits.iter().map(|h| h.doc_id.clone()).collect();
            metric(&ranked, &q.relevance_map, k)
        })
        .collect();
    let mean = |xs: &[f64]| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f64>() / xs.len() as f64
        }
    };
    let (point, ci_lower, ci_upper) =
        bootstrap_ci(&scores, mean, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED);
    build_metric_result(name.to_owned(), k, point, ci_lower, ci_upper)
}

/// Group queries by category and compute the same metric set per group.
fn build_per_category_metrics(
    results: &[QueryResult],
    queries: &[EvalQuery],
) -> BTreeMap<String, Vec<MetricResult>> {
    let mut buckets: BTreeMap<String, (Vec<QueryResult>, Vec<EvalQuery>)> = BTreeMap::new();
    for (r, q) in results.iter().zip(queries.iter()) {
        let entry = buckets.entry(q.category.clone()).or_default();
        entry.0.push(r.clone());
        entry.1.push(q.clone());
    }
    buckets
        .into_iter()
        .map(|(cat, (rs, qs))| (cat, build_global_metrics(&rs, &qs)))
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn compute_latency_percentiles(results: &[QueryResult]) -> (f64, f64) {
    if results.is_empty() {
        return (0.0, 0.0);
    }
    let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable();
    let p50_idx = latencies.len() / 2;
    let p95_idx = ((latencies.len() * 95) / 100).min(latencies.len() - 1);
    (latencies[p50_idx] as f64, latencies[p95_idx] as f64)
}

/// Epoch-seconds string. Phase 1d trades strict ISO-8601 for keeping `chrono`
/// out of the dependency tree; Phase 1f may upgrade if downstream tooling needs
/// strict format.
fn iso_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

/// FNV-1a 64-bit hash over the three fixture JSONL files. Used as the
/// `fixture_hash` field on [`BaselineSnapshot`]; sha2 is intentionally avoided
/// to keep the dependency graph small (collision risk is acceptable for a
/// fixture-changed signal).
fn hash_fixture_dir() -> String {
    let dir = fixture_dir();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for name in ["documents.jsonl", "queries.jsonl", "known_answers.jsonl"] {
        if let Ok(content) = fs::read(dir.join(name)) {
            for byte in &content {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}
