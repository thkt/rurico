//! MLX smoke-test binary — subprocess-isolated model verification.
//!
//! Invoked by integration tests via `Command` so MLX FFI crashes
//! (SIGABRT) are contained without killing the test runner.
//!
//! Loads the default embed model from the local HF Hub cache. The model
//! must be downloaded before running smoke tests.
//!
//! # Modes
//!
//! - default (no args): the legacy smoke assertions that other integration
//!   tests rely on. Running bare `mlx_smoke` keeps the contract with
//!   `tests/mlx_smoke.rs`.
//! - `capture-fixture`: run W1/W2/W3 and write the current-branch output to
//!   `tests/fixtures/phase2_baseline/w{1,2,3}.bin` so later PRs can compare
//!   bucket-batched output against today's main-branch baseline.
//! - `measure-baseline`: run W1/W2/W3 timing `embed_documents_batch` and a
//!   sequential equivalent, emitting one `baseline[wN] ...` line per workload
//!   so the numbers can be copied into `docs/benchmarks/phase1_baseline.md`.
//! - `measure-records`: default/nondefault options and ordinary/measured APIs,
//!   alternating batch/sequential runs with immutable JSONL records on stdout.
//! - `measure-overhead`: the same API/options comparisons on the first three W2
//!   documents, for short measurements with controlled background load.
//! - `summarize-records FILE`: replay JSONL summaries without loading a model.
//! - `compare-records BASE CURRENT`: refuse incompatible conditions and report
//!   revision latency separately from batch/sequential efficiency, offline.
//! - `verify-fixture`: run W1/W2/W3, load the committed fixtures, and assert
//!   numerical equivalence within Spec NFR-001 tolerances
//!   (`cosine_similarity ≥ 0.99999 AND max_abs_diff ≤ 1e-5`). Fails non-zero
//!   when any workload diverges; used by `tests/mlx_smoke.rs::smoke_verify_fixture`
//!   to drive T-BIT-001〜003.

#[path = "mlx_smoke/comparison.rs"]
mod comparison;
#[path = "mlx_smoke/records.rs"]
mod records;

use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, stderr};
use std::path::PathBuf;
use std::time::Instant;

use rurico::embed::{
    self, BatchMetrics, Embed,
    fixtures::{self, DEFAULT_COSINE_MIN, DEFAULT_MAX_ABS_DIFF},
    linreg::{linear_regression, r_squared},
    workloads::{workload_w1, workload_w2, workload_w3},
};
use rurico::handle_probe_if_needed;
use rurico::sandbox;

/// Minimal `tracing` subscriber that writes every record to stderr.
///
/// Kept inside the smoke binary so that debug logs emitted by the library are
/// visible when this binary runs, without forcing a subscriber on library
/// consumers.
fn init_tracing_subscriber() {
    if let Err(e) = tracing_subscriber::fmt()
        .with_writer(stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .try_init()
    {
        eprintln!("mlx_smoke: tracing subscriber init failed: {e}");
    }
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_default();
    if mode == "compare-records" {
        comparison::compare_files(
            &env::args().nth(2).expect("compare-records BASE CURRENT"),
            &env::args().nth(3).expect("compare-records BASE CURRENT"),
        );
        return;
    }
    if mode == "summarize-records" {
        records::summarize_file(&env::args().nth(2).expect("summarize-records JSONL"));
        return;
    }
    init_tracing_subscriber();

    // Also acts as a probe subprocess when probe env vars are set.
    handle_probe_if_needed();

    sandbox::exit_if_seatbelt(env!("CARGO_BIN_NAME"));

    let artifacts = embed::cached_artifacts(embed::ModelId::DEFAULT)
        .expect("cache lookup failed")
        .expect("model not cached; run download first");

    let context = matches!(
        mode.as_str(),
        "measure-baseline" | "measure-records" | "measure-overhead"
    )
    .then(records::context);
    let load_start = Instant::now();
    let embedder = embed::Embedder::new(&artifacts).expect("model load");
    let load_elapsed = load_start.elapsed();
    if let Some(context) = &context {
        records::emit_load(context, load_elapsed);
    }
    match mode.as_str() {
        "capture-fixture" => run_capture_fixture(&embedder),
        "measure-baseline" => run_measure_baseline(&embedder, context.as_ref().unwrap()),
        "measure-records" => {
            records::run(&embedder, context.as_ref().unwrap(), records::Mode::Records);
        }
        "measure-overhead" => {
            records::run(
                &embedder,
                context.as_ref().unwrap(),
                records::Mode::Overhead,
            );
        }
        "verify-fixture" => run_verify_fixture(&embedder),
        "" => run_assertions(&embedder),
        unknown => {
            eprintln!(
                "mlx_smoke: unknown mode {unknown:?} \
                 (known: capture-fixture, measure-baseline, measure-records, measure-overhead, summarize-records, compare-records, verify-fixture); \
                 running default assertions"
            );
            run_assertions(&embedder);
        }
    }
}

// ── Legacy smoke assertions ──────────────────────────────────────────────────

fn run_assertions(embedder: &embed::Embedder) {
    let dims = embedder.embedding_dims();

    let q = embedder.embed_query("authentication logic").expect("query");
    assert_eq!(q.len(), dims, "query dims");

    let q2 = embedder
        .embed_query("authentication logic")
        .expect("query2");
    assert_eq!(q, q2, "deterministic");

    let d = embedder
        .embed_document("function useAuth() { return user; }")
        .expect("short doc");
    assert_eq!(d.chunks().len(), 1, "short doc: 1 chunk");
    assert_eq!(d.chunks()[0].len(), dims, "short doc dims");

    let batch = embedder
        .embed_documents_batch(&[
            "function useAuth() { return user; }",
            "function Button() { return <div/>; }",
        ])
        .expect("batch");
    assert_eq!(batch.len(), 2, "batch count");

    // T-BKT-009: empty `texts` → `Vec::new()` (early-return branch regression guard)
    let empty = embedder.embed_documents_batch(&[]).expect("empty batch");
    assert!(empty.is_empty(), "empty texts must return Vec::new()");

    let sentence = "apple pie is a traditional dessert enjoyed around the world. ";
    let long_text = sentence.repeat(800);
    let ld = embedder.embed_document(&long_text).expect("long doc");
    assert!(ld.chunks().len() >= 2, "long doc: ≥2 chunks");
    for (i, chunk) in ld.chunks().iter().enumerate() {
        assert_eq!(chunk.len(), dims, "long doc chunk {i} dims");
    }

    for text in ["apple pie", "the cat", "Rust", "This is a test"] {
        let r = embedder.embed_document(text).expect(text);
        assert_eq!(r.chunks().len(), 1, "'{text}': 1 chunk");
        assert_eq!(r.chunks()[0].len(), dims, "'{text}' dims");
    }

    eprintln!("smoke: all checks passed");
}

// ── capture-fixture mode ─────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from("tests/fixtures/phase2_baseline")
}

fn as_refs(texts: &[String]) -> Vec<&str> {
    texts.iter().map(String::as_str).collect()
}

fn run_capture_fixture(embedder: &embed::Embedder) {
    let dir = fixture_dir();
    fs::create_dir_all(&dir).expect("create fixture dir");

    for (name, texts) in [
        ("w1", workload_w1()),
        ("w2", workload_w2()),
        ("w3", workload_w3()),
    ] {
        let refs = as_refs(&texts);
        let out = embedder
            .embed_documents_batch(&refs)
            .unwrap_or_else(|e| panic!("embed {name}: {e}"));
        let path = dir.join(format!("{name}.bin"));
        let file = File::create(&path).expect("create fixture file");
        let mut w = BufWriter::new(file);
        fixtures::save(&mut w, &out).expect("save fixture");
        eprintln!(
            "capture[{name}] wrote {} docs to {}",
            out.len(),
            path.display()
        );
    }
    eprintln!("capture-fixture: done");
}

// ── verify-fixture mode ──────────────────────────────────────────────────────

fn run_verify_fixture(embedder: &embed::Embedder) {
    let dir = fixture_dir();
    let mut failures = Vec::new();

    for (name, texts) in [
        ("w1", workload_w1()),
        ("w2", workload_w2()),
        ("w3", workload_w3()),
    ] {
        let refs = as_refs(&texts);
        let actual = embedder
            .embed_documents_batch(&refs)
            .unwrap_or_else(|e| panic!("embed {name}: {e}"));

        let path = dir.join(format!("{name}.bin"));
        let file =
            File::open(&path).unwrap_or_else(|e| panic!("open fixture {}: {e}", path.display()));
        let mut r = BufReader::new(file);
        let expected = fixtures::load(&mut r).unwrap_or_else(|e| panic!("load {name}: {e}"));

        match fixtures::compare(&expected, &actual) {
            Ok(diff) => {
                eprintln!(
                    "verify[{name}] cosine_min={:.6} max_abs_diff={:.3e} \
                     (thresholds: cos>={DEFAULT_COSINE_MIN}, diff<={DEFAULT_MAX_ABS_DIFF:.0e})",
                    diff.cosine_min, diff.max_abs_diff
                );
                if diff.cosine_min < DEFAULT_COSINE_MIN || diff.max_abs_diff > DEFAULT_MAX_ABS_DIFF
                {
                    failures.push(format!(
                        "{name}: cosine_min={:.6} max_abs_diff={:.3e} exceeds NFR-001",
                        diff.cosine_min, diff.max_abs_diff
                    ));
                }
            }
            Err(shape) => failures.push(format!("{name}: shape mismatch: {shape:?}")),
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("verify-fixture FAIL: {f}");
        }
        panic!("verify-fixture: {} workload(s) diverged", failures.len());
    }
    eprintln!("verify-fixture: all workloads match fixtures within NFR-001");
}

// ── measure-baseline mode ────────────────────────────────────────────────────

/// Batch/sequential efficiency, not revision regression or an absolute SLA.
/// Zero-resolution timings are indeterminate and must never imply achievement.
fn workload_ratio(batch_ms: u128, sequential_ms: u128) -> f64 {
    if batch_ms > 0 && sequential_ms > 0 {
        batch_ms as f64 / sequential_ms as f64
    } else {
        f64::NAN
    }
}

fn run_measure_baseline(embedder: &embed::Embedder, context: &serde_json::Value) {
    let run = records::run(embedder, context, records::Mode::Baseline);
    let records = &run.records;
    let mut results = Vec::new();
    let mut forward_medians = Vec::new();
    for name in ["w1", "w2", "w3"] {
        let batch: Vec<_> = records
            .iter()
            .filter(|r| {
                r.workload == name && r.state == "warm" && r.method == records::Method::Batch
            })
            .collect();
        let sequential: Vec<_> = records
            .iter()
            .filter(|r| {
                r.workload == name && r.state == "warm" && r.method == records::Method::Sequential
            })
            .collect();
        // Use the very same warm wall distributions already emitted as summaries.
        let batch_time = &run
            .summaries
            .iter()
            .find(|s| s.matches(batch[0]))
            .expect("batch warm summary")
            .wall;
        let sequential_time = &run
            .summaries
            .iter()
            .find(|s| s.matches(sequential[0]))
            .expect("sequential warm summary")
            .wall;
        let forward = records::distribution(
            batch
                .iter()
                .map(|r| r.calls[0].as_ref().unwrap().forward_eval)
                .collect(),
        );
        for record in batch.iter().chain(&sequential) {
            assert!(
                !record.wall.is_zero(),
                "measurement indeterminate[{name}]: zero-resolution wall"
            );
            assert!(
                !record.calls.is_empty(),
                "measurement indeterminate[{name}]: missing calls"
            );
            for call in &record.calls {
                let m = call
                    .as_ref()
                    .expect("measurement indeterminate: missing MLX telemetry");
                assert!(
                    !m.forward_eval.is_zero(),
                    "measurement indeterminate[{name}]: zero-resolution forward_eval"
                );
                assert!(
                    m.real_tokens > 0,
                    "measurement indeterminate[{name}]: missing real tokens"
                );
            }
        }
        let m = batch[0].calls[0].as_ref().unwrap();
        let ratio = workload_ratio(
            batch_time.median.as_nanos(),
            sequential_time.median.as_nanos(),
        );
        let padding = m.padded_tokens as f32 / m.real_tokens as f32;
        let fw = forward.median.as_secs_f64() * 1000.0;
        forward_medians.push(fw);
        eprintln!(
            "baseline[{name}] summary=true n={} batch_ms={:.6} sequential_ms={:.6} ratio={ratio:.3} padding_ratio={padding:.3} real_tokens={} padded_tokens={} forward_eval_ms={fw:.6} tokenize_ms=unmeasured chunk_plan_ms=unmeasured num_chunks={} bucket_hist={:?}",
            batch_time.n,
            batch_time.median.as_secs_f64() * 1000.0,
            sequential_time.median.as_secs_f64() * 1000.0,
            m.real_tokens,
            m.padded_tokens,
            m.num_chunks,
            m.bucket_hist
        );
        eprintln!(
            "mdrow[{name}] summary | {name} | unmeasured | unmeasured | {fw:.6} | {padding:.3} | {} | {:?} |",
            m.num_chunks, m.bucket_hist
        );
        let hs = embedder.embedding_dims();
        for (trial, record) in batch.iter().enumerate() {
            let measured = record.calls[0].as_ref().unwrap();
            let accesses = measured.readback_elements.as_ref().unwrap();
            eprintln!(
                "readback_shape[{name}]: trial={trial} hidden_size={hs} total_rows={} total_flat={} readback_count={} expected_count={} expected_flat={}",
                measured.num_chunks,
                accesses.iter().sum::<usize>(),
                accesses.len(),
                measured.forwards.len(),
                measured.num_chunks * hs
            );
        }
        eprintln!(
            "scope[{name}] batch_sequential_efficiency_primary={} padding_primary={} absolute_latency_sla=not_defined revision_regression=unmeasured (use compare-records BASE CURRENT)",
            is_sla_amenable(&m.bucket_hist),
            !is_bucket_saturated(&m.bucket_hist)
        );
        // Shape-only data for existing threshold checks. Timing aggregates stay
        // separate from per-invocation snapshots in the immutable raw records.
        results.push(WorkloadResult {
            name,
            ratio,
            metrics: BatchMetrics {
                padding_ratio: padding,
                real_tokens: m.real_tokens,
                padded_tokens: m.padded_tokens,
                num_chunks: m.num_chunks,
                bucket_hist: m.bucket_hist,
                ..BatchMetrics::default()
            },
        });
    }

    // Fit `forward_eval_ms = slope * real_tokens + intercept` over the three
    // workload points and emit R² so NFR-005 (≥ 0.95) can be asserted.
    let xs: Vec<f64> = results
        .iter()
        .map(|r| r.metrics.real_tokens as f64)
        .collect();
    let ys = forward_medians;
    let (slope, intercept, r2) = fit(&xs, &ys);
    eprintln!(
        "linearity scope=scale_fit_only speed_guarantee=false; non-finite/constant/zero-resolution fit is indeterminate"
    );
    eprintln!("linearity slope={slope:.6} intercept={intercept:.3} r_squared={r2:.4}");
    for (r, (x, y)) in results.iter().zip(xs.iter().zip(ys.iter())) {
        let predicted = slope * x + intercept;
        let residual = y - predicted;
        eprintln!(
            "residual[{name}] real_tokens={rt} forward_eval_ms={fw} predicted={predicted:.3} \
             residual={residual:.3}",
            name = r.name,
            rt = r.metrics.real_tokens,
            fw = y,
        );
    }

    let report = check_thresholds(&results, r2);
    // Deliberately separate log prefixes so phase2_result.md / downstream
    // parsers can distinguish the three tiers — see spec NFR-003/NFR-004
    // primary vs aspirational split and `is_bucket_saturated`.
    for v in &report.saturated_informational {
        eprintln!("saturated: {v:?}");
    }
    for v in &report.aspirational_diagnostics {
        eprintln!("aspirational: {v:?}");
    }
    if !report.primary_violations.is_empty() {
        for v in &report.primary_violations {
            eprintln!("primary violation: {v:?}");
        }
        panic!(
            "measure-baseline: {} primary threshold violation(s); see stderr",
            report.primary_violations.len()
        );
    }
    eprintln!(
        "measure-baseline: primary thresholds passed ({} aspirational diagnostic(s), \
         {} saturated diagnostic(s)); scope=eligible_efficiency_padding_and_scale_fit; W1/W3_speed=not_guaranteed; all_workload_targets=not_guaranteed",
        report.aspirational_diagnostics.len(),
        report.saturated_informational.len(),
    );
}

fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    if xs.len() < 2
        || xs.len() != ys.len()
        || xs.iter().chain(ys).any(|v| !v.is_finite() || *v <= 0.0)
        || xs.iter().all(|x| *x == xs[0])
        || ys.iter().all(|y| *y == ys[0])
    {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let (slope, intercept) = linear_regression(xs, ys);
    (slope, intercept, r_squared(xs, ys, slope, intercept))
}

// ── Phase 2E: threshold checking (SLA + padding + R²) ────────────────────────

struct WorkloadResult {
    name: &'static str,
    /// Computed once via [`workload_ratio`] when the workload is measured;
    /// `check_thresholds` reads this field instead of re-deriving the ratio,
    /// so the displayed and gated values cannot diverge.
    ratio: f64,
    metrics: BatchMetrics,
}

#[derive(Debug, PartialEq)]
enum Violation {
    Sla {
        workload: &'static str,
        actual: f64,
        threshold: f64,
    },
    Padding {
        workload: &'static str,
        actual: f32,
        threshold: f32,
    },
    Indeterminate {
        workload: &'static str,
        metric: &'static str,
        reason: &'static str,
    },
    RSquared {
        actual: f64,
        threshold: f64,
    },
}

/// Primary SLA threshold — SOW Why's direct translation: "batch is at least
/// as fast as sequential, so batch API keeps its value." NFR-004-primary.
const PRIMARY_SLA_THRESHOLD: f64 = 1.00;
/// Aspirational SLA target — original SOW number. Phase 3/5a goal.
/// NFR-004-aspirational.
const ASPIRATIONAL_SLA_THRESHOLD: f64 = 0.80;
/// Primary padding threshold — observational floor for bucket-amenable
/// workloads, reflecting the irreducible padding from length variance inside
/// a single bucket. NFR-003-primary.
const PRIMARY_PADDING_THRESHOLD: f32 = 1.20;
/// Aspirational padding target — original SOW number. Phase 3/5a goal.
/// NFR-003-aspirational.
const ASPIRATIONAL_PADDING_THRESHOLD: f32 = 1.10;
const R_SQUARED_THRESHOLD: f64 = 0.95;

/// A workload is bucket-saturated when every chunk lands in a single bucket
/// whose max_seq_len is material (index ≥ 2 = seq_len > 512). Under that
/// shape, bucket batching degenerates to a single sub-batch whose padding
/// waste is bound by the bucket ceiling — the same state Phase 1 was in.
/// SOW Why ("chunk 長分布に依らず") does not cover this regime; Phase 3/5a
/// (GPU pool / mutex scope) are the designated improvement points.
///
/// Buckets 0 and 1 (`max_seq_len ≤ 512`) are excluded because their padding
/// overhead is already negligible — W2's `[100,0,0,0]` shape sits at
/// `padding_ratio ≈ 1.005` in practice, so classifying it as saturated would
/// hide a genuine bucket batching win.
fn is_bucket_saturated(bucket_hist: &[usize; 4]) -> bool {
    let non_empty = bucket_hist.iter().filter(|&&n| n > 0).count();
    non_empty == 1 && bucket_hist[2..].iter().any(|&n| n > 0)
}

/// A workload is SLA-amenable when every chunk sits in a short bucket
/// (index 0 or 1, `max_seq_len ≤ 512`). Under that shape the MLX kernel
/// compile cost and Metal scheduler noise are small enough relative to the
/// wall-clock that the median-of-3 `batch_ms / sequential_ms` ratio is
/// stable across runs and can carry a single-run primary SLA assertion.
///
/// Workloads with any chunk in bucket index ≥ 2 (like W3's `[5,0,5,0]`)
/// run both batched and sequential passes through two distinct kernels,
/// and the empirical run-to-run variance is ~10% — enough to flip a
/// `ratio ≤ 1.0` assertion between pass and fail on successive runs. For
/// those, the primary SLA assertion is skipped; the aspirational threshold
/// still fires as a diagnostic. `is_bucket_saturated` is a strict subset of
/// `!is_sla_amenable`.
fn is_sla_amenable(bucket_hist: &[usize; 4]) -> bool {
    bucket_hist[2..].iter().all(|&n| n == 0)
}

#[derive(Debug, Default, PartialEq)]
struct ThresholdReport {
    /// Primary enforced failures on bucket-amenable workloads — cause panic.
    /// Bound to SOW Why "batch ≥ sequential" (ratio ≤ 1.0) and the bucket
    /// observational floor (padding ≤ 1.20), plus the global R² check.
    primary_violations: Vec<Violation>,
    /// Aspirational-target misses on bucket-amenable workloads — diagnostic
    /// only. Surface the gap toward Phase 3/5a improvement targets without
    /// blocking Phase 2 merge.
    aspirational_diagnostics: Vec<Violation>,
    /// Bucket-saturated workloads — out of Phase 2 scope. All deviations
    /// against the aspirational threshold land here so regressions on the
    /// diagnostic numbers are still visible after Phase 3/5a work.
    saturated_informational: Vec<Violation>,
}

#[cfg(test)]
impl ThresholdReport {
    fn is_clean(&self) -> bool {
        self.primary_violations.is_empty()
            && self.aspirational_diagnostics.is_empty()
            && self.saturated_informational.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
enum Tier {
    Primary,
    Aspirational,
    Saturated,
}

/// Route a single metric reading against the 3-tier thresholds. Returns
/// `Some((tier, threshold))` if `value` exceeds the tier's bound, `None` if
/// it sits within every bound. `primary_enforced=false` on an amenable
/// workload skips the primary gate entirely and routes over-threshold
/// values directly to aspirational — used by SLA on padding-only-amenable
/// workloads where kernel-compile variance makes the primary assertion
/// unstable. Saturated workloads always evaluate only against
/// `aspirational` and route to the saturated bucket.
fn classify_deviation<T: PartialOrd + Copy>(
    value: T,
    primary: T,
    aspirational: T,
    saturated: bool,
    primary_enforced: bool,
) -> Option<(Tier, T)> {
    if saturated {
        (value > aspirational).then_some((Tier::Saturated, aspirational))
    } else if primary_enforced && value > primary {
        Some((Tier::Primary, primary))
    } else if value > aspirational {
        Some((Tier::Aspirational, aspirational))
    } else {
        None
    }
}

fn push_tier(report: &mut ThresholdReport, tier: Tier, v: Violation) {
    match tier {
        Tier::Primary => report.primary_violations.push(v),
        Tier::Aspirational => report.aspirational_diagnostics.push(v),
        Tier::Saturated => report.saturated_informational.push(v),
    }
}

fn check_thresholds(results: &[WorkloadResult], r2: f64) -> ThresholdReport {
    let mut report = ThresholdReport::default();
    if results.is_empty() {
        report.primary_violations.push(Violation::Indeterminate {
            workload: "all",
            metric: "measurements",
            reason: "missing workloads",
        });
    }
    for r in results {
        if !r.ratio.is_finite()
            || r.ratio <= 0.0
            || !r.metrics.padding_ratio.is_finite()
            || r.metrics.padding_ratio < 1.0
        {
            report.primary_violations.push(Violation::Indeterminate {
                workload: r.name,
                metric: "ratio/padding",
                reason: "non-finite, zero-resolution or missing measurement",
            });
            continue;
        }
        let saturated = is_bucket_saturated(&r.metrics.bucket_hist);
        let sla_amenable = is_sla_amenable(&r.metrics.bucket_hist);
        let ratio = r.ratio;
        let padding = r.metrics.padding_ratio;

        if let Some((tier, threshold)) = classify_deviation(
            ratio,
            PRIMARY_SLA_THRESHOLD,
            ASPIRATIONAL_SLA_THRESHOLD,
            saturated,
            sla_amenable,
        ) {
            push_tier(
                &mut report,
                tier,
                Violation::Sla {
                    workload: r.name,
                    actual: ratio,
                    threshold,
                },
            );
        }
        if let Some((tier, threshold)) = classify_deviation(
            padding,
            PRIMARY_PADDING_THRESHOLD,
            ASPIRATIONAL_PADDING_THRESHOLD,
            saturated,
            !saturated,
        ) {
            push_tier(
                &mut report,
                tier,
                Violation::Padding {
                    workload: r.name,
                    actual: padding,
                    threshold,
                },
            );
        }
    }
    if !r2.is_finite() {
        report.primary_violations.push(Violation::Indeterminate {
            workload: "all",
            metric: "r_squared",
            reason: "non-finite or missing fit",
        });
    } else if r2 < R_SQUARED_THRESHOLD {
        report.primary_violations.push(Violation::RSquared {
            actual: r2,
            threshold: R_SQUARED_THRESHOLD,
        });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_EPS: f64 = 1e-9;

    // Bucket-hist archetypes used by the 3-workload tests, matching the
    // `measure-baseline` observations so `is_bucket_saturated` and
    // `is_sla_amenable` classify each fixture correctly.
    const BUCKET_SATURATED: [usize; 4] = [0, 0, 0, 3];
    const BUCKET_SHORT_ONLY: [usize; 4] = [100, 0, 0, 0];
    const BUCKET_MIXED: [usize; 4] = [5, 0, 5, 0];

    fn mk_result(
        name: &'static str,
        batch_ms: u128,
        sequential_ms: u128,
        padding_ratio: f32,
        bucket_hist: [usize; 4],
    ) -> WorkloadResult {
        WorkloadResult {
            name,
            ratio: workload_ratio(batch_ms, sequential_ms),
            metrics: BatchMetrics {
                padding_ratio,
                real_tokens: 1000,
                padded_tokens: 0,
                forward_eval_ms: 1000,
                num_chunks: 0,
                bucket_hist,
                max_seq_len: 0,
                batch_size: 0,
                tokenize_ms: 0,
                chunk_plan_ms: 0,
            },
        }
    }

    // T-WLD-001..006 + T-MET-003 happy path: every workload within every
    // threshold — primary, aspirational, and saturated buckets all empty.
    #[test]
    fn check_thresholds_all_within_limits_returns_empty() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(report.is_clean(), "expected clean report, got {report:?}");
    }

    // T-WLD-001 + T-WLD-004: W1 is bucket-saturated, so any deviation from
    // the aspirational threshold lands in `saturated_informational` and does
    // not trigger a primary violation. Expect two saturated entries (Sla +
    // Padding) and an otherwise-empty report.
    #[test]
    fn check_thresholds_flags_w1_deviation_as_saturated() {
        let results = [
            mk_result("w1", 900, 1000, 1.5, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(
            report.primary_violations.is_empty(),
            "W1 must not trigger primary violations: {:?}",
            report.primary_violations
        );
        assert!(
            report.aspirational_diagnostics.is_empty(),
            "saturated workloads do not populate aspirational: {:?}",
            report.aspirational_diagnostics
        );
        assert_eq!(
            report.saturated_informational.len(),
            2,
            "expected saturated Sla + Padding, got {:?}",
            report.saturated_informational
        );
        assert!(
            report
                .saturated_informational
                .iter()
                .any(|v| matches!(v, Violation::Sla { workload: "w1", .. })),
            "missing saturated Sla for w1 in {:?}",
            report.saturated_informational
        );
        assert!(
            report
                .saturated_informational
                .iter()
                .any(|v| matches!(v, Violation::Padding { workload: "w1", .. })),
            "missing saturated Padding for w1 in {:?}",
            report.saturated_informational
        );
    }

    // T-WLD-006: W3 is bucket-amenable; a padding overshoot of the PRIMARY
    // 1.20 floor must land in `primary_violations` and *not* double-report
    // as aspirational — the primary violation already implies the
    // aspirational gap.
    #[test]
    fn check_thresholds_flags_w3_padding_primary_violation() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.5, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert_eq!(
            report.primary_violations.len(),
            1,
            "expected 1 primary violation, got {:?}",
            report.primary_violations
        );
        assert!(
            report.aspirational_diagnostics.is_empty(),
            "primary violation should not re-emit as aspirational: {:?}",
            report.aspirational_diagnostics
        );
        assert!(
            report.saturated_informational.is_empty(),
            "W3 is amenable, saturated bucket must stay empty: {:?}",
            report.saturated_informational
        );
        match &report.primary_violations[0] {
            Violation::Padding {
                workload,
                actual,
                threshold,
            } => {
                assert_eq!(*workload, "w3");
                assert!(
                    (*actual - 1.5).abs() < f32::EPSILON,
                    "expected actual ≈ 1.5, got {actual:?}"
                );
                assert!(
                    (*threshold - 1.20).abs() < f32::EPSILON,
                    "expected threshold 1.20, got {threshold:?}"
                );
            }
            other => panic!("expected Violation::Padding, got {other:?}"),
        }
    }

    // T-WLD-006 aspirational only: W3 padding between the aspirational
    // 1.10 and primary 1.20 gates must surface as aspirational diagnostic,
    // not a primary violation — Phase 2 passes, Phase 3/5a goal missed.
    #[test]
    fn check_thresholds_flags_w3_padding_aspirational_diagnostic() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.15, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(
            report.primary_violations.is_empty(),
            "within primary floor, expected no primary: {:?}",
            report.primary_violations
        );
        assert_eq!(
            report.aspirational_diagnostics.len(),
            1,
            "expected 1 aspirational padding diagnostic, got {:?}",
            report.aspirational_diagnostics
        );
        match &report.aspirational_diagnostics[0] {
            Violation::Padding {
                workload,
                threshold,
                ..
            } => {
                assert_eq!(*workload, "w3");
                assert!(
                    (*threshold - 1.10).abs() < f32::EPSILON,
                    "expected aspirational threshold 1.10, got {threshold:?}"
                );
            }
            other => panic!("expected Violation::Padding, got {other:?}"),
        }
    }

    // T-WLD-003: W3 is bucket-amenable for padding but NOT sla-amenable
    // (bucket_hist=[5,0,5,0] hits bucket 2). Ratio deviations must land
    // only in aspirational_diagnostics — the primary SLA gate is skipped
    // because kernel-compile variance across the two shapes makes a
    // single-run ratio assertion flip between pass and fail.
    #[test]
    fn check_thresholds_flags_w3_ratio_as_aspirational_only() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            // ratio 1.08 — past primary 1.0, but must not trigger a primary
            // violation because W3 is not sla-amenable.
            mk_result("w3", 1080, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(
            report.primary_violations.is_empty(),
            "W3 SLA must not be primary-enforced (not sla-amenable): {:?}",
            report.primary_violations
        );
        assert_eq!(
            report.aspirational_diagnostics.len(),
            1,
            "expected W3 Sla in aspirational, got {:?}",
            report.aspirational_diagnostics
        );
        match &report.aspirational_diagnostics[0] {
            Violation::Sla {
                workload,
                actual,
                threshold,
            } => {
                assert_eq!(*workload, "w3");
                assert!(
                    (actual - 1.08).abs() < FLOAT_EPS,
                    "expected actual ≈ 1.08, got {actual:?}"
                );
                assert!(
                    (threshold - 0.80).abs() < FLOAT_EPS,
                    "expected aspirational threshold 0.80, got {threshold:?}"
                );
            }
            other => panic!("expected Violation::Sla for w3, got {other:?}"),
        }
    }

    // T-MET-003: R² is workload-independent and only emits one tier; a
    // sub-threshold value is always a primary violation.
    #[test]
    fn check_thresholds_flags_r_squared_primary_violation() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.90);
        assert_eq!(
            report.primary_violations.len(),
            1,
            "expected 1 primary violation, got {:?}",
            report.primary_violations
        );
        assert!(
            report.aspirational_diagnostics.is_empty() && report.saturated_informational.is_empty(),
            "R² emits one tier only, got report {report:?}"
        );
        match &report.primary_violations[0] {
            Violation::RSquared { actual, threshold } => {
                assert!(
                    (actual - 0.90).abs() < FLOAT_EPS,
                    "expected actual ≈ 0.90, got {actual:?}"
                );
                assert!(
                    (threshold - 0.95).abs() < FLOAT_EPS,
                    "expected threshold 0.95, got {threshold:?}"
                );
            }
            other => panic!("expected Violation::RSquared, got {other:?}"),
        }
    }

    // T-WLD-002: W2 at ratio 0.85 sits between aspirational (0.80) and
    // primary (1.00) — must land only in aspirational_diagnostics. Phase 2
    // passes, Phase 3/5a target not yet met.
    #[test]
    fn check_thresholds_flags_w2_slow_ratio_as_aspirational() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 850, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(
            report.primary_violations.is_empty(),
            "ratio 0.85 is within primary 1.0, got {:?}",
            report.primary_violations
        );
        assert_eq!(
            report.aspirational_diagnostics.len(),
            1,
            "expected 1 aspirational Sla, got {:?}",
            report.aspirational_diagnostics
        );
        match &report.aspirational_diagnostics[0] {
            Violation::Sla {
                workload,
                actual,
                threshold,
            } => {
                assert_eq!(*workload, "w2");
                assert!(
                    (actual - 0.85).abs() < FLOAT_EPS,
                    "expected actual ≈ 0.85, got {actual:?}"
                );
                assert!(
                    (threshold - 0.80).abs() < FLOAT_EPS,
                    "expected aspirational threshold 0.80, got {threshold:?}"
                );
            }
            other => panic!("expected Violation::Sla for w2, got {other:?}"),
        }
    }

    // Combined: W1 saturated (2 entries) + W3 primary padding + R² primary
    // must each go to the right bucket without leaking into aspirational —
    // the bucket-routing invariant for the whole reporter.
    #[test]
    fn check_thresholds_splits_all_three_tiers() {
        let results = [
            mk_result("w1", 900, 1000, 1.5, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.5, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.90);
        assert_eq!(
            report.primary_violations.len(),
            2,
            "expected primary (W3 Padding + R²), got {:?}",
            report.primary_violations
        );
        assert!(
            report.aspirational_diagnostics.is_empty(),
            "primary catches should not leak to aspirational: {:?}",
            report.aspirational_diagnostics
        );
        assert_eq!(
            report.saturated_informational.len(),
            2,
            "expected W1 Sla + Padding in saturated, got {:?}",
            report.saturated_informational
        );
        assert!(
            report
                .primary_violations
                .iter()
                .any(|v| matches!(v, Violation::Padding { workload: "w3", .. })),
            "missing primary Padding for w3: {:?}",
            report.primary_violations
        );
        assert!(
            report
                .primary_violations
                .iter()
                .any(|v| matches!(v, Violation::RSquared { .. })),
            "missing primary RSquared: {:?}",
            report.primary_violations
        );
    }

    #[test]
    fn workload_ratio_divides_batch_by_sequential() {
        assert!((workload_ratio(500, 1000) - 0.5).abs() < FLOAT_EPS);
        assert!((workload_ratio(1080, 1000) - 1.08).abs() < FLOAT_EPS);
    }

    #[test]
    fn workload_ratio_rejects_zero_resolution() {
        for (batch, sequential) in [(500, 0), (0, 0), (0, 500)] {
            assert!(workload_ratio(batch, sequential).is_nan());
        }
    }

    // A zero denominator used to appear as achieved ratio=0. It now fails
    // with an indeterminate diagnostic, independently of the numeric SLA tier.
    #[test]
    fn check_thresholds_zero_sequential_is_indeterminate() {
        let results = [
            mk_result("w1", 500, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 0, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 500, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(matches!(
            report.primary_violations.as_slice(),
            [Violation::Indeterminate { workload: "w2", .. }]
        ));
    }

    #[test]
    fn invalid_measurements_fail_even_outside_primary_speed_scope() {
        for (invalid, invalid_padding) in [
            (f64::NAN, f32::NAN),
            (f64::INFINITY, f32::INFINITY),
            (f64::NEG_INFINITY, f32::NEG_INFINITY),
        ] {
            for hist in [BUCKET_SATURATED, BUCKET_SHORT_ONLY, BUCKET_MIXED] {
                let mut r = mk_result("w1", 500, 1000, 1.05, hist);
                r.ratio = invalid;
                assert!(!check_thresholds(&[r], 0.98).primary_violations.is_empty());
                let r = mk_result("w1", 500, 1000, invalid_padding, hist);
                assert!(!check_thresholds(&[r], 0.98).primary_violations.is_empty());
            }
            assert!(
                !check_thresholds(
                    &[mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY)],
                    invalid
                )
                .primary_violations
                .is_empty()
            );
        }
        assert!(!check_thresholds(&[], 0.98).primary_violations.is_empty());
    }

    #[test]
    fn scale_fit_cannot_turn_missing_zero_or_constant_measurements_into_success() {
        for (xs, ys) in [
            (vec![], vec![]),
            (vec![1.0, 2.0], vec![0.0, 0.0]),
            (vec![1.0, 2.0], vec![5.0, 5.0]),
            (vec![1.0, 1.0], vec![5.0, 6.0]),
            (vec![1.0, 2.0], vec![5.0, f64::NAN]),
            (vec![1.0, 2.0], vec![5.0, f64::INFINITY]),
        ] {
            assert!(fit(&xs, &ys).2.is_nan());
        }
        assert!((fit(&[1.0, 2.0, 3.0], &[5.0, 10.0, 15.0]).2 - 1.0).abs() < FLOAT_EPS);
    }

    #[test]
    fn w1_and_w3_hundredfold_slowdown_is_outside_primary_speed_guarantee() {
        let results = [
            mk_result("w1", 100_000, 1000, 1.05, BUCKET_SATURATED),
            mk_result("w2", 500, 1000, 1.05, BUCKET_SHORT_ONLY),
            mk_result("w3", 100_000, 1000, 1.05, BUCKET_MIXED),
        ];
        let report = check_thresholds(&results, 0.98);
        assert!(report.primary_violations.is_empty());
        assert!(matches!(
            report.saturated_informational.as_slice(),
            [Violation::Sla {
                workload: "w1",
                actual: 100.0,
                ..
            }]
        ));
        assert!(matches!(
            report.aspirational_diagnostics.as_slice(),
            [Violation::Sla {
                workload: "w3",
                actual: 100.0,
                ..
            }]
        ));
    }
}
