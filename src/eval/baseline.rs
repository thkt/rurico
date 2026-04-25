//! Baseline snapshot serialisation (FR-015..FR-016).
//!
//! Persists per-baseline metric results, CI bounds, latency, and provenance
//! (model id / revision / mlx-rs version / fixture hash) so a future run can
//! verify against the committed JSON via `verify-baseline` mode.
//!
//! Phase 1c RED — JSON / markdown writers are stubs. Tests pin the
//! per-category `uninformative` flag logic (FR-016 / BR-002).
//!
//! [`MetricResult`] reuses the [`serde::Serialize`] / [`serde::Deserialize`]
//! derive added on the same type in `metrics.rs`; this module's
//! [`BaselineSnapshot`] composes it directly so its own derive is sufficient.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::eval::metrics::MetricResult;

/// Half-width threshold above which a per-category metric is flagged
/// `uninformative` (FR-016 / BR-002).
pub const UNINFORMATIVE_HALF_WIDTH: f64 = 0.10;

/// Frozen baseline produced by `eval_harness capture-baseline` and verified
/// later by `eval_harness verify-baseline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    /// ISO-8601 capture timestamp.
    pub timestamp: String,
    /// Hugging Face repo id of the embed model used.
    pub model_id: String,
    /// Pinned revision (commit hash) of the embed model.
    pub model_revision: String,
    /// `mlx-rs` semver string at capture time.
    pub mlx_rs_version: String,
    /// Content hash over `documents.jsonl + queries.jsonl + known_answers.jsonl`.
    pub fixture_hash: String,
    /// Global metric results (regression gate per BR-001).
    pub global: Vec<MetricResult>,
    /// Per-category metric breakdown for exploratory inspection.
    pub per_category: BTreeMap<String, Vec<MetricResult>>,
    /// Median per-query latency across the fixture in milliseconds.
    pub latency_p50_ms: f64,
    /// 95th percentile per-query latency in milliseconds.
    pub latency_p95_ms: f64,
}

/// Errors surfaced when writing a [`BaselineSnapshot`].
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum BaselineError {
    /// Filesystem failure while creating the output file.
    #[error("baseline io error: {0}")]
    Io(#[from] io::Error),
    /// JSON serialisation failure.
    #[error("baseline serialise error: {0}")]
    Serialise(#[from] serde_json::Error),
}

/// Build a [`MetricResult`] and set its `uninformative` flag from the CI
/// half-width (FR-016 / BR-002).
///
/// `uninformative` is `true` when `(ci_upper - ci_lower) / 2 > 0.10`, i.e.
/// strictly above [`UNINFORMATIVE_HALF_WIDTH`]. Equality with the threshold
/// stays informative.
#[must_use]
pub fn build_metric_result(
    name: String,
    k: usize,
    point: f64,
    ci_lower: f64,
    ci_upper: f64,
) -> MetricResult {
    let half_width = (ci_upper - ci_lower) / 2.0;
    let uninformative = half_width > UNINFORMATIVE_HALF_WIDTH;
    MetricResult {
        name,
        k,
        point_estimate: point,
        ci_lower,
        ci_upper,
        uninformative,
    }
}

/// Serialise `snapshot` as pretty JSON to `path`.
///
/// # Errors
///
/// Returns [`BaselineError::Io`] when the destination cannot be created and
/// [`BaselineError::Serialise`] when JSON encoding fails.
pub fn write_json(snapshot: &BaselineSnapshot, path: &Path) -> Result<(), BaselineError> {
    let json = serde_json::to_string_pretty(snapshot)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

/// Render `snapshot` as a human-readable markdown report at `path`.
///
/// # Errors
///
/// Returns [`BaselineError::Io`] when the destination cannot be created.
pub fn write_markdown(snapshot: &BaselineSnapshot, path: &Path) -> Result<(), BaselineError> {
    let mut output = String::new();
    output.push_str("# Baseline Snapshot\n\n");
    output.push_str(&format!("- Captured: {}\n", snapshot.timestamp));
    output.push_str(&format!(
        "- Model: {} @ {}\n",
        snapshot.model_id, snapshot.model_revision
    ));
    output.push_str(&format!("- mlx-rs: {}\n", snapshot.mlx_rs_version));
    output.push_str(&format!("- Fixture hash: {}\n", snapshot.fixture_hash));
    output.push_str(&format!(
        "- Latency p50/p95: {:.2}/{:.2} ms\n",
        snapshot.latency_p50_ms, snapshot.latency_p95_ms
    ));
    output.push_str("\n## Global metrics\n\n");
    write_metric_lines(&mut output, &snapshot.global);
    output.push_str("\n## Per-category metrics\n");
    for (category, metrics) in &snapshot.per_category {
        output.push_str(&format!("\n### {category}\n\n"));
        write_metric_lines(&mut output, metrics);
    }
    let mut file = File::create(path)?;
    file.write_all(output.as_bytes())?;
    Ok(())
}

fn write_metric_lines(buf: &mut String, metrics: &[MetricResult]) {
    for metric in metrics {
        let flag = if metric.uninformative {
            ", uninformative"
        } else {
            ""
        };
        buf.push_str(&format!(
            "- {} @{}: {:.4} (CI: {:.4}..{:.4}{flag})\n",
            metric.name, metric.k, metric.point_estimate, metric.ci_lower, metric.ci_upper
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::build_metric_result;

    // T-018: build_metric_result_wide_ci_flags_uninformative
    // FR-016 / BR-002: half-width = (0.65 - 0.35) / 2 = 0.15 > 0.10 →
    //                  MetricResult.uninformative == true.
    #[test]
    fn build_metric_result_wide_ci_flags_uninformative() {
        let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.35, 0.65);

        assert!(
            result.uninformative,
            "FR-016: half-width 0.15 > threshold 0.10 → uninformative must be true, \
             got result = {result:?}"
        );
    }

    // T-018b: build_metric_result_narrow_ci_stays_informative
    // FR-016 / BR-002: half-width = (0.55 - 0.45) / 2 = 0.05 < 0.10 →
    //                  MetricResult.uninformative == false (negative case).
    #[test]
    fn build_metric_result_narrow_ci_stays_informative() {
        let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.45, 0.55);

        assert!(
            !result.uninformative,
            "FR-016: half-width 0.05 < threshold 0.10 → uninformative must be false, \
             got result = {result:?}"
        );
    }
}
