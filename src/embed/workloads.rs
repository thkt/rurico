//! Deterministic text generators for embed-pipeline benchmarking.
//!
//! These workloads pin the shapes used by `docs/benchmarks/*.md` and
//! `tests/fixtures/phase2_baseline/*.bin`, so the fixture and baseline numbers
//! remain reproducible across runs. Any edit to these functions invalidates the
//! captured fixtures and must be accompanied by `mlx_smoke capture-fixture` and
//! `mlx_smoke measure-baseline`.

/// 2 long texts (~48K + ~22K chars). Multi-chunk after splitting.
pub fn workload_w1() -> Vec<String> {
    vec![
        "apple pie is a traditional dessert enjoyed around the world. ".repeat(800),
        "the rain in Spain falls mainly on the plain. ".repeat(500),
    ]
}

/// 100 short texts, each ~55 characters (single chunk, very small `max_seq_len`).
pub fn workload_w2() -> Vec<String> {
    (0..100)
        .map(|i| format!("short text number {i} for benchmarking W2 workload"))
        .collect()
}

/// 10 texts alternating long (~4K-5.6K tokens) and short (~100 bytes).
/// Stresses the long×short padding-waste case that bucket batching targets.
pub fn workload_w3() -> Vec<String> {
    (0..5)
        .flat_map(|i| {
            vec![
                "benchmarking long text for W3 workload. ".repeat(100 + i * 10),
                format!("short text {i}"),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests;
