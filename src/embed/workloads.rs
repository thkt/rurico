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
mod tests {
    use super::*;

    #[test]
    fn w1_has_two_long_texts() {
        let w = workload_w1();
        assert_eq!(w.len(), 2);
        assert!(w[0].contains("apple pie"));
        assert!(w[1].contains("Spain"));
        assert!(
            w[0].len() > 40000,
            "w1[0] should be long, got {}",
            w[0].len()
        );
        assert!(
            w[1].len() > 20000,
            "w1[1] should be long, got {}",
            w[1].len()
        );
    }

    #[test]
    fn w2_has_hundred_short_texts() {
        let w = workload_w2();
        assert_eq!(w.len(), 100);
        assert!(
            w.iter().all(|t| t.len() < 80),
            "all W2 texts should be short"
        );
        assert!(w[0].contains("number 0"));
        assert!(w[99].contains("number 99"));
    }

    #[test]
    fn w3_alternates_long_and_short() {
        let w = workload_w3();
        assert_eq!(w.len(), 10);
        for (i, text) in w.iter().enumerate() {
            if i.is_multiple_of(2) {
                assert!(
                    text.len() > 3000,
                    "w3[{i}] should be long, got {}",
                    text.len()
                );
            } else {
                assert!(
                    text.len() < 50,
                    "w3[{i}] should be short, got {}",
                    text.len()
                );
            }
        }
    }

    #[test]
    fn workloads_are_deterministic() {
        assert_eq!(workload_w1(), workload_w1());
        assert_eq!(workload_w2(), workload_w2());
        assert_eq!(workload_w3(), workload_w3());
    }
}
