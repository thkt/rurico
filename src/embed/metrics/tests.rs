use tracing_test::traced_test;

use super::*;

// padding_ratio: safe on zero real, ratio = padded / real otherwise
#[test]
fn padding_ratio_zero_real_returns_zero() {
    assert_eq!(padding_ratio(0, 0), 0.0);
    assert_eq!(padding_ratio(0, 100), 0.0);
}

#[test]
fn padding_ratio_no_padding_returns_one() {
    assert!((padding_ratio(100, 100) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn padding_ratio_half_real_returns_two() {
    assert!((padding_ratio(50, 100) - 2.0).abs() < f32::EPSILON);
}

// PhaseMetrics::padding_ratio delegates to the free function
#[test]
fn phase_metrics_padding_ratio_matches_fields() {
    let m = PhaseMetrics {
        real_tokens: 80,
        padded_tokens: 160,
        ..Default::default()
    };
    assert!((m.padding_ratio() - 2.0).abs() < f32::EPSILON);
}

#[test]
fn phase_metrics_default_padding_ratio_is_zero() {
    let m = PhaseMetrics::default();
    assert_eq!(m.padding_ratio(), 0.0);
}

// T-MET-001: log() emits named fields for the bucket histogram so
// subscribers can read per-bucket counts without parsing a format string.
#[traced_test]
#[test]
fn t_met_001_log_emits_named_bucket_hist_fields() {
    let mut m = PhaseMetrics::new(EmbedKind::Batch);
    m.bucket_hist = [3, 5, 2, 1];
    m.num_chunks = 11;
    m.log();
    assert!(
        logs_contain("bucket_hist_0=3"),
        "expected named field bucket_hist_0=3 in subscriber output",
    );
    assert!(
        logs_contain("bucket_hist_1=5"),
        "expected named field bucket_hist_1=5 in subscriber output",
    );
    assert!(
        logs_contain("bucket_hist_2=2"),
        "expected named field bucket_hist_2=2 in subscriber output",
    );
    assert!(
        logs_contain("bucket_hist_3=1"),
        "expected named field bucket_hist_3=1 in subscriber output",
    );
    assert!(
        logs_contain("num_chunks=11"),
        "expected named field num_chunks=11 in subscriber output",
    );
    assert!(
        logs_contain("embed phase metrics"),
        "expected event message in subscriber output",
    );
}
