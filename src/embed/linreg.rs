//! Ordinary least squares linear regression and coefficient of determination.
//!
//! Kept self-contained so the smoke binary can assert NFR-005
//! (R² ≥ 0.95 of `(real_tokens, forward_eval_ms)`) without a scipy
//! dependency. Exposed as `#[doc(hidden)] pub mod linreg` from the crate
//! root so the `mlx_smoke` bin target can reach the helpers while they
//! stay out of rustdoc and the advertised public surface.

/// Ordinary least squares fit over f64 paired samples.
///
/// Returns `(slope, intercept)` for the best-fit line `y = slope * x + intercept`.
///
/// # Panics
///
/// Panics if `xs` and `ys` have mismatched lengths, if either is empty, or if
/// `xs` has zero variance (all values identical — slope is undefined).
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len(), "linear_regression: length mismatch");
    assert!(!xs.is_empty(), "linear_regression: empty input");

    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    assert!(
        den > 0.0,
        "linear_regression: xs has zero variance; slope undefined"
    );

    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

/// Coefficient of determination R² against the line `y = slope * x + intercept`.
///
/// Computed as `1 - ss_res/ss_tot`. The value is at most `1.0`; it approaches
/// `1.0` on tight fits and falls to `0.0` when the line is no better than
/// the mean of `ys`. It can be negative when the supplied coefficients fit
/// worse than the mean (`ss_res > ss_tot`) — callers that need a non-negative
/// score should clamp. When the total variance of `ys` is zero (all `ys`
/// identical), returns `1.0` — a constant series is treated as trivially fit
/// since any horizontal line through the mean has zero residual error.
pub fn r_squared(xs: &[f64], ys: &[f64], slope: f64, intercept: f64) -> f64 {
    assert_eq!(xs.len(), ys.len(), "r_squared: length mismatch");
    assert!(!xs.is_empty(), "r_squared: empty input");

    let mean_y = ys.iter().sum::<f64>() / (ys.len() as f64);
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let predicted = slope * x + intercept;
        ss_res += (y - predicted).powi(2);
        ss_tot += (y - mean_y).powi(2);
    }

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // T-MET-003: linear_regression returns the exact slope/intercept of a
    // noise-free line, establishing the self-implemented OLS baseline used
    // to assert forward_eval linearity per NFR-005.
    #[test]
    fn linreg_fits_perfect_line_returns_slope_and_intercept() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 5.0, 7.0];
        let (slope, intercept) = linear_regression(&xs, &ys);
        assert!(
            (slope - 2.0).abs() < EPS,
            "slope expected ≈ 2.0, got {slope:?}"
        );
        assert!(
            (intercept - 1.0).abs() < EPS,
            "intercept expected ≈ 1.0, got {intercept:?}"
        );
    }

    // T-MET-003: R² of an exact line through the sample points must be 1.0;
    // this anchors the "perfect fit" end of the threshold check.
    #[test]
    fn r_squared_on_perfect_line_is_approximately_one() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 5.0, 7.0];
        let r2 = r_squared(&xs, &ys, 2.0, 1.0);
        assert!((r2 - 1.0).abs() < EPS, "R² expected ≈ 1.0, got {r2:?}");
    }

    // T-MET-003: three collinear points yield R² = 1.0 — validates that the
    // n=3 regime used by W1/W2/W3 still scores cleanly when linearity holds.
    #[test]
    fn r_squared_on_three_collinear_points_approaches_one() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [10.0, 20.0, 30.0];
        let r2 = r_squared(&xs, &ys, 10.0, 0.0);
        assert!((r2 - 1.0).abs() < EPS, "R² expected ≈ 1.0, got {r2:?}");
    }

    // T-MET-003: a step pattern is not explainable by a line — R² must fall
    // below the 0.5 sanity bound so the threshold check can actually fail
    // when padding is not eliminated.
    #[test]
    fn r_squared_on_noisy_data_falls_below_threshold() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 10.0, 1.0, 10.0];
        let (slope, intercept) = linear_regression(&xs, &ys);
        let r2 = r_squared(&xs, &ys, slope, intercept);
        assert!(
            (0.0..0.5).contains(&r2),
            "R² expected in [0, 0.5) for step data, got {r2:?}"
        );
    }

    #[test]
    fn r_squared_on_constant_ys_is_one() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [5.0, 5.0, 5.0];
        let r2 = r_squared(&xs, &ys, 0.0, 5.0);
        assert_eq!(r2, 1.0, "constant ys must be treated as trivially fit");
    }

    #[test]
    fn linreg_two_points_yields_perfect_fit() {
        let xs = [0.0, 1.0];
        let ys = [0.0, 5.0];
        let (slope, intercept) = linear_regression(&xs, &ys);
        assert!(
            (slope - 5.0).abs() < EPS,
            "slope expected 5.0, got {slope:?}"
        );
        assert!(
            (intercept - 0.0).abs() < EPS,
            "intercept expected 0.0, got {intercept:?}"
        );
        let r2 = r_squared(&xs, &ys, slope, intercept);
        assert!((r2 - 1.0).abs() < EPS, "R² expected 1.0, got {r2:?}");
    }

    #[test]
    #[should_panic]
    fn linreg_empty_inputs_panic() {
        let xs: [f64; 0] = [];
        let ys: [f64; 0] = [];
        let _ = linear_regression(&xs, &ys);
    }

    #[test]
    #[should_panic]
    fn linreg_mismatched_lengths_panics() {
        let xs = [1.0, 2.0];
        let ys = [1.0];
        let _ = linear_regression(&xs, &ys);
    }
}
