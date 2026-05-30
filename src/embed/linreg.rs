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
mod tests;
