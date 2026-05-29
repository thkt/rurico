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
