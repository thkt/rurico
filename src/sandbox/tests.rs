use super::*;

// T-106-001: is_seatbelt_env
#[test]
fn is_seatbelt_env_returns_true_when_value_matches() {
    assert!(is_seatbelt_env(Some(SEATBELT_VALUE)));
}

// T-106-002: is_seatbelt_env
#[test]
fn is_seatbelt_env_returns_false_when_absent() {
    assert!(!is_seatbelt_env(None));
}

// T-106-003: is_seatbelt_env
#[test]
fn is_seatbelt_env_returns_false_for_other_value() {
    assert!(!is_seatbelt_env(Some("1")));
}
