//! Codex seatbelt sandbox detection for MLX runtime gating.
//!
//! MLX / Metal initialization aborts in Codex Desktop's seatbelt sandbox.
//! Smoke binaries self-skip with [`SEATBELT_SKIP_EXIT`] via [`exit_if_seatbelt`],
//! and `test-mlx` runtime tests bail via [`require_unsandboxed_mlx_runtime`].

use std::env;
use std::process;

/// Exit code used by smoke binaries when skipping in a Codex seatbelt sandbox.
///
/// Matches BSD `sysexits.h` `EX_CONFIG` (configuration error) so the reason is
/// distinguishable from a real failure.
pub const SEATBELT_SKIP_EXIT: i32 = 78;

const ENV_VAR: &str = "CODEX_SANDBOX";
const SEATBELT_VALUE: &str = "seatbelt";

/// Returns `true` when running under Codex Desktop's seatbelt sandbox.
pub fn codex_seatbelt_sandbox_active() -> bool {
    env::var(ENV_VAR).is_ok_and(|v| v == SEATBELT_VALUE)
}

/// Exit the current smoke binary with [`SEATBELT_SKIP_EXIT`] when the Codex
/// seatbelt sandbox is active. Otherwise return without side effects.
///
/// Pass `env!("CARGO_BIN_NAME")` as `binary_name` so the log line identifies
/// the caller.
pub fn exit_if_seatbelt(binary_name: &str) {
    if codex_seatbelt_sandbox_active() {
        eprintln!(
            "{binary_name}: skipped in Codex seatbelt sandbox; \
             run outside sandbox for MLX verification"
        );
        process::exit(SEATBELT_SKIP_EXIT);
    }
}

/// Panics if running under Codex seatbelt — MLX runtime tests must opt out.
///
/// # Panics
///
/// Panics when `CODEX_SANDBOX=seatbelt`.
pub fn require_unsandboxed_mlx_runtime() {
    assert!(
        !codex_seatbelt_sandbox_active(),
        "MLX runtime tests must run outside Codex seatbelt sandbox"
    );
}
