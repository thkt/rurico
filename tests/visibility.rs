//! UI tests for visibility invariants enforced by the compiler.
//!
//! Each fixture under `tests/ui/` is a standalone integration-test source file
//! that intentionally violates a `pub(crate)` boundary. trybuild compiles each
//! fixture as an external crate against the published API surface and matches
//! the captured rustc diagnostic against the adjacent `.stderr` snapshot.
//!
//! Replaces the source-string match formerly in
//! `src/model_probe/tests.rs::T-016` (Issue #118 / TC-008).

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
