//! Tests for `crate::model_probe`.
//!
//! Existing tests cover `resolve_probe_env`, `interpret_probe_output`,
//! `compute_probe_exit`, and the wait/spawn machinery. New tests added for
//! Issue #107 Phase 1 (AC-1, AC-4) cover `validate_probe_paths_with_root`,
//! the production wrapper `validate_probe_paths`, and the visibility
//! downgrade of `embed::CandidateArtifacts::from_paths` /
//! `reranker::CandidateArtifacts::from_paths`.

use super::*;
use std::io;
use std::process::ExitStatus;

// ── Existing tests preserved verbatim ───────────────────────────────────────

#[test]
fn resolve_probe_env_returns_none_when_model_absent() {
    assert!(resolve_probe_env(None, None, None).is_none());
    assert!(resolve_probe_env(None, Some("c".into()), Some("t".into())).is_none());
}

#[test]
fn resolve_probe_env_returns_err_when_incomplete() {
    assert_eq!(
        resolve_probe_env(Some("m".into()), None, Some("t".into())),
        Some(Err(PROBE_EXIT_ENV_INCOMPLETE))
    );
    assert_eq!(
        resolve_probe_env(Some("m".into()), Some("c".into()), None),
        Some(Err(PROBE_EXIT_ENV_INCOMPLETE))
    );
}

#[test]
fn resolve_probe_env_returns_paths_when_all_present() {
    let (model, config, tokenizer) =
        resolve_probe_env(Some("/m".into()), Some("/c".into()), Some("/t".into()))
            .unwrap()
            .unwrap();
    assert_eq!(model, PathBuf::from("/m"));
    assert_eq!(config, PathBuf::from("/c"));
    assert_eq!(tokenizer, PathBuf::from("/t"));
}

fn exit_status(code: i32) -> ExitStatus {
    Command::new("sh")
        .args(["-c", &format!("exit {code}")])
        .status()
        .unwrap()
}

#[test]
fn interpret_available_on_exit_0() {
    let output = Output {
        status: exit_status(0),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    assert_eq!(
        interpret_probe_output(&output).unwrap(),
        ProbeStatus::Available
    );
}

#[test]
fn interpret_model_load_failed_on_nonzero_exit() {
    let output = Output {
        status: exit_status(1),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"inference error: bad model".to_vec(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::ModelLoadFailed { ref reason } if reason.contains("bad model")),
        "{err}"
    );
}

#[test]
fn interpret_model_load_failed_empty_stderr() {
    let output = Output {
        status: exit_status(1),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::ModelLoadFailed { ref reason } if reason == "model load failed"),
        "{err}"
    );
}

#[test]
fn interpret_handler_not_installed_on_missing_ack() {
    let output = Output {
        status: exit_status(0),
        stdout: b"unexpected output".to_vec(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(matches!(err, ProbeError::HandlerNotInstalled), "{err}");
}

#[cfg(unix)]
#[test]
fn interpret_timeout_output_returns_backend_unavailable() {
    let output = super::build_timeout_output();
    assert_eq!(
        interpret_probe_output(&output).unwrap(),
        ProbeStatus::BackendUnavailable,
    );
}

#[test]
fn probe_exit_env_incomplete() {
    let action = compute_probe_exit(Err(PROBE_EXIT_ENV_INCOMPLETE));
    assert_eq!(action.code, PROBE_EXIT_ENV_INCOMPLETE);
    assert!(action.message.is_some());
}

#[test]
fn probe_exit_load_success() {
    let action = compute_probe_exit(Ok(Ok(())));
    assert_eq!(action.code, 0);
    assert!(action.message.is_none());
}

#[test]
fn probe_exit_load_failure() {
    let action = compute_probe_exit(Ok(Err("bad model".into())));
    assert_eq!(action.code, 1);
    assert_eq!(action.message.as_deref(), Some("bad model"));
}

#[test]
fn interpret_backend_unavailable_on_signal() {
    let status = Command::new("sh")
        .args(["-c", "kill -ABRT $$"])
        .status()
        .unwrap();
    let output = Output {
        status,
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    assert_eq!(
        interpret_probe_output(&output).unwrap(),
        ProbeStatus::BackendUnavailable
    );
}

#[test]
fn wait_with_timeout_drains_verbose_failure_before_timeout() {
    let mut child = Command::new("sh")
        .args([
            "-c",
            "printf '%s\\n' \"$0\"; \
                 i=0; \
                 while [ \"$i\" -lt 5000 ]; do \
                   printf 'verbose probe failure line %04d\\n' \"$i\" 1>&2; \
                   i=$((i + 1)); \
                 done; \
                 exit 1",
            PROBE_ACK,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let output = super::wait_with_timeout(&mut child, Duration::from_secs(2)).unwrap();
    assert_eq!(output.status.code(), Some(1), "child should exit normally");
    assert!(
        output.stderr.len() > 100_000,
        "stderr should be fully drained, got {} bytes",
        output.stderr.len()
    );

    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::ModelLoadFailed { .. }),
        "expected verbose failure to remain ModelLoadFailed, got {err}"
    );
}

#[cfg(unix)]
#[test]
fn wait_with_timeout_returns_when_grandchild_inherits_pipes() {
    let mut child = Command::new("sh")
        .args(["-c", "sleep 10 & printf '%s\\n' \"$0\"; exit 1", PROBE_ACK])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let start = Instant::now();
    let output = super::wait_with_timeout(&mut child, Duration::from_secs(30)).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(
        output.status.code(),
        Some(1),
        "direct child should exit with 1 before the grandchild finishes"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "collect_pipe must not wait for the grandchild's inherited FDs; elapsed {elapsed:?}"
    );
}

#[test]
fn probe_via_subprocess_with_reports_spawn_failure_for_missing_exe() {
    let exe = PathBuf::from("/nonexistent/rurico-probe-test-binary");
    let err = probe_via_subprocess_with(exe, &[]).unwrap_err();
    assert!(
        matches!(err, ProbeError::SubprocessFailed(_)),
        "expected SubprocessFailed for missing executable, got {err}"
    );
}

#[test]
fn wait_with_timeout_with_kills_long_running_child_on_short_timeout() {
    let mut child = Command::new("sh")
        .args(["-c", "sleep 30"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let start = Instant::now();
    let output = super::wait_with_timeout_with(
        &mut child,
        Duration::from_millis(50),
        Duration::from_millis(1),
    )
    .unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "1ms poll interval should detect the 50ms deadline promptly; elapsed {elapsed:?}"
    );
    assert!(
        output.status.code().is_none() || output.status.code() == Some(0),
        "child should be killed (no exit code) or have exited; got {:?}",
        output.status.code()
    );
}

// ── Issue #107 Phase 1 tests (AC-1, AC-4) ────────────────────────────────────
//
// Red-phase tests against `validate_probe_paths_with_root`,
// `validate_probe_paths`, and the constants `PROBE_EXIT_CANONICALIZE_FAILED`
// (4) / `PROBE_EXIT_PATH_OUTSIDE_CACHE` (5) / `PROBE_EXIT_CACHE_ROOT_INVALID`
// (6). None of these symbols exist yet, so the Red signal is a compile error.
// The Green implementation must declare them with `pub(crate)` visibility.

use std::fs;

/// Helper: write the three artifact files into `dir` and return their paths.
///
/// Used by T-001/T-002/T-003/T-006 (and the symlink variants in T-005/T-021)
/// to keep arrangement uniform.
fn write_three_artifacts(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let model = dir.join("model.safetensors");
    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    fs::write(&model, b"weights").unwrap();
    fs::write(&config, b"{}").unwrap();
    fs::write(&tokenizer, b"{}").unwrap();
    (model, config, tokenizer)
}

// T-001: validate_probe_paths_with_root happy path -- 3 cached files inside cache root return Ok(())
#[test]
fn t_001_validate_probe_paths_with_root_returns_ok_when_all_paths_under_cache_root() {
    // Arrange
    let cache_dir = tempfile::tempdir().unwrap();
    let (model, config, tokenizer) = write_three_artifacts(cache_dir.path());

    // Act
    let cache = hf_hub::Cache::new(cache_dir.path().to_path_buf());
    let result = super::validate_probe_paths_with_cache(&cache, &model, &config, &tokenizer);

    // Assert
    assert_eq!(result, Ok(()), "expected Ok(()) for paths under cache root");
}

// T-002: validate_probe_paths_with_root rejects path outside cache_root with Err(5)
#[test]
fn t_002_validate_probe_paths_with_root_rejects_path_outside_cache_root() {
    // Arrange
    let cache_dir = tempfile::tempdir().unwrap();
    let outside_dir = tempfile::tempdir().unwrap();
    let (_, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let outside_model = outside_dir.path().join("outside.bin");
    fs::write(&outside_model, b"evil").unwrap();

    // Act
    let cache = hf_hub::Cache::new(cache_dir.path().to_path_buf());
    let result =
        super::validate_probe_paths_with_cache(&cache, &outside_model, &config, &tokenizer);

    // Assert
    assert_eq!(
        result,
        Err(super::PROBE_EXIT_PATH_OUTSIDE_CACHE),
        "expected Err(5) for path outside cache root"
    );
    assert_eq!(super::PROBE_EXIT_PATH_OUTSIDE_CACHE, 5);
}

// T-003: validate_probe_paths_with_root reports canonicalize failure with Err(4)
#[test]
fn t_003_validate_probe_paths_with_root_returns_err_4_for_nonexistent_candidate_path() {
    // Arrange
    let cache_dir = tempfile::tempdir().unwrap();
    let (_, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let missing_model = cache_dir.path().join("nonexistent.safetensors");
    // Intentionally do not create the file.

    // Act
    let cache = hf_hub::Cache::new(cache_dir.path().to_path_buf());
    let result =
        super::validate_probe_paths_with_cache(&cache, &missing_model, &config, &tokenizer);

    // Assert
    assert_eq!(
        result,
        Err(super::PROBE_EXIT_CANONICALIZE_FAILED),
        "expected Err(4) for nonexistent candidate path"
    );
    assert_eq!(super::PROBE_EXIT_CANONICALIZE_FAILED, 4);
}

// T-004: validate_probe_paths_with_root reports invalid cache root with Err(6)
#[test]
fn t_004_validate_probe_paths_with_root_returns_err_6_for_nonexistent_cache_root() {
    // Arrange
    let cache_dir = tempfile::tempdir().unwrap();
    let (model, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let bogus_root = PathBuf::from("/nonexistent/rurico-test-cache-root");

    // Act
    let cache = hf_hub::Cache::new(bogus_root.clone());
    let result = super::validate_probe_paths_with_cache(&cache, &model, &config, &tokenizer);

    // Assert
    assert_eq!(
        result,
        Err(super::PROBE_EXIT_CACHE_ROOT_INVALID),
        "expected Err(6) when cache_root cannot be canonicalized"
    );
    assert_eq!(super::PROBE_EXIT_CACHE_ROOT_INVALID, 6);
}

// T-005: validate_probe_paths_with_root accepts symlink path inside cache_root
//
// The function MUST return Ok(()) and MUST NOT modify the caller's PathBuf
// (signature is `&Path` in, `Result<(), i32>` out — symlink invariant guarded
// by FR-005's type-level constraint, exercised here through behavior).
#[cfg(unix)]
#[test]
fn t_005_validate_probe_paths_with_root_accepts_symlink_under_cache_root() {
    // Arrange: HF cache layout — blobs/<etag> is the real file, snapshots/<commit>/<name>
    // is a symlink pointing at the blob.
    let cache_dir = tempfile::tempdir().unwrap();
    let blobs_dir = cache_dir.path().join("blobs");
    let snapshot_dir = cache_dir.path().join("snapshots").join("commit-abc");
    fs::create_dir_all(&blobs_dir).unwrap();
    fs::create_dir_all(&snapshot_dir).unwrap();

    let blob_model = blobs_dir.join("etag-model");
    let blob_config = blobs_dir.join("etag-config");
    let blob_tokenizer = blobs_dir.join("etag-tokenizer");
    fs::write(&blob_model, b"weights").unwrap();
    fs::write(&blob_config, b"{}").unwrap();
    fs::write(&blob_tokenizer, b"{}").unwrap();

    use std::os::unix::fs::symlink;
    let model = snapshot_dir.join("model.safetensors");
    let config = snapshot_dir.join("config.json");
    let tokenizer = snapshot_dir.join("tokenizer.json");
    symlink(&blob_model, &model).unwrap();
    symlink(&blob_config, &config).unwrap();
    symlink(&blob_tokenizer, &tokenizer).unwrap();

    // Act
    let cache = hf_hub::Cache::new(cache_dir.path().to_path_buf());
    let result = super::validate_probe_paths_with_cache(&cache, &model, &config, &tokenizer);

    // Assert: function accepts the symlink as long as the canonicalized form
    // resolves under cache_root (which it does — blobs/ is under cache_dir).
    assert_eq!(result, Ok(()));
}

// T-006: validate_probe_paths_with_root rejects component-wise prefix collision
//
// `cache_x_evil/...` shares a string prefix with `cache_x` but is NOT a
// component-wise descendant. PathBuf::starts_with handles this correctly via
// component comparison; the test guards against a regression to byte-prefix
// comparison.
#[test]
fn t_006_validate_probe_paths_with_root_rejects_string_prefix_sibling() {
    // Arrange: two sibling tempdirs, "cache_x" and "cache_x_evil".
    let workspace = tempfile::tempdir().unwrap();
    let cache_root = workspace.path().join("cache_x");
    let evil_root = workspace.path().join("cache_x_evil");
    fs::create_dir_all(&cache_root).unwrap();
    fs::create_dir_all(&evil_root).unwrap();

    let (_, config, tokenizer) = write_three_artifacts(&cache_root);
    let evil_model = evil_root.join("model.bin");
    fs::write(&evil_model, b"evil").unwrap();

    // Act
    let cache = hf_hub::Cache::new(cache_root.clone());
    let result = super::validate_probe_paths_with_cache(&cache, &evil_model, &config, &tokenizer);

    // Assert: even though `evil_root` shares a string prefix with `cache_root`,
    // component-wise starts_with rejects it.
    assert_eq!(
        result,
        Err(super::PROBE_EXIT_PATH_OUTSIDE_CACHE),
        "component-wise prefix check must reject `cache_x_evil` against `cache_x`"
    );
}

// T-016: CandidateArtifacts::from_paths visibility -- pub(crate) downgrade
//
// Build verification: source-level inspection at red phase shows `from_paths`
// is currently `pub`. The Phase 1 Green change downgrades to `pub(crate)`.
// This test does not discriminate visibility levels at runtime (a `pub`
// function and a `pub(crate)` function both compile from inside the crate);
// the discriminator lives in the source check below and in downstream
// `cargo build --workspace` after rev bump (covered by AC-4-3, out of scope
// for this in-crate test file).
//
// To make T-016 a meaningful Red signal in the source-level sense, this test
// reads the current declaration and asserts the expected `pub(crate) fn`
// prefix. After the Green change, this assertion will pass; before the Green
// change, the assertion fails on the `pub fn` line.
#[test]
fn t_016_from_paths_is_declared_pub_crate_in_embed_and_reranker_modules() {
    // Arrange
    let embed_src = fs::read_to_string("src/embed.rs").unwrap();
    let reranker_src = fs::read_to_string("src/reranker.rs").unwrap();

    // Assert
    assert!(
        embed_src.contains("pub(crate) fn from_paths("),
        "embed::CandidateArtifacts::from_paths must be declared `pub(crate) fn from_paths(`"
    );
    assert!(
        reranker_src.contains("pub(crate) fn from_paths("),
        "reranker::CandidateArtifacts::from_paths must be declared `pub(crate) fn from_paths(`"
    );
    // Negative assertion: bare `pub fn from_paths(` must not exist.
    assert!(
        !embed_src.contains("\n    pub fn from_paths("),
        "embed::CandidateArtifacts::from_paths must not be declared `pub fn`"
    );
    assert!(
        !reranker_src.contains("\n    pub fn from_paths("),
        "reranker::CandidateArtifacts::from_paths must not be declared `pub fn`"
    );
}

// T-021: validate_probe_paths (production wrapper) HF_HOME unset fallback (FR-101 + FR-006)
//
// The production wrapper resolves cache_root via `hf_hub::Cache::from_env()`.
// When `HF_HOME` is unset, hf_hub falls back to `dirs::home_dir() /
// .cache/huggingface/hub`. On Unix, `dirs::home_dir()` reads `$HOME` first
// (verified against `dirs-sys-0.5.0/src/lib.rs` line 33-71). Setting
// `HOME=tempdir` redirects the fallback into the test's tempdir.
//
// NOTE for Phase 1 Green implementer: `validate_probe_paths` MUST call
// `Cache::from_env()` at call time. A `LazyLock` / `OnceLock` cache root would
// poison parallel tests (the first caller fixes the value crate-wide).
#[cfg(unix)]
#[test]
fn t_021_validate_probe_paths_falls_back_to_home_when_hf_home_unset() {
    // Arrange: tempdir + HF cache layout under <tempdir>/.cache/huggingface/hub.
    let home_dir = tempfile::tempdir().unwrap();
    let hub_dir = home_dir.path().join(".cache/huggingface/hub");
    let snapshot_dir = hub_dir.join("models--cl-nagoya--ruri-v3-310m/snapshots/commit-xyz");
    fs::create_dir_all(&snapshot_dir).unwrap();
    let (model, config, tokenizer) = write_three_artifacts(&snapshot_dir);

    // Act + Assert: scope HOME / HF_HOME / HF_HUB_CACHE inside the closure.
    let home_path = home_dir.path().to_path_buf();
    temp_env::with_vars(
        [
            ("HF_HOME", None::<&str>),
            ("HF_HUB_CACHE", None::<&str>),
            ("HOME", Some(home_path.to_str().unwrap())),
        ],
        || {
            let result = super::validate_probe_paths(&model, &config, &tokenizer);
            assert_eq!(
                result,
                Ok(()),
                "fallback to $HOME/.cache/huggingface/hub must accept paths under it"
            );
        },
    );
}

// ── Issue #107 Phase 2 tests (AC-2) ──────────────────────────────────────────
//
// Tests for typed setup-failure error variant `ProbeError::SetupRejected { code }`
// and the `interpret_probe_output` dispatch from setup-phase exit codes 4 / 5 / 6.

// T-007: exit 4 -> ProbeError::SetupRejected { code: 4 }
#[test]
fn t_007_interpret_probe_output_maps_exit_4_to_setup_rejected() {
    let output = Output {
        status: exit_status(4),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"path canonicalize failed".to_vec(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::SetupRejected { code: 4 }),
        "expected SetupRejected {{ code: 4 }}, got {err}"
    );
}

// T-008: exit 5 -> ProbeError::SetupRejected { code: 5 }
#[test]
fn t_008_interpret_probe_output_maps_exit_5_to_setup_rejected() {
    let output = Output {
        status: exit_status(5),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::SetupRejected { code: 5 }),
        "expected SetupRejected {{ code: 5 }}, got {err}"
    );
}

// T-007a: exit 3 (env incomplete) -> ProbeError::SetupRejected { code: 3 }
#[test]
fn t_007a_interpret_probe_output_maps_exit_3_to_setup_rejected() {
    let output = Output {
        status: exit_status(3),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"probe setup rejected: env incomplete".to_vec(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::SetupRejected { code: 3 }),
        "expected SetupRejected {{ code: 3 }}, got {err}"
    );
}

// T-009: exit 6 -> ProbeError::SetupRejected { code: 6 }
#[test]
fn t_009_interpret_probe_output_maps_exit_6_to_setup_rejected() {
    let output = Output {
        status: exit_status(6),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::SetupRejected { code: 6 }),
        "expected SetupRejected {{ code: 6 }}, got {err}"
    );
}

// T-011a: SetupRejected { code: 4 } Display label
#[test]
fn t_011a_setup_rejected_display_for_code_4_includes_label() {
    let err = ProbeError::SetupRejected { code: 4 };
    let s = format!("{err}");
    assert!(s.contains('4'), "{s}");
    assert!(s.contains("path canonicalize failed"), "{s}");
}

// T-011b: SetupRejected { code: 5 } Display label
#[test]
fn t_011b_setup_rejected_display_for_code_5_includes_label() {
    let err = ProbeError::SetupRejected { code: 5 };
    let s = format!("{err}");
    assert!(s.contains('5'), "{s}");
    assert!(s.contains("path outside cache"), "{s}");
}

// T-011c: SetupRejected { code: 6 } Display label
#[test]
fn t_011c_setup_rejected_display_for_code_6_includes_label() {
    let err = ProbeError::SetupRejected { code: 6 };
    let s = format!("{err}");
    assert!(s.contains('6'), "{s}");
    assert!(s.contains("cache root invalid"), "{s}");
}

// T-011d: SetupRejected { code: 3 } Display label
#[test]
fn t_011d_setup_rejected_display_for_code_3_includes_label() {
    let err = ProbeError::SetupRejected { code: 3 };
    let s = format!("{err}");
    assert!(s.contains('3'), "{s}");
    assert!(s.contains("env incomplete"), "{s}");
}

// ── Issue #107 Phase 3 tests (AC-3) ──────────────────────────────────────────
//
// Tests for the FORWARD env allowlist constant, the `child_env_for_spawn` test
// seam (FR-016 — pure function returning the spawn env map), and the silent
// skip behavior for undefined FORWARD keys (FR-102).

// T-012: FORWARD list contains exactly the 15 expected keys
#[test]
fn t_012_forward_list_contains_expected_15_keys() {
    let expected: &[&str] = &[
        "PATH",
        "HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "RUST_LOG",
        "RUST_BACKTRACE",
    ];
    assert_eq!(super::FORWARD.len(), 15);
    for key in expected {
        assert!(
            super::FORWARD.contains(key),
            "FORWARD list missing key {key}"
        );
    }
}

// T-013: child_env_for_spawn forwards HF_HOME, drops attacker-injected env
#[test]
fn t_013_child_env_for_spawn_drops_attacker_env_keeps_forward() {
    temp_env::with_vars(
        [
            ("HF_HOME", Some("/tmp/test_hf")),
            ("RURICO_TEST_ATTACKER_ENV", Some("evil")),
        ],
        || {
            let env = super::child_env_for_spawn(&[]);
            assert_eq!(
                env.get("HF_HOME").map(String::as_str),
                Some("/tmp/test_hf"),
                "HF_HOME must be forwarded from parent"
            );
            assert!(
                !env.contains_key("RURICO_TEST_ATTACKER_ENV"),
                "RURICO_TEST_ATTACKER_ENV must NOT be forwarded (env_clear hardening)"
            );
        },
    );
}

// T-022: child_env_for_spawn skips undefined FORWARD keys silently (FR-102)
#[test]
fn t_022_child_env_for_spawn_skips_undefined_forward_keys() {
    temp_env::with_vars([("HF_HOME", None::<&str>)], || {
        let env = super::child_env_for_spawn(&[]);
        assert!(
            !env.contains_key("HF_HOME"),
            "HF_HOME must NOT be in map when undefined in parent"
        );
    });
}

// T-013b: e2e — `probe_via_subprocess_with` actually applies env_clear + FORWARD
// to a real spawned child (TC-001 from /audit). Spawns a sh script that dumps its
// env to a tempfile, then asserts attacker-injected env is absent and FORWARD
// keys propagate.
#[cfg(unix)]
#[test]
fn t_013b_probe_via_subprocess_with_env_clear_blocks_attacker_env_in_real_child() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let env_dump = dir.path().join("env_dump.txt");
    let script = dir.path().join("probe_env_dump.sh");
    fs::write(
        &script,
        format!(
            "#!/bin/sh\nenv > {}\nprintf '{PROBE_ACK}\\n'\nexit 0\n",
            env_dump.display()
        ),
    )
    .unwrap();
    let mut perms = fs::metadata(&script).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script, perms).unwrap();

    temp_env::with_vars(
        [
            ("HF_HOME", Some("/tmp/test_hf_clear")),
            ("RURICO_TEST_ATTACKER_ENV", Some("evil")),
        ],
        || {
            // exec succeeds; we ignore the ProbeStatus and inspect env_dump directly.
            let _ = super::probe_via_subprocess_with(script.clone(), &[]);
            let dumped = fs::read_to_string(&env_dump).unwrap();
            assert!(
                dumped.lines().any(|l| l == "HF_HOME=/tmp/test_hf_clear"),
                "child must inherit HF_HOME via FORWARD allowlist; dump:\n{dumped}"
            );
            assert!(
                !dumped.contains("RURICO_TEST_ATTACKER_ENV"),
                "child must NOT inherit non-FORWARD parent env; dump:\n{dumped}"
            );
        },
    );
}

// ── Issue #94 (probe IO error propagation) tests ────────────────────────────
//
// `emit_ack_to` is a testable seam exposed for these tests; production calls
// `emit_ack` which delegates to `emit_ack_to(&mut io::stdout())`. The new
// IO-infrastructure exit codes `PROBE_EXIT_ACK_FAILED` (7) and
// `PROBE_EXIT_STDERR_FAILED` (8) signal that the probe child could not
// emit its handshake (stdout) or its failure reason (stderr) — distinct
// from `HandlerNotInstalled` (caller forgot the probe handler) and
// `ModelLoadFailed` (load itself failed).

// T-023: emit_ack_to writes PROBE_ACK + newline on a valid writer
#[test]
fn t_023_emit_ack_to_writes_ack_token_with_newline() {
    let mut buf: Vec<u8> = Vec::new();
    let result = super::emit_ack_to(&mut buf);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    assert_eq!(buf, format!("{PROBE_ACK}\n").into_bytes());
}

// T-024: emit_ack_to propagates IO errors from a failing writer
//
// `FailingWriter::write` fails first and `?` short-circuits before `flush`
// runs, so this test only exercises the write-failure arm. The flush-only
// arm (write success + flush error) is not unit-tested; in production the
// pipe-broken case typically surfaces at write time on `io::stdout()`.
#[test]
fn t_024_emit_ack_to_propagates_writer_error() {
    struct FailingWriter;
    impl io::Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }
    }
    let err = super::emit_ack_to(&mut FailingWriter)
        .expect_err("expected emit_ack_to to propagate writer error");
    assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
}

// T-025: interpret_probe_output maps exit 7 to SubprocessFailed even without ACK
//
// IO infrastructure failures (ACK write itself failed) MUST be detected
// before the ACK presence check — otherwise an empty stdout caused by a
// failed ACK write would be misclassified as `HandlerNotInstalled`.
#[test]
fn t_025_interpret_probe_output_maps_exit_7_to_subprocess_failed_even_without_ack() {
    let output = Output {
        status: exit_status(super::PROBE_EXIT_ACK_FAILED),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    let ProbeError::SubprocessFailed(msg) = &err else {
        panic!("expected SubprocessFailed, got {err}");
    };
    assert!(
        msg.contains("ACK") || msg.contains("handshake"),
        "expected ACK/handshake mention, got: {msg}"
    );
    assert_eq!(super::PROBE_EXIT_ACK_FAILED, 7);
}

// T-026: interpret_probe_output maps exit 8 to SubprocessFailed when ACK is present
//
// `dispatch_probe` only emits exit 8 after `emit_ack` succeeds, so a real
// probe failure of this kind always carries the ACK in stdout.
#[test]
fn t_026_interpret_probe_output_maps_exit_8_to_subprocess_failed() {
    let output = Output {
        status: exit_status(super::PROBE_EXIT_STDERR_FAILED),
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    let ProbeError::SubprocessFailed(msg) = &err else {
        panic!("expected SubprocessFailed, got {err}");
    };
    assert!(
        msg.contains("stderr") || msg.contains("reason"),
        "expected stderr/reason mention, got: {msg}"
    );
    assert_eq!(super::PROBE_EXIT_STDERR_FAILED, 8);
}

// T-027: exit 8 without ACK preserves HandlerNotInstalled diagnostic
//
// Regression guard: if a host binary lacks the probe handler and happens to
// exit with code 8, the missing ACK must classify it as
// `HandlerNotInstalled`, not `SubprocessFailed`. Exit 8 is meaningful only
// when emitted by `dispatch_probe` after a successful ACK; an exit 8 from
// any other code path predates the probe contract.
#[test]
fn t_027_interpret_probe_output_exit_8_without_ack_is_handler_not_installed() {
    let output = Output {
        status: exit_status(super::PROBE_EXIT_STDERR_FAILED),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    let err = interpret_probe_output(&output).unwrap_err();
    assert!(
        matches!(err, ProbeError::HandlerNotInstalled),
        "expected HandlerNotInstalled when exit 8 is emitted without ACK (host \
         binary lacks probe handler), got {err}"
    );
}
