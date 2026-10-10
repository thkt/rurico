//! Tests for `crate::model_probe`.

use super::*;
use std::fs;
use std::io;
use std::process::ExitStatus;

mod test_writers {
    use std::io;

    pub struct FailingWriter;
    impl io::Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
        }
    }

    #[derive(Default)]
    pub struct FlushTrackingWriter {
        pub buf: Vec<u8>,
        pub flush_count: usize,
    }
    impl io::Write for FlushTrackingWriter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.buf.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flush_count += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    pub struct FlushFailingWriter {
        pub buf: Vec<u8>,
    }
    impl io::Write for FlushFailingWriter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.buf.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "flush after write failed",
            ))
        }
    }
}

#[test]
fn resolve_probe_env_returns_none_when_model_absent() {
    assert!(resolve_probe_env(None, None, None).is_none());
    assert!(resolve_probe_env(None, Some("c".into()), Some("t".into())).is_none());
}

#[test]
fn resolve_probe_env_returns_err_when_incomplete() {
    assert_eq!(
        resolve_probe_env(Some("m".into()), None, Some("t".into())),
        Some(Err(SetupReason::EnvIncomplete))
    );
    assert_eq!(
        resolve_probe_env(Some("m".into()), Some("c".into()), None),
        Some(Err(SetupReason::EnvIncomplete))
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
    use std::os::unix::process::ExitStatusExt;
    ExitStatus::from_raw(code << 8)
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
    let action = compute_probe_exit(Err(SetupReason::EnvIncomplete));
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
    use std::os::unix::process::ExitStatusExt;
    // POSIX wait status: signal 6 (SIGABRT), with no normal exit code.
    let status = ExitStatus::from_raw(6);
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
    use std::os::unix::process::CommandExt;
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
        .process_group(0)
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
    use std::os::unix::process::CommandExt;
    let mut child = Command::new("sh")
        .args(["-c", "sleep 10 & printf '%s\\n' \"$0\"; exit 1", PROBE_ACK])
        .process_group(0)
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
        "collection must stop the grandchild holding inherited FDs; elapsed {elapsed:?}"
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
    use std::os::unix::process::CommandExt;
    let mut child = Command::new("sh")
        .args(["-c", "sleep 30"])
        .process_group(0)
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
        output.status.code().is_none(),
        "hung child should be killed (no exit code); got {:?}",
        output.status.code()
    );
}

/// Write the three artifact files into `dir` and return their paths.
fn write_three_artifacts(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let model = dir.join("model.safetensors");
    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    fs::write(&model, b"weights").unwrap();
    fs::write(&config, b"{}").unwrap();
    fs::write(&tokenizer, b"{}").unwrap();
    (model, config, tokenizer)
}

#[test]
fn validate_probe_paths_with_root_returns_ok_when_all_paths_under_cache_root() {
    let cache_dir = tempfile::tempdir().unwrap();
    let (model, config, tokenizer) = write_three_artifacts(cache_dir.path());

    let result =
        super::validate_probe_paths_with_cache(cache_dir.path(), &model, &config, &tokenizer);

    assert_eq!(result, Ok(()), "expected Ok(()) for paths under cache root");
}

#[test]
fn validate_probe_paths_with_root_rejects_path_outside_cache_root() {
    let cache_dir = tempfile::tempdir().unwrap();
    let outside_dir = tempfile::tempdir().unwrap();
    let (_, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let outside_model = outside_dir.path().join("outside.bin");
    fs::write(&outside_model, b"evil").unwrap();

    let result = super::validate_probe_paths_with_cache(
        cache_dir.path(),
        &outside_model,
        &config,
        &tokenizer,
    );

    assert_eq!(
        result,
        Err(SetupReason::PathOutsideCache),
        "expected Err(SetupReason::PathOutsideCache) for path outside cache root"
    );
}

#[test]
fn validate_probe_paths_with_root_returns_err_4_for_nonexistent_candidate_path() {
    let cache_dir = tempfile::tempdir().unwrap();
    let (_, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let missing_model = cache_dir.path().join("nonexistent.safetensors");

    let result = super::validate_probe_paths_with_cache(
        cache_dir.path(),
        &missing_model,
        &config,
        &tokenizer,
    );

    assert_eq!(
        result,
        Err(SetupReason::CanonicalizeFailed),
        "expected Err(SetupReason::CanonicalizeFailed) for nonexistent candidate path"
    );
}

#[test]
fn validate_probe_paths_with_root_returns_err_6_for_nonexistent_cache_root() {
    let cache_dir = tempfile::tempdir().unwrap();
    let (model, config, tokenizer) = write_three_artifacts(cache_dir.path());
    let bogus_root = PathBuf::from("/nonexistent/rurico-test-cache-root");

    let result = super::validate_probe_paths_with_cache(&bogus_root, &model, &config, &tokenizer);

    assert_eq!(
        result,
        Err(SetupReason::CacheRootInvalid),
        "expected Err(SetupReason::CacheRootInvalid) when cache_root cannot be canonicalized"
    );
}

#[cfg(unix)]
#[test]
fn validate_probe_paths_with_root_accepts_symlink_under_cache_root() {
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

    let result =
        super::validate_probe_paths_with_cache(cache_dir.path(), &model, &config, &tokenizer);

    assert_eq!(result, Ok(()));
}

// `cache_x_evil/...` shares a string prefix with `cache_x` but is NOT a
// component-wise descendant. PathBuf::starts_with handles this correctly via
// component comparison; the test guards against a regression to byte-prefix
// comparison.
#[test]
fn validate_probe_paths_with_root_rejects_string_prefix_sibling() {
    let workspace = tempfile::tempdir().unwrap();
    let cache_root = workspace.path().join("cache_x");
    let evil_root = workspace.path().join("cache_x_evil");
    fs::create_dir_all(&cache_root).unwrap();
    fs::create_dir_all(&evil_root).unwrap();

    let (_, config, tokenizer) = write_three_artifacts(&cache_root);
    let evil_model = evil_root.join("model.bin");
    fs::write(&evil_model, b"evil").unwrap();

    let result =
        super::validate_probe_paths_with_cache(&cache_root, &evil_model, &config, &tokenizer);

    assert_eq!(
        result,
        Err(SetupReason::PathOutsideCache),
        "component-wise prefix check must reject `cache_x_evil` against `cache_x`"
    );
}

// validate_probe_paths (production wrapper) HF_HOME unset fallback (FR-101 + FR-006)
//
// The production wrapper resolves cache_root via `hf_hub::resolve_cache_dir()`.
// When `HF_HUB_CACHE` and `HF_HOME` are unset, hf-hub falls back to
// `dirs::home_dir() / .cache/huggingface/hub`. On Unix, `dirs::home_dir()`
// reads `$HOME` first (verified against `dirs-sys-0.5.0/src/lib.rs` line
// 33-71). Setting `HOME=tempdir` redirects the fallback into the test's
// tempdir.
//
// NOTE: `validate_probe_paths` MUST call `resolve_cache_dir()` at call time. A
// `LazyLock` / `OnceLock` cache root would poison parallel tests (the first
// caller fixes the value crate-wide).
#[cfg(unix)]
#[test]
fn validate_probe_paths_falls_back_to_home_when_hf_home_unset() {
    let home_dir = tempfile::tempdir().unwrap();
    let hub_dir = home_dir.path().join(".cache/huggingface/hub");
    let snapshot_dir = hub_dir.join("models--cl-nagoya--ruri-v3-310m/snapshots/commit-xyz");
    fs::create_dir_all(&snapshot_dir).unwrap();
    let (model, config, tokenizer) = write_three_artifacts(&snapshot_dir);

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

// T-007 / T-007a / T-008 / T-009: setup-phase exit codes 3..=6 each map to
// ProbeError::SetupRejected with the typed reason preserved.
#[test]
fn interpret_probe_output_maps_setup_exit_codes_to_setup_rejected() {
    for code in [
        PROBE_EXIT_ENV_INCOMPLETE,
        PROBE_EXIT_CANONICALIZE_FAILED,
        PROBE_EXIT_PATH_OUTSIDE_CACHE,
        PROBE_EXIT_CACHE_ROOT_INVALID,
    ] {
        let output = Output {
            status: exit_status(code),
            stdout: format!("{PROBE_ACK}\n").into_bytes(),
            stderr: Vec::new(),
        };
        let err = interpret_probe_output(&output).unwrap_err();
        assert!(
            matches!(err, ProbeError::SetupRejected { reason } if reason.code() == code),
            "exit {code}: expected SetupRejected with reason.code() == {code}, got {err}"
        );
    }
}

// T-011a / T-011b / T-011c / T-011d: SetupRejected Display surfaces both the
// exit code and a human-readable label per variant. The wire format
// "probe setup rejected (code N: label)" is contractual — forensic logs
// rely on the numeric code being present.
#[test]
fn setup_rejected_display_includes_code_and_label_per_variant() {
    let cases: &[(SetupReason, &str)] = &[
        (SetupReason::EnvIncomplete, "env incomplete"),
        (SetupReason::CanonicalizeFailed, "path canonicalize failed"),
        (SetupReason::PathOutsideCache, "path outside cache"),
        (SetupReason::CacheRootInvalid, "cache root invalid"),
    ];
    for &(reason, label) in cases {
        let s = format!("{}", ProbeError::SetupRejected { reason });
        assert!(
            s.contains(&reason.code().to_string()),
            "{reason:?}: Display must contain the exit code, got: {s}"
        );
        assert!(
            s.contains(label),
            "{reason:?}: Display must contain label {label:?}, got: {s}"
        );
    }
}

#[test]
fn setup_reason_code_matches_probe_exit_constant_per_variant() {
    assert_eq!(SetupReason::EnvIncomplete.code(), PROBE_EXIT_ENV_INCOMPLETE);
    assert_eq!(
        SetupReason::CanonicalizeFailed.code(),
        PROBE_EXIT_CANONICALIZE_FAILED
    );
    assert_eq!(
        SetupReason::PathOutsideCache.code(),
        PROBE_EXIT_PATH_OUTSIDE_CACHE
    );
    assert_eq!(
        SetupReason::CacheRootInvalid.code(),
        PROBE_EXIT_CACHE_ROOT_INVALID
    );
}

#[test]
fn setup_reason_try_from_round_trips_for_every_variant() {
    for reason in [
        SetupReason::EnvIncomplete,
        SetupReason::CanonicalizeFailed,
        SetupReason::PathOutsideCache,
        SetupReason::CacheRootInvalid,
    ] {
        assert_eq!(
            SetupReason::try_from(reason.code()),
            Ok(reason),
            "round-trip must yield Ok({reason:?})"
        );
    }
}

#[test]
fn setup_reason_try_from_returns_err_for_non_setup_exit_codes() {
    for code in [
        0,
        1,
        2,
        PROBE_EXIT_ACK_FAILED,
        PROBE_EXIT_STDERR_FAILED,
        9,
        999,
        -1,
    ] {
        assert!(
            SetupReason::try_from(code).is_err(),
            "code {code} must NOT map to any SetupReason variant"
        );
    }
}

// child stderr forensic log (SEC-002 dynamic message) reads from
// label() — guard against an accidental rename that would silently shift it.
#[test]
fn setup_reason_label_is_stable_per_variant() {
    let cases: &[(SetupReason, &str)] = &[
        (SetupReason::EnvIncomplete, "env incomplete"),
        (SetupReason::CanonicalizeFailed, "path canonicalize failed"),
        (SetupReason::PathOutsideCache, "path outside cache"),
        (SetupReason::CacheRootInvalid, "cache root invalid"),
    ];
    for &(reason, expected) in cases {
        assert_eq!(reason.label(), expected, "{reason:?} label drift");
    }
}

#[test]
fn setup_reason_display_combines_code_and_label() {
    // Full literal (code 5 == PROBE_EXIT_PATH_OUTSIDE_CACHE) so a change to
    // Display's format string is caught, not mirrored by rebuilding it here.
    assert_eq!(
        format!("{}", SetupReason::PathOutsideCache),
        "code 5: path outside cache",
    );
}

// FORWARD is exactly this allowlist, in declaration order. Equality
// (not len + contains) so an extra key — an env-injection vector — fails the
// test instead of slipping past a containment-only check.
#[test]
fn forward_list_is_exact_allowlist() {
    let expected: &[&str] = &[
        "PATH",
        "HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HF_ENDPOINT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
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
    assert_eq!(super::FORWARD, expected);
}

// child_env_for_spawn forwards HF_HOME, drops attacker-injected env
#[test]
fn child_env_for_spawn_drops_attacker_env_keeps_forward() {
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

// child_env_for_spawn skips undefined FORWARD keys silently (FR-102)
#[test]
fn child_env_for_spawn_skips_undefined_forward_keys() {
    temp_env::with_vars([("HF_HOME", None::<&str>)], || {
        let env = super::child_env_for_spawn(&[]);
        assert!(
            !env.contains_key("HF_HOME"),
            "HF_HOME must NOT be in map when undefined in parent"
        );
    });
}

// e2e — `probe_via_subprocess_with` actually applies env_clear + FORWARD
// to a real spawned child (TC-001 from /audit). Spawns a sh script that dumps its
// env to a tempfile, then asserts attacker-injected env is absent and FORWARD
// keys propagate.
#[cfg(unix)]
#[test]
fn probe_via_subprocess_with_env_clear_blocks_attacker_env_in_real_child() {
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

#[test]
fn emit_ack_to_writes_ack_token_with_newline() {
    let mut buf: Vec<u8> = Vec::new();
    let result = super::emit_ack_to(&mut buf);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    assert_eq!(buf, format!("{PROBE_ACK}\n").into_bytes());
}

// `FailingWriter::write` fails first and `?` short-circuits before `flush`
// runs, so this test only exercises the write-failure arm. The flush-only
// arm (write success + flush error) is not unit-tested; in production the
// pipe-broken case typically surfaces at write time on `io::stdout()`.
#[test]
fn emit_ack_to_propagates_writer_error() {
    let err = super::emit_ack_to(&mut test_writers::FailingWriter)
        .expect_err("expected emit_ack_to to propagate writer error");
    assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
}

// interpret_probe_output maps exit 7 to SubprocessFailed even without ACK
//
// IO infrastructure failures (ACK write itself failed) MUST be detected
// before the ACK presence check — otherwise an empty stdout caused by a
// failed ACK write would be misclassified as `HandlerNotInstalled`.
#[test]
fn interpret_probe_output_maps_exit_7_to_subprocess_failed_even_without_ack() {
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
}

// interpret_probe_output maps exit 8 to SubprocessFailed when ACK is present
//
// `dispatch_probe` only emits exit 8 after `emit_ack` succeeds, so a real
// probe failure of this kind always carries the ACK in stdout.
#[test]
fn interpret_probe_output_maps_exit_8_to_subprocess_failed() {
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
}

// exit 8 without ACK preserves HandlerNotInstalled diagnostic
//
// Regression guard: if a host binary lacks the probe handler and happens to
// exit with code 8, the missing ACK must classify it as
// `HandlerNotInstalled`, not `SubprocessFailed`. Exit 8 is meaningful only
// when emitted by `dispatch_probe` after a successful ACK; an exit 8 from
// any other code path predates the probe contract.
#[test]
fn interpret_probe_output_exit_8_without_ack_is_handler_not_installed() {
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

#[test]
fn emit_failure_to_writes_message_and_flushes() {
    let mut w = test_writers::FlushTrackingWriter::default();
    super::emit_failure_to(&mut w, "model load failed: bad weights").unwrap();
    assert_eq!(w.buf, b"model load failed: bad weights");
    assert_eq!(
        w.flush_count, 1,
        "emit_failure_to must call flush exactly once after the write"
    );
}

#[test]
fn emit_failure_to_propagates_writer_error() {
    let err = super::emit_failure_to(&mut test_writers::FailingWriter, "anything")
        .expect_err("expected emit_failure_to to propagate writer error");
    assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
}

// emit_failure_to surfaces flush errors after a successful write — the
// failure mode `emit_failure_to` exists to detect (bytes accepted into a
// buffer but never delivered to the kernel before `process::exit`).
#[test]
fn emit_failure_to_surfaces_flush_error_after_successful_write() {
    let mut w = test_writers::FlushFailingWriter::default();
    let err = super::emit_failure_to(&mut w, "msg")
        .expect_err("expected emit_failure_to to surface flush error");
    assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(
        w.buf, b"msg",
        "write must succeed before flush is attempted"
    );
}

#[test]
fn probe_reports_late_ack_and_truncated_failure_without_changing_classification() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("verbose-probe.sh");
    fs::write(&script, format!("#!/bin/sh\nhead -c 524288 /dev/zero\nprintf '\\n{PROBE_ACK}\\n'\nprintf 'actual model failure\\n' >&2\nhead -c 524288 /dev/zero >&2\nexit 1\n")).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let error = probe_via_subprocess_with(script.clone(), &[]).unwrap_err();
    let ProbeError::ModelLoadFailed { reason } = error else {
        panic!("late ACK changed error classification: {error}");
    };
    assert!(reason.starts_with("actual model failure"));
    assert!(reason.contains("output truncated at 256 KiB"));
    assert!(reason.len() <= 256 * 1024);
    for diagnostic in [
        format!("diagnostic mentions {PROBE_ACK} without a handshake"),
        format!("{PROBE_ACK} is disabled"),
        format!("startup\n{PROBE_ACK} is disabled"),
        PROBE_ACK.to_owned(), // Missing the newline emitted by the handler.
    ] {
        fs::write(
            &script,
            format!("#!/bin/sh\nprintf '%s' '{diagnostic}'\nexit 0\n"),
        )
        .unwrap();
        assert!(
            matches!(
                probe_via_subprocess_with(script.clone(), &[]),
                Err(ProbeError::HandlerNotInstalled)
            ),
            "accepted diagnostic as ACK: {diagnostic:?}"
        );
    }
}
