use super::*;
use std::env;
use std::fs;
use std::io::Write;
use std::os::fd::BorrowedFd;
use std::process::{self, ChildStdout};

// Only the selected test runs in the observed process, even under parallel Cargo.
pub(crate) fn reexec_for_thread_observation(test_name: &str) -> bool {
    const ISOLATED_TEST: &str = "RURICO_THREAD_OBSERVATION_TEST";
    if env::var(ISOLATED_TEST).as_deref() == Ok(test_name) {
        return false;
    }
    let mut cmd = Command::new(env::current_exe().unwrap());
    cmd.args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(ISOLATED_TEST, test_name);
    let mut child = spawn(&mut cmd).unwrap();
    let result = collect(
        &mut child,
        Duration::from_secs(20),
        Duration::from_millis(1),
        b"",
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&result.output.stdout);
    let stderr = String::from_utf8_lossy(&result.output.stderr);
    assert!(
        !result.timed_out && result.output.status.success(),
        "isolated test {test_name} failed: {stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("running 1 test"),
        "isolated test was not selected: {stdout}"
    );
    true
}

pub(crate) fn thread_count() -> usize {
    let result = Command::new("ps")
        .args(["-M", "-p", &process::id().to_string()])
        .output()
        .unwrap();
    assert!(result.status.success());
    let rows = String::from_utf8(result.stdout).unwrap();
    assert!(
        rows.lines().count() >= 2,
        "missing process/thread observation: {rows}"
    );
    rows.lines().count() - 1
}

fn collect_script(script: &str, timeout: Duration, ack: &[u8]) -> Collected {
    let mut cmd = Command::new("sh");
    cmd.args(["-c", script]);
    let mut child = spawn(&mut cmd).unwrap();
    let result = collect(&mut child, timeout, Duration::from_millis(1), ack).unwrap();
    assert!(!process_group_exists(child.id()).unwrap());
    result
}

#[test]
fn simultaneous_large_streams_keep_late_ack_and_exit_reason_with_bounded_memory() {
    let result = collect_script(
        "head -c 524288 /dev/zero; printf '\\nRURICO_PROBE_OK\\n'; head -c 524288 /dev/zero >&2; exit 1",
        Duration::from_secs(5),
        b"RURICO_PROBE_OK",
    );
    assert!(result.ack, "ACK outside retained output must survive");
    assert_eq!(result.output.status.code(), Some(1));
    assert_eq!(result.output.stdout.len(), 256 * 1024);
    assert_eq!(result.output.stderr.len(), 256 * 1024);
    assert!(result.stdout_truncated);
    assert!(result.output.stdout.ends_with(TRUNCATED));
    assert!(result.output.stderr.ends_with(TRUNCATED));
}

#[test]
fn repeated_grandchild_fd_holders_stop_writes_without_reader_or_process_accumulation() {
    if reexec_for_thread_observation(
        "owned_process::tests::repeated_grandchild_fd_holders_stop_writes_without_reader_or_process_accumulation",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let before = thread_count();
    for i in 0..5 {
        let path = dir.path().join(format!("writes-{i}"));
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "(i=0; while [ \"$i\" -lt 1000 ]; do i=$((i+1)); printf x >> \"$1\"; sleep 0.01; done) & printf 'RURICO_PROBE_OK\\n'; exit 1", "fixture"]).arg(&path);
        let mut child = spawn(&mut cmd).unwrap();
        let start = Instant::now();
        let result = collect(
            &mut child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            b"RURICO_PROBE_OK",
        )
        .unwrap();
        assert_eq!(result.output.status.code(), Some(1));
        assert!(result.ack);
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(!process_group_exists(child.id()).unwrap());
        assert!(child.try_wait().unwrap().is_some());
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        thread::sleep(Duration::from_millis(30));
        assert_eq!(fs::metadata(&path).map(|m| m.len()).unwrap_or(0), size);
    }
    assert_eq!(thread_count(), before);
}

struct FailingPipe {
    pipe: ChildStdout,
    read_once: bool,
    kind: io::ErrorKind,
}
impl AsFd for FailingPipe {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.pipe.as_fd()
    }
}
impl Read for FailingPipe {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_once {
            return Err(io::Error::new(self.kind, "injected reader I/O failure"));
        }
        let n = self.pipe.read(buf)?;
        self.read_once |= n > 0;
        Ok(n)
    }
}

#[test]
fn reader_io_failure_uses_production_cleanup_and_preserves_other_group() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("unrelated");
    let mut other_cmd = Command::new("sh");
    other_cmd
        .args([
            "-c",
            "i=0; while [ \"$i\" -lt 1000 ]; do i=$((i+1)); printf x >> \"$1\"; sleep 0.01; done",
            "fixture",
        ])
        .arg(&marker);
    let mut other = spawn(&mut other_cmd).unwrap();
    let outcome = {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf 'ACK'; sleep 30"]);
        let mut child = spawn(&mut cmd).unwrap();
        let mut stdout = Capture::new(Some(FailingPipe {
            pipe: child.stdout.take().unwrap(),
            read_once: false,
            kind: io::ErrorKind::Other,
        }));
        let mut stderr = Capture::new(child.stderr.take());
        let error = collect_captures(
            &mut child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            b"ACK",
            &mut stdout,
            &mut stderr,
            Instant::now() + Duration::from_secs(5),
        )
        .err()
        .unwrap();
        let reaped = child.try_wait().unwrap().is_some();
        let group_gone = !process_group_exists(child.id()).unwrap();
        let size = fs::metadata(&marker).map(|m| m.len()).unwrap_or(0);
        thread::sleep(Duration::from_millis(50));
        let grew = fs::metadata(&marker).unwrap().len() > size;
        (error, reaped, group_gone, grew)
    };
    let _ = collect(&mut other, Duration::ZERO, Duration::from_millis(1), b"").unwrap();
    assert!(
        outcome
            .0
            .to_string()
            .contains("injected reader I/O failure")
    );
    assert_eq!(outcome.0.kind(), io::ErrorKind::Other);
    assert!(
        outcome.1 && outcome.2,
        "failing reader left process behind (child reaped: {}, group gone: {}): {}",
        outcome.1,
        outcome.2,
        outcome.0
    );
    assert!(outcome.3, "cleanup killed unrelated writer");
}

#[test]
fn group_operation_failures_preserve_cause_after_real_reclamation() {
    use std::cell::Cell;

    for (signal_kind, observation_fails) in [
        (io::ErrorKind::PermissionDenied, false),
        (io::ErrorKind::Other, false),
        (io::ErrorKind::PermissionDenied, true),
        (io::ErrorKind::Other, true),
    ] {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf 'ACK\\n'; sleep 30"]);
        let mut child = spawn(&mut cmd).unwrap();
        let mut stdout = Capture::new(child.stdout.take());
        let mut stderr = Capture::new(child.stderr.take());
        let signal_calls = Cell::new(0);
        let observation_calls = Cell::new(0);
        let start = Instant::now();
        let result = collect_captures_with_group_ops(
            &mut child,
            (
                Duration::from_millis(50),
                Duration::from_millis(1),
                start + Duration::from_millis(50),
            ),
            b"ACK\n",
            &mut stdout,
            &mut stderr,
            (
                |pgid| {
                    let calls = signal_calls.get();
                    signal_calls.set(calls + 1);
                    kill_process_group(pgid)?;
                    // A subsequent EPERM must not erase an earlier signal fault.
                    let kind = if calls == 0 {
                        signal_kind
                    } else {
                        io::ErrorKind::PermissionDenied
                    };
                    Err(io::Error::new(kind, "injected signal failure"))
                },
                |pgid| {
                    let exists = process_group_exists(pgid)?;
                    let calls = observation_calls.get();
                    observation_calls.set(calls + 1);
                    if calls == 0 && observation_fails {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "injected group observation failure",
                        ))
                    } else {
                        Ok(exists)
                    }
                },
            ),
        );
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(child.try_wait().unwrap().is_some());
        assert!(!process_group_exists(child.id()).unwrap());
        assert!(stdout.pipe.is_none() && stderr.pipe.is_none());
        assert!(signal_calls.get() > 0 && observation_calls.get() > 0);
        if observation_fails {
            assert!(signal_calls.get() >= 2 && observation_calls.get() >= 2);
            let error = result.err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains("group observation failed"));
            assert!(
                error
                    .to_string()
                    .contains("injected group observation failure")
            );
        } else if signal_kind == io::ErrorKind::Other {
            assert!(signal_calls.get() >= 2);
            let error = result.err().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert!(error.to_string().contains("SIGKILL"));
            assert!(error.to_string().contains("injected signal failure"));
        } else {
            let collected = result.unwrap();
            assert!(collected.timed_out);
            assert!(!collected.output.status.success());
        }
    }
}

#[test]
fn unrecoverable_open_pipe_returns_cleanup_error_within_grace() {
    use std::cell::Cell;
    use std::os::unix::net::UnixStream;
    for fail_reader in [false, true] {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf ACK; exit 0"]);
        let mut child = spawn(&mut cmd).unwrap();
        let (reader, mut outside_writer) = UnixStream::pair().unwrap();
        outside_writer.write_all(b"ACK").unwrap();
        // This FD holder is outside the owned process group. Stopping that group
        // cannot close it; the production collector must report incomplete cleanup.
        let mut stdout = Capture::new(Some(reader));
        let start = Instant::now();
        let absence_observed = Cell::new(false);
        let kill_calls = Cell::new(0);
        let observation_calls = Cell::new(0);
        let kill_group = |pgid| {
            assert!(
                !absence_observed.get(),
                "SIGKILL after confirmed group absence"
            );
            kill_calls.set(kill_calls.get() + 1);
            kill_process_group(pgid)
        };
        let observe_group = |pgid| {
            assert!(
                !absence_observed.get(),
                "signal 0 after confirmed group absence"
            );
            observation_calls.set(observation_calls.get() + 1);
            let exists = process_group_exists(pgid)?;
            absence_observed.set(!exists);
            Ok(exists)
        };
        let result = if fail_reader {
            let mut stderr = Capture::new(Some(FailingPipe {
                pipe: child.stdout.take().unwrap(),
                read_once: false,
                kind: io::ErrorKind::PermissionDenied,
            }));
            collect_captures_with_group_ops(
                &mut child,
                (
                    Duration::from_secs(5),
                    Duration::from_millis(1),
                    Instant::now() + Duration::from_secs(5),
                ),
                b"ACK",
                &mut stdout,
                &mut stderr,
                (kill_group, observe_group),
            )
        } else {
            let mut stderr = Capture::new(child.stderr.take());
            collect_captures_with_group_ops(
                &mut child,
                (
                    Duration::from_secs(5),
                    Duration::from_millis(1),
                    Instant::now() + Duration::from_secs(5),
                ),
                b"ACK",
                &mut stdout,
                &mut stderr,
                (kill_group, observe_group),
            )
        };
        let error = result.err().unwrap();
        assert!(
            absence_observed.get(),
            "owned group absence was never observed"
        );
        assert!(kill_calls.get() > 0);
        assert!(observation_calls.get() > 0);
        assert!(error.to_string().contains("group exists: false"));
        assert!(error.to_string().contains("group reclaimed: true"));
        assert!(error.to_string().contains("stdout closed: false"));
        assert!(
            error
                .to_string()
                .contains("cleanup exceeded 2 second grace"),
            "{error}"
        );
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(child.try_wait().unwrap().is_some());
        assert!(!process_group_exists(child.id()).unwrap());
        if fail_reader {
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
            assert!(error.to_string().contains("injected reader I/O failure"));
        }
    }
}

#[test]
fn continuous_output_cannot_starve_processing_deadline() {
    let mut cmd = Command::new("sh");
    cmd.args(["-c", "printf 'ACK'; yes x & yes y >&2 & wait"]);
    let mut child = spawn(&mut cmd).unwrap();
    let start = Instant::now();
    let result = collect(
        &mut child,
        Duration::from_millis(100),
        Duration::from_millis(1),
        b"ACK",
    )
    .unwrap();
    assert!(result.timed_out && result.ack);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(result.output.stdout.len() <= 256 * 1024);
    assert!(result.output.stderr.len() <= 256 * 1024);
    assert!(!process_group_exists(child.id()).unwrap());
}

#[test]
fn complete_ack_lines_survive_one_byte_reads_without_accepting_mentions() {
    use std::os::unix::net::UnixStream;
    struct OneByte(UnixStream);
    impl AsFd for OneByte {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.0.as_fd()
        }
    }
    impl Read for OneByte {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(&mut buf[..1])
        }
    }
    for (output, expected) in [
        ("RURICO_PROBE_OK\n", true),
        ("startup\nRURICO_PROBE_OK\n", true),
        ("RURICO_PROBE_OK is disabled\n", false),
        ("startup\nRURICO_PROBE_OK is disabled\n", false),
        ("startup RURICO_PROBE_OK\n", false),
        ("RURICO_PROBE_OK", false),
    ] {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(output.as_bytes()).unwrap();
        drop(writer);
        let mut capture = Capture::new(Some(OneByte(reader)));
        while capture.pipe.is_some() {
            capture.drain(b"RURICO_PROBE_OK\n").unwrap();
        }
        assert_eq!(capture.ack, expected, "{output:?}");
    }
}
