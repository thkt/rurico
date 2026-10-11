//! Shared bounded lifetime for re-executed probe and download workers.

use std::io::{self, Read};
use std::mem;
use std::os::fd::AsFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use rurico_ffi::{kill_process_group, process_group_exists, set_nonblocking};

pub(crate) const OUTPUT_LIMIT: usize = 256 * 1024;
pub(crate) const CLEANUP_GRACE: Duration = Duration::from_secs(2);
const TRUNCATED: &[u8] = b"\n[rurico: output truncated at 256 KiB]\n";

pub(crate) struct Collected {
    pub output: Output,
    pub ack: bool,
    pub timed_out: bool,
    pub stdout_truncated: bool,
}

/// Offset immediately after a complete ACK line, never a token mention.
/// `ack` includes the newline written by the dispatcher.
pub(crate) fn ack_payload_start(bytes: &[u8], ack: &[u8]) -> Option<usize> {
    ack_line_end(bytes, ack, true)
}

fn ack_line_end(bytes: &[u8], ack: &[u8], at_stream_start: bool) -> Option<usize> {
    if ack.is_empty() {
        return None;
    }
    bytes
        .windows(ack.len())
        .enumerate()
        .find_map(|(offset, frame)| {
            (frame == ack
                && ((offset == 0 && at_stream_start) || (offset > 0 && bytes[offset - 1] == b'\n')))
                .then_some(offset + ack.len())
        })
}

struct Capture<R> {
    pipe: Option<R>,
    bytes: Vec<u8>,
    truncated: bool,
    ack_tail: Vec<u8>,
    ack: bool,
    setup_error: Option<io::Error>,
}

impl<R: Read + AsFd> Capture<R> {
    fn new(mut pipe: Option<R>) -> Self {
        let setup_error = pipe
            .as_ref()
            .and_then(|pipe| set_nonblocking(pipe.as_fd()).err());
        if setup_error.is_some() {
            pipe = None;
        }
        Self {
            pipe,
            bytes: Vec::new(),
            truncated: false,
            ack_tail: vec![b'\n'],
            ack: false,
            setup_error,
        }
    }

    fn drain(&mut self, ack: &[u8]) -> io::Result<()> {
        if let Some(error) = self.setup_error.take() {
            return Err(error);
        }
        let mut buf = [0; 8192];
        // A continuously writing child must not starve the other stream or
        // deadline checks. Each iteration drains at most 64 KiB per stream.
        for _ in 0..8 {
            let Some(pipe) = self.pipe.as_mut() else {
                break;
            };
            match pipe.read(&mut buf) {
                Ok(0) => {
                    self.pipe = None;
                    break;
                }
                Ok(n) => {
                    if !ack.is_empty() && !self.ack {
                        self.ack_tail.extend_from_slice(&buf[..n]);
                        self.ack = ack_line_end(&self.ack_tail, ack, false).is_some();
                        let keep_from = self.ack_tail.len().saturating_sub(ack.len());
                        self.ack_tail.drain(..keep_from);
                    }
                    let keep = n.min(OUTPUT_LIMIT - self.bytes.len());
                    self.bytes.extend_from_slice(&buf[..keep]);
                    self.truncated |= keep != n;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.pipe = None;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.truncated {
            self.bytes.truncate(OUTPUT_LIMIT - TRUNCATED.len());
            self.bytes.extend_from_slice(TRUNCATED);
        }
        mem::take(&mut self.bytes)
    }
}

pub(crate) fn spawn(cmd: &mut Command) -> io::Result<Child> {
    cmd.process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// The child must have been spawned in its own process group via `spawn`.
/// No reader threads exist; every return closes both owned pipe descriptors.
pub(crate) fn collect(
    child: &mut Child,
    timeout: Duration,
    poll: Duration,
    ack: &[u8],
) -> io::Result<Collected> {
    let deadline = Instant::now() + timeout;
    let mut stdout = Capture::new(child.stdout.take());
    let mut stderr = Capture::new(child.stderr.take());
    collect_captures(
        child,
        timeout,
        poll,
        ack,
        &mut stdout,
        &mut stderr,
        deadline,
    )
}

fn collect_captures<O: Read + AsFd, E: Read + AsFd>(
    child: &mut Child,
    timeout: Duration,
    poll: Duration,
    ack: &[u8],
    stdout: &mut Capture<O>,
    stderr: &mut Capture<E>,
    deadline: Instant,
) -> io::Result<Collected> {
    collect_captures_with_group_ops(
        child,
        (timeout, poll, deadline),
        ack,
        stdout,
        stderr,
        (kill_process_group, process_group_exists),
    )
}

fn collect_captures_with_group_ops<O: Read + AsFd, E: Read + AsFd>(
    child: &mut Child,
    timing: (Duration, Duration, Instant),
    ack: &[u8],
    stdout: &mut Capture<O>,
    stderr: &mut Capture<E>,
    (mut kill_group, mut observe_group): (
        impl FnMut(u32) -> io::Result<()>,
        impl FnMut(u32) -> io::Result<bool>,
    ),
) -> io::Result<Collected> {
    let (timeout, poll, deadline) = timing;
    let mut status = None;
    let mut group_reclaimed = false;
    let mut group_exists = true;
    let mut cleanup_deadline = None;
    let mut failure = None;
    let mut signal_failure = None;
    let mut timed_out = false;
    loop {
        for result in [stdout.drain(ack), stderr.drain(&[])] {
            if let Err(e) = result {
                failure.get_or_insert(e);
            }
        }
        if status.is_none() {
            match child.try_wait() {
                Ok(s) => status = s,
                Err(e) => {
                    failure.get_or_insert(e);
                }
            }
        }
        if cleanup_deadline.is_none()
            && (status.is_some() || failure.is_some() || Instant::now() >= deadline)
        {
            timed_out = status.is_none() && failure.is_none();
            if timed_out {
                tracing::warn!(?timeout, "owned subprocess processing deadline exceeded");
            }
            cleanup_deadline = Some(Instant::now() + CLEANUP_GRACE);
        }
        if let Some(end) = cleanup_deadline {
            // Retry fork races only until reap and group absence are confirmed.
            // Afterwards the PGID may be reused by an unrelated process group.
            if !group_reclaimed {
                if let Err(e) = kill_group(child.id()) {
                    let e = io::Error::new(
                        e.kind(),
                        format!("kill(-{}, SIGKILL) failed: {e}", child.id()),
                    );
                    if signal_failure.is_none() || e.kind() != io::ErrorKind::PermissionDenied {
                        signal_failure = Some(e);
                    }
                }
                group_exists = match observe_group(child.id()) {
                    Ok(exists) => exists,
                    Err(e) => {
                        failure.get_or_insert(io::Error::new(
                            e.kind(),
                            format!("kill(-{}, 0) group observation failed: {e}", child.id()),
                        ));
                        true
                    }
                };
                group_reclaimed = status.is_some() && !group_exists;
            }
            if let Some(exit_status) = status
                && group_reclaimed
                && stdout.pipe.is_none()
                && stderr.pipe.is_none()
            {
                if let Some(e) = failure {
                    return Err(e);
                }
                // EPERM can mean only zombies remain on Darwin. It is resolved
                // only by confirmed group absence, child reap and both EOFs.
                if let Some(e) = signal_failure
                    && e.kind() != io::ErrorKind::PermissionDenied
                {
                    return Err(e);
                }
                if stdout.truncated || stderr.truncated {
                    tracing::warn!(
                        stdout_truncated = stdout.truncated,
                        stderr_truncated = stderr.truncated,
                        limit = OUTPUT_LIMIT,
                        "owned subprocess output truncated"
                    );
                }
                let stdout_truncated = stdout.truncated;
                let ack = stdout.ack;
                return Ok(Collected {
                    output: Output {
                        status: exit_status,
                        stdout: stdout.finish(),
                        stderr: stderr.finish(),
                    },
                    ack,
                    timed_out,
                    stdout_truncated,
                });
            }
            if Instant::now() >= end {
                let cause = failure.as_ref().or(signal_failure.as_ref());
                return Err(io::Error::new(
                    cause.map_or(io::ErrorKind::Other, io::Error::kind),
                    format!(
                        "process group {} cleanup exceeded 2 second grace (child reaped: {}, group exists: {group_exists}, group reclaimed: {group_reclaimed}, stdout closed: {}, stderr closed: {}); original failure: {}; signal failure: {}",
                        child.id(),
                        status.is_some(),
                        stdout.pipe.is_none(),
                        stderr.pipe.is_none(),
                        failure
                            .as_ref()
                            .map_or_else(|| "none".to_owned(), ToString::to_string),
                        signal_failure
                            .as_ref()
                            .map_or_else(|| "none".to_owned(), ToString::to_string)
                    ),
                ));
            }
        }
        thread::sleep(poll.min(Duration::from_millis(10)));
    }
}

#[cfg(test)]
pub(crate) mod tests;
