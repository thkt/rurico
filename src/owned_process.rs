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
    let mut status = None;
    let mut cleanup_deadline = None;
    let mut failure = None;
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
            if let Err(e) = kill_process_group(child.id()) {
                failure.get_or_insert(e);
            }
        }
        if let Some(end) = cleanup_deadline {
            let group_exists = process_group_exists(child.id())?;
            if let Some(exit_status) = status
                && !group_exists
                && stdout.pipe.is_none()
                && stderr.pipe.is_none()
            {
                if let Some(e) = failure {
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
                return Err(io::Error::other(format!(
                    "process group {} cleanup exceeded 2 second grace (child reaped: {}, group exists: {group_exists})",
                    child.id(),
                    status.is_some()
                )));
            }
        }
        thread::sleep(poll.min(Duration::from_millis(10)));
    }
}

#[cfg(test)]
pub(crate) mod tests;
