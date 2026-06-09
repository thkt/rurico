//! Shared subprocess probe infrastructure for model loading verification.
//!
//! Host binaries call [`handle_probe_if_needed`](crate::handle_probe_if_needed)
//! once at the start of `main()` — that lives in [`crate::dispatch`] because it
//! needs to know about both embed and reranker domains. Individual modules
//! (embed, reranker) call [`probe_via_subprocess`] to re-exec the current
//! binary as an isolated probe.
//!
//! # Probe child exit codes
//!
//! | Code | Constant                          | Parent's interpretation              |
//! | ---- | --------------------------------- | ------------------------------------ |
//! | 0    | (success)                         | `ProbeStatus::Available`             |
//! | 1    | (model load failed)               | `ProbeError::ModelLoadFailed`        |
//! | 3    | `PROBE_EXIT_ENV_INCOMPLETE`       | `ProbeError::SetupRejected`          |
//! | 4    | `PROBE_EXIT_CANONICALIZE_FAILED`  | `ProbeError::SetupRejected`          |
//! | 5    | `PROBE_EXIT_PATH_OUTSIDE_CACHE`   | `ProbeError::SetupRejected`          |
//! | 6    | `PROBE_EXIT_CACHE_ROOT_INVALID`   | `ProbeError::SetupRejected`          |
//! | 7    | `PROBE_EXIT_ACK_FAILED`           | `ProbeError::SubprocessFailed` (IO)  |
//! | 8    | `PROBE_EXIT_STDERR_FAILED`        | `ProbeError::SubprocessFailed` (IO)  |
//!
//! Code 2 is intentionally unused. Killed by signal (no exit code) maps to
//! `ProbeStatus::BackendUnavailable`. Missing handshake ACK in stdout maps to
//! `ProbeError::HandlerNotInstalled` regardless of exit code (except code 7,
//! which is checked first because the ACK write itself failed).

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(test)]
use std::process::ExitStatus;

use crate::model_io::ModelPaths;

/// Handshake token written to stdout by probe subprocesses.
pub const PROBE_ACK: &str = "RURICO_PROBE_OK";

/// Exit code used when a probe subprocess was invoked with the primary model env var
/// set but the config or tokenizer env vars were missing.
pub(crate) const PROBE_EXIT_ENV_INCOMPLETE: i32 = 3;

/// Exit code returned by the probe child when `std::fs::canonicalize` fails on
/// any of the candidate paths (file missing, permission denied, symlink loop).
pub(crate) const PROBE_EXIT_CANONICALIZE_FAILED: i32 = 4;

/// Exit code returned by the probe child when a candidate path's canonical form
/// is not a component-wise descendant of the canonical cache root.
pub(crate) const PROBE_EXIT_PATH_OUTSIDE_CACHE: i32 = 5;

/// Exit code returned by the probe child when `std::fs::canonicalize` fails on
/// the cache root itself (HF cache directory missing or unreadable).
pub(crate) const PROBE_EXIT_CACHE_ROOT_INVALID: i32 = 6;

/// Exit code returned by the probe child when emitting the handshake ACK to
/// stdout fails (e.g., the parent's pipe reader dropped, or `flush()` fails).
///
/// Distinct from `HandlerNotInstalled`: this signals an IO infrastructure
/// failure inside the probe, not a missing handler installation.
pub(crate) const PROBE_EXIT_ACK_FAILED: i32 = 7;

/// Exit code returned by the probe child when writing the failure-reason
/// message to stderr fails. The reason is unobservable to the parent in this
/// case, but the exit code itself signals "stderr write failed" rather than
/// "model load failed with empty reason".
pub(crate) const PROBE_EXIT_STDERR_FAILED: i32 = 8;

/// Reason a probe subprocess was rejected during the setup phase. Each
/// variant maps 1:1 to a `PROBE_EXIT_*` constant via the `#[repr(i32)]`
/// discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
pub enum SetupReason {
    /// Primary model env var set, but config or tokenizer is missing.
    EnvIncomplete = PROBE_EXIT_ENV_INCOMPLETE,
    /// `std::fs::canonicalize` failed on a candidate path.
    CanonicalizeFailed = PROBE_EXIT_CANONICALIZE_FAILED,
    /// Candidate path canonicalized outside the HF cache root.
    PathOutsideCache = PROBE_EXIT_PATH_OUTSIDE_CACHE,
    /// `std::fs::canonicalize` failed on the cache root itself.
    CacheRootInvalid = PROBE_EXIT_CACHE_ROOT_INVALID,
}

impl SetupReason {
    /// Exit code for this reason.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// Human-readable label. Surfaced in the child stderr message
    /// `"probe setup rejected: <label>"` (forensic log anchor; SEC-002).
    pub fn label(self) -> &'static str {
        match self {
            Self::EnvIncomplete => "env incomplete",
            Self::CanonicalizeFailed => "path canonicalize failed",
            Self::PathOutsideCache => "path outside cache",
            Self::CacheRootInvalid => "cache root invalid",
        }
    }
}

impl TryFrom<i32> for SetupReason {
    type Error = ();

    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            PROBE_EXIT_ENV_INCOMPLETE => Ok(Self::EnvIncomplete),
            PROBE_EXIT_CANONICALIZE_FAILED => Ok(Self::CanonicalizeFailed),
            PROBE_EXIT_PATH_OUTSIDE_CACHE => Ok(Self::PathOutsideCache),
            PROBE_EXIT_CACHE_ROOT_INVALID => Ok(Self::CacheRootInvalid),
            _ => Err(()),
        }
    }
}

impl fmt::Display for SetupReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "code {}: {}", self.code(), self.label())
    }
}

/// Verify that `model`, `config`, and `tokenizer` paths all canonicalize to
/// component-wise descendants of `cache_root`. The candidate paths are
/// not modified — the load step receives the original symlink path so HF
/// cache snapshot symlinks (`snapshots/<commit>/<file>` -> `blobs/<etag>`)
/// keep their original filenames, preserving MLX's extension-based dispatch.
///
/// # Errors
///
/// - [`SetupReason::CacheRootInvalid`] if `cache_root` cannot be canonicalized
/// - [`SetupReason::CanonicalizeFailed`] if any candidate path cannot be canonicalized
/// - [`SetupReason::PathOutsideCache`] if any candidate path's canonical form
///   is not a component-wise descendant of `cache_root`'s canonical form
pub(crate) fn validate_probe_paths_with_cache(
    cache: &hf_hub::Cache,
    model: &Path,
    config: &Path,
    tokenizer: &Path,
) -> Result<(), SetupReason> {
    let canon_root = fs::canonicalize(cache.path()).map_err(|_| SetupReason::CacheRootInvalid)?;
    for p in [model, config, tokenizer] {
        let canon = fs::canonicalize(p).map_err(|_| SetupReason::CanonicalizeFailed)?;
        if !canon.starts_with(&canon_root) {
            return Err(SetupReason::PathOutsideCache);
        }
    }
    Ok(())
}

/// Production wrapper for [`validate_probe_paths_with_cache`] that resolves
/// the cache via [`hf_hub::Cache::from_env`].
///
/// `Cache::from_env()` reads `HF_HOME` and falls back to the user's default
/// `~/.cache/huggingface/hub`. The cache resolution happens at call time so
/// parallel tests using `temp_env::with_vars` to scope env vars do not poison
/// each other.
///
/// # Errors
///
/// Same as [`validate_probe_paths_with_cache`].
pub(crate) fn validate_probe_paths(
    model: &Path,
    config: &Path,
    tokenizer: &Path,
) -> Result<(), SetupReason> {
    validate_probe_paths_with_cache(&hf_hub::Cache::from_env(), model, config, tokenizer)
}

/// Result of a model probe — whether the backend can load the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Model loaded successfully.
    Available,
    /// Backend crashed or is unsupported on this hardware.
    BackendUnavailable,
}

/// Errors from probe subprocess operations.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    /// Probe handler not installed in host binary.
    #[error(
        "probe handler not installed; call rurico::handle_probe_if_needed() \
         at the start of main()"
    )]
    HandlerNotInstalled,
    /// Model load failed in probe subprocess.
    #[error("model load failed: {reason}")]
    ModelLoadFailed {
        /// Failure detail from stderr.
        reason: String,
    },
    /// Setup-phase rejection (path validation or env hardening) in the probe
    /// child. Display surfaces as `"probe setup rejected (code N: label)"`
    /// — the wire format forensic logs grep on.
    #[error("probe setup rejected ({reason})")]
    SetupRejected {
        /// Setup-phase rejection reason.
        reason: SetupReason,
    },
    /// Subprocess spawn or wait failure.
    #[error("probe error: {0}")]
    SubprocessFailed(String),
}

/// Resolve probe env vars into a `(model, config, tokenizer)` path triple.
///
/// Returns `None` if the primary model env var is absent (not a probe invocation).
/// Returns `Some(Err(SetupReason::EnvIncomplete))` if model is set but config or
/// tokenizer is missing.
pub(crate) fn resolve_probe_env(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<(PathBuf, PathBuf, PathBuf), SetupReason>> {
    let model = model?;
    Some(match (config, tokenizer) {
        (Some(config), Some(tokenizer)) => Ok((model.into(), config.into(), tokenizer.into())),
        _ => Err(SetupReason::EnvIncomplete),
    })
}

/// Timeout for probe subprocesses.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Default polling interval used by [`wait_with_timeout`] to check whether the
/// child has exited. Tests inject a smaller interval via
/// [`wait_with_timeout_with`] to keep timeout-path assertions fast.
const DEFAULT_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Exit action computed by [`compute_probe_exit`].
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) struct ProbeExitAction {
    pub(crate) code: i32,
    pub(crate) message: Option<String>,
}

/// Determine the exit code and stderr message for a probe dispatch.
///
/// - `Err(reason)`: setup rejected → exit with `reason.code()`
/// - `Ok(Ok(()))`: model loaded successfully → exit 0
/// - `Ok(Err(load_err))`: model load failed → exit 1 with the load error message
pub(crate) fn compute_probe_exit(
    result: Result<Result<(), String>, SetupReason>,
) -> ProbeExitAction {
    match result {
        Err(reason) => ProbeExitAction {
            code: reason.code(),
            // label() only — not full Display — so the stderr wire format
            // stays `"probe setup rejected: <label>"` (SEC-002 forensic log).
            message: Some(format!("probe setup rejected: {}", reason.label())),
        },
        Ok(Ok(())) => ProbeExitAction {
            code: 0,
            message: None,
        },
        Ok(Err(load_err)) => ProbeExitAction {
            code: 1,
            message: Some(load_err),
        },
    }
}

/// Execute a probe dispatch: emit ACK, load model, exit with appropriate code.
///
/// Always terminates the process via [`std::process::exit`]. IO failures
/// during ACK emission or stderr write surface as the dedicated exit codes
/// [`PROBE_EXIT_ACK_FAILED`] / [`PROBE_EXIT_STDERR_FAILED`] so the parent
/// can distinguish them from `HandlerNotInstalled` (caller forgot the probe
/// handler) or `ModelLoadFailed` (load itself failed with a reason).
pub(crate) fn dispatch_probe<P, E: fmt::Display>(
    result: Result<P, SetupReason>,
    load: impl FnOnce(P) -> Result<(), E>,
) -> ! {
    if emit_ack().is_err() {
        process::exit(PROBE_EXIT_ACK_FAILED);
    }
    let outcome = match result {
        Err(reason) => Err(reason),
        Ok(candidate) => Ok(load(candidate).map_err(|e| e.to_string())),
    };
    let action = compute_probe_exit(outcome);
    if let Some(ref msg) = action.message
        && emit_failure_to(&mut io::stderr(), msg).is_err()
    {
        process::exit(PROBE_EXIT_STDERR_FAILED);
    }
    process::exit(action.code);
}

fn emit_ack() -> io::Result<()> {
    emit_ack_to(&mut io::stdout())
}

/// Testable seam over [`emit_ack`]. Production calls `emit_ack` which writes
/// to `io::stdout()`; tests inject a `Vec<u8>` or a failing writer via this
/// entry point to cover the success and IO-failure paths without spawning a
/// subprocess.
fn emit_ack_to<W: Write>(stdout: &mut W) -> io::Result<()> {
    writeln!(stdout, "{PROBE_ACK}")?;
    stdout.flush()
}

/// Write a probe-failure reason to `stderr` and flush, returning any IO error.
///
/// Symmetric to [`emit_ack_to`] for the failure path. Probe IPC contract:
/// the child must complete `flush` before [`std::process::exit`] so the
/// parent's `collect_pipe` observes the full message regardless of stderr's
/// buffering policy.
fn emit_failure_to<W: Write>(stderr: &mut W, msg: &str) -> io::Result<()> {
    write!(stderr, "{msg}")?;
    stderr.flush()
}

/// Re-exec the current binary as a probe subprocess with the given env vars.
///
/// Uses `Command` (fork+exec internally) so the child gets a clean process
/// state, avoiding the async-signal-safety issues of bare fork().
///
/// - Exit 0 → [`ProbeStatus::Available`]
/// - Exit non-zero → [`ProbeError::ModelLoadFailed`] (reason from stderr)
/// - Killed by signal / timeout / no exit code → [`ProbeStatus::BackendUnavailable`]
pub fn probe_via_subprocess(env_pairs: &[(&str, &str)]) -> Result<ProbeStatus, ProbeError> {
    let exe = env::current_exe()
        .map_err(|e| ProbeError::SubprocessFailed(format!("cannot locate executable: {e}")))?;
    probe_via_subprocess_with(exe, env_pairs)
}

/// Allowlist of environment variable keys that the probe child inherits from
/// the parent. After [`Command::env_clear`], only these keys plus the caller's
/// `env_pairs` are applied to the spawn command, eliminating env-injection
/// vectors that bypass `__RURICO_PROBE_*` validation (e.g. `HF_HOME`
/// re-targeting).
///
/// New runtime dependencies that read environment variables must be added here
/// (with rationale) for the probe to function in their presence.
pub(super) const FORWARD: &[&str] = &[
    // exe lookup
    "PATH",
    // Cache::from_env default fallback root resolution (`<HOME>/.cache/huggingface/hub`)
    "HOME",
    // HF cache root override
    "HF_HOME",
    // HF cache root override (alt)
    "HF_HUB_CACHE",
    // private repo authentication (optional)
    "HF_TOKEN",
    // HF Hub endpoint override — enterprise / private mirror
    // (hf-hub-0.5.0 api/{tokio,sync}.rs:248-249)
    "HF_ENDPOINT",
    // HTTP(S) proxy — corp proxy on egress. reqwest 0.12 via hyper-util-0.1.20
    // (matcher.rs:230-234) reads upper- and lowercase pairs; only uppercase
    // forms are forwarded as they are canonical in modern setups (curl/wget/
    // pip convention). Add lowercase variants if a legacy setup surfaces.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    // OpenSSL CA bundle / dir — corp proxy with custom CA on Linux native-tls path
    // (macOS native-tls uses Security framework keychain, ignored there but harmless)
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    // macOS dynamic linker
    "DYLD_LIBRARY_PATH",
    // macOS dynamic linker fallback
    "DYLD_FALLBACK_LIBRARY_PATH",
    // Linux dynamic linker
    "LD_LIBRARY_PATH",
    // locale
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    // tokenizers / tempfile cache
    "TMPDIR",
    "TMP",
    // diagnostic
    "RUST_LOG",
    "RUST_BACKTRACE",
];

/// Compute the exact env map that [`probe_via_subprocess_with`] applies to
/// the child after [`Command::env_clear`]. Pure function — does not spawn a
/// child or mutate parent env. Tests assert against the returned map directly,
/// avoiding the Rust 2024 unsafe-`set_var` parallel-test problem.
///
/// The map combines two sources:
/// 1. Parent env vars matching keys in [`FORWARD`] (undefined keys are skipped silently)
/// 2. Caller `env_pairs` (overlays / overrides FORWARD entries)
pub(super) fn child_env_for_spawn(env_pairs: &[(&str, &str)]) -> HashMap<String, String> {
    let mut env = HashMap::new();
    for &key in FORWARD {
        if let Ok(value) = env::var(key) {
            env.insert(key.into(), value);
        }
    }
    for &(key, value) in env_pairs {
        env.insert(key.into(), value.into());
    }
    env
}

/// Like [`probe_via_subprocess`] but the executable to re-exec is provided
/// explicitly. Used by tests to avoid depending on `env::current_exe()`.
///
/// # IPC Contract
///
/// **Child** (`dispatch_probe`): every byte written to stdout (handshake
/// ACK) or stderr (failure reason) MUST be flushed before [`process::exit`].
/// `emit_ack_to` and `emit_failure_to` are the only sanctioned write paths
/// and both call `flush` before returning. A missed flush would not drop
/// bytes today (`io::stdout()` is line-buffered, `io::stderr()` is
/// unbuffered), but the contract guards against future buffering policy
/// changes that could leave bytes stranded in user-space buffers when the
/// child exits.
///
/// **Parent** (`wait_with_timeout` + `collect_pipe`): each piped stream is
/// drained by a dedicated reader thread held by `DrainHandle`. On `recv`
/// success the thread is `join`ed immediately so that O(N) probe
/// invocations do not accumulate OS threads. On `recv_timeout` (a
/// grandchild inherited the pipe FDs and keeps them open past the direct
/// child's exit) the reader is left running — the thread is blocked on
/// `read_to_end` and a `join` would inherit the same indefinite block.
/// The leak is acceptable because grandchild FD inheritance is an
/// exceptional scenario; the happy path reaps every reader.
pub fn probe_via_subprocess_with(
    exe: PathBuf,
    env_pairs: &[(&str, &str)],
) -> Result<ProbeStatus, ProbeError> {
    let mut cmd = Command::new(exe);
    cmd.env_clear();
    let env = child_env_for_spawn(env_pairs);
    for (key, value) in &env {
        cmd.env(key, value);
    }
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ProbeError::SubprocessFailed(format!("probe spawn failed: {e}")))?;

    let output = wait_with_timeout(&mut child, PROBE_TIMEOUT)?;
    interpret_probe_output(&output)
}

/// Reader-thread handle paired with the channel that delivers its drained
/// bytes. Returned from [`spawn_drain_pipe`] so that [`collect_pipe`] can
/// `join` the reader thread once the channel recv succeeds — preventing
/// thread accumulation across many probes (parent IPC contract, see
/// [`probe_via_subprocess_with`]).
struct DrainHandle {
    rx: Receiver<Vec<u8>>,
    join: thread::JoinHandle<()>,
}

fn spawn_drain_pipe<R>(pipe: Option<R>, label: &'static str) -> Option<DrainHandle>
where
    R: Read + Send + 'static,
{
    pipe.map(|mut stream| {
        let (tx, rx) = mpsc::channel();
        let join = thread::spawn(move || {
            let mut buf = Vec::new();
            if let Err(e) = stream.read_to_end(&mut buf) {
                tracing::warn!(label, error = %e, "probe: failed to drain child");
            }
            let _ = tx.send(buf);
        });
        DrainHandle { rx, join }
    })
}

/// Upper bound on how long `collect_pipe` waits for a reader thread's buffer.
///
/// Capped because a grandchild that inherited the pipe FDs prevents EOF on
/// the reader's `read_to_end` even after the direct child exits. See
/// `probe_via_subprocess_with` for the full IPC contract (parent + child).
const COLLECT_PIPE_TIMEOUT: Duration = Duration::from_secs(2);

fn collect_pipe(handle: Option<DrainHandle>, label: &str) -> Vec<u8> {
    let Some(DrainHandle { rx, join }) = handle else {
        return Vec::new();
    };
    match rx.recv_timeout(COLLECT_PIPE_TIMEOUT) {
        Ok(buf) => {
            // recv success means `tx.send(buf)` ran, which only happens after
            // `read_to_end` returned. The reader thread is finished or about
            // to finish — `join` returns promptly without blocking.
            let _ = join.join();
            buf
        }
        Err(RecvTimeoutError::Timeout) => {
            tracing::warn!(
                label,
                timeout_secs = COLLECT_PIPE_TIMEOUT.as_secs(),
                "probe: child drain timed out; reader thread will leak"
            );
            Vec::new()
        }
        Err(RecvTimeoutError::Disconnected) => {
            tracing::warn!(label, "probe: child reader thread dropped channel");
            Vec::new()
        }
    }
}

/// Re-exec the current binary as a probe subprocess, passing `paths` as env vars.
///
/// Thin wrapper that converts [`crate::model_io::ModelPaths`] fields to string
/// env-var pairs and delegates to [`probe_via_subprocess`].
pub(crate) fn probe_paths_via_subprocess(
    paths: &ModelPaths,
    model_key: &str,
    config_key: &str,
    tokenizer_key: &str,
) -> Result<ProbeStatus, ProbeError> {
    let model = paths.model.to_string_lossy();
    let config = paths.config.to_string_lossy();
    let tokenizer = paths.tokenizer.to_string_lossy();
    probe_via_subprocess(&[
        (model_key, model.as_ref()),
        (config_key, config.as_ref()),
        (tokenizer_key, tokenizer.as_ref()),
    ])
}

/// Wait for a child process with a timeout. Kill and return a synthetic
/// timeout output if the deadline is exceeded.
fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Result<Output, ProbeError> {
    wait_with_timeout_with(child, timeout, DEFAULT_WAIT_POLL_INTERVAL)
}

/// Like [`wait_with_timeout`] but the polling interval is provided explicitly.
/// Used by tests to keep timeout-path assertions fast (e.g. 1 ms).
fn wait_with_timeout_with(
    child: &mut Child,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<Output, ProbeError> {
    let deadline = Instant::now() + timeout;
    let stdout = spawn_drain_pipe(child.stdout.take(), "stdout");
    let stderr = spawn_drain_pipe(child.stderr.take(), "stderr");

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Ok(Output {
                    status,
                    stdout: collect_pipe(stdout, "stdout"),
                    stderr: collect_pipe(stderr, "stderr"),
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    tracing::warn!(
                        timeout_secs = timeout.as_secs(),
                        "probe subprocess timed out, killing"
                    );
                    let _ = child.kill();
                    let status = child.wait().map_err(|e| {
                        ProbeError::SubprocessFailed(format!("probe wait after kill failed: {e}"))
                    })?;
                    return Ok(Output {
                        status,
                        stdout: collect_pipe(stdout, "stdout"),
                        stderr: collect_pipe(stderr, "stderr"),
                    });
                }
                thread::sleep(poll_interval);
            }
            Err(e) => {
                return Err(ProbeError::SubprocessFailed(format!(
                    "probe wait failed: {e}"
                )));
            }
        }
    }
}

/// Build a synthetic Output that `interpret_probe_output` maps to
/// `BackendUnavailable` (no exit code, PROBE_ACK present so it doesn't
/// trigger the "handler not installed" error).
#[cfg(all(test, unix))]
fn build_timeout_output() -> Output {
    use std::os::unix::process::ExitStatusExt;
    Output {
        status: ExitStatus::from_raw(9), // SIGKILL
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"probe timed out".to_vec(),
    }
}

/// Interpret the output of a probe subprocess.
///
/// - Exit [`PROBE_EXIT_ACK_FAILED`] → [`ProbeError::SubprocessFailed`]
///   (checked before the ACK presence test because the ACK write itself failed)
/// - No ACK in stdout → [`ProbeError::HandlerNotInstalled`]
/// - Exit [`PROBE_EXIT_STDERR_FAILED`] with ACK → [`ProbeError::SubprocessFailed`]
///   (only emitted after `emit_ack` succeeds, so ACK presence is required to
///   distinguish from a host binary that lacks the probe handler and happens
///   to exit with code 8)
/// - Exit 0 with ACK → [`ProbeStatus::Available`]
/// - Exit non-zero with ACK → [`ProbeError::ModelLoadFailed`]
/// - Killed by signal with ACK → [`ProbeStatus::BackendUnavailable`]
pub(crate) fn interpret_probe_output(output: &Output) -> Result<ProbeStatus, ProbeError> {
    if output.status.code() == Some(PROBE_EXIT_ACK_FAILED) {
        return Err(ProbeError::SubprocessFailed(
            "probe child failed to emit handshake ACK".into(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.starts_with(PROBE_ACK) {
        return Err(ProbeError::HandlerNotInstalled);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if output.status.signal().is_some() {
            return Ok(ProbeStatus::BackendUnavailable);
        }
    }

    match output.status.code() {
        Some(0) => Ok(ProbeStatus::Available),
        Some(PROBE_EXIT_STDERR_FAILED) => Err(ProbeError::SubprocessFailed(
            "probe child failed to write failure reason to stderr".into(),
        )),
        Some(code) => match SetupReason::try_from(code) {
            Ok(reason) => Err(ProbeError::SetupRejected { reason }),
            Err(()) => {
                let stderr_text = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                Err(ProbeError::ModelLoadFailed {
                    reason: if stderr_text.is_empty() {
                        "model load failed".into()
                    } else {
                        stderr_text
                    },
                })
            }
        },
        None => Ok(ProbeStatus::BackendUnavailable),
    }
}

#[cfg(test)]
mod tests;
