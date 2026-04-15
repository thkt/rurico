//! Shared subprocess probe infrastructure for model loading verification.
//!
//! Host binaries call `handle_probe_if_needed` once at the start of `main()`.
//! Individual modules (embed, reranker) call `probe_via_subprocess` to
//! re-exec the current binary as an isolated probe.

use std::env;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{self, Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::embed;
use crate::model_io::ModelPaths;
use crate::reranker;

/// Handshake token written to stdout by probe subprocesses.
pub const PROBE_ACK: &str = "RURICO_PROBE_OK";

/// Exit code used when a probe subprocess was invoked with the primary model env var
/// set but the config or tokenizer env vars were missing.
pub(crate) const PROBE_EXIT_ENV_INCOMPLETE: i32 = 3;

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
        "probe handler not installed; call rurico::model_probe::handle_probe_if_needed() \
         at the start of main()"
    )]
    HandlerNotInstalled,
    /// Model load failed in probe subprocess.
    #[error("model load failed: {reason}")]
    ModelLoadFailed {
        /// Failure detail from stderr.
        reason: String,
    },
    /// Subprocess spawn or wait failure.
    #[error("probe error: {0}")]
    SubprocessFailed(String),
}

// --- Embed probe env var keys ---
pub(crate) const EMBED_PROBE_ENV_MODEL: &str = "__RURICO_PROBE_MODEL";
pub(crate) const EMBED_PROBE_ENV_CONFIG: &str = "__RURICO_PROBE_CONFIG";
pub(crate) const EMBED_PROBE_ENV_TOKENIZER: &str = "__RURICO_PROBE_TOKENIZER";

// --- Reranker probe env var keys ---
pub(crate) const RERANKER_PROBE_ENV_MODEL: &str = "__RURICO_RERANKER_PROBE_MODEL";
pub(crate) const RERANKER_PROBE_ENV_CONFIG: &str = "__RURICO_RERANKER_PROBE_CONFIG";
pub(crate) const RERANKER_PROBE_ENV_TOKENIZER: &str = "__RURICO_RERANKER_PROBE_TOKENIZER";

/// Resolve probe env vars into a `(model, config, tokenizer)` path triple.
///
/// Returns `None` if the primary model env var is absent (not a probe invocation).
/// Returns `Some(Err(PROBE_EXIT_ENV_INCOMPLETE))` if model is set but config or
/// tokenizer is missing.
pub(crate) fn resolve_probe_env(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<(PathBuf, PathBuf, PathBuf), i32>> {
    let model = model?;
    Some(match (config, tokenizer) {
        (Some(config), Some(tokenizer)) => Ok((model.into(), config.into(), tokenizer.into())),
        _ => Err(PROBE_EXIT_ENV_INCOMPLETE),
    })
}

/// Timeout for probe subprocesses.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Single entry point for probe subprocess dispatch in host binaries.
///
/// Call at the start of `main()`. When invoked as a probe subprocess (detected
/// via env vars), this function loads the appropriate model and exits. Otherwise
/// returns immediately.
///
/// # Process Behavior
///
/// When the current process is a probe subprocess, this function terminates
/// with `std::process::exit`:
///
/// - Exit 0: model loaded successfully
/// - Exit 1: model load failed (reason written to stderr)
/// - Exit 3: primary env var set but config or tokenizer env var missing
///
/// When the current process is not a probe subprocess, it returns immediately.
pub fn handle_probe_if_needed() {
    // --- Embed probe ---
    if let Some(result) = embed::probe_env_to_paths(
        env::var(EMBED_PROBE_ENV_MODEL).ok(),
        env::var(EMBED_PROBE_ENV_CONFIG).ok(),
        env::var(EMBED_PROBE_ENV_TOKENIZER).ok(),
    ) {
        dispatch_probe(result, |candidate| {
            let artifacts = candidate.verify().map_err(|e| e.to_string())?;
            embed::Embedder::new(&artifacts)
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
    }

    // --- Reranker probe ---
    if let Some(result) = reranker::probe_env_to_paths(
        env::var(RERANKER_PROBE_ENV_MODEL).ok(),
        env::var(RERANKER_PROBE_ENV_CONFIG).ok(),
        env::var(RERANKER_PROBE_ENV_TOKENIZER).ok(),
    ) {
        dispatch_probe(result, |candidate| {
            let artifacts = candidate.verify().map_err(|e| e.to_string())?;
            reranker::Reranker::new(&artifacts)
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
    }
}

/// Exit action computed by [`compute_probe_exit`].
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) struct ProbeExitAction {
    pub(crate) code: i32,
    pub(crate) message: Option<String>,
}

/// Determine the exit code and stderr message for a probe dispatch.
///
/// - `Err(code)`: env vars incomplete → exit with that code
/// - `Ok(Ok(()))`: model loaded successfully → exit 0
/// - `Ok(Err(reason))`: model load failed → exit 1 with reason
pub(crate) fn compute_probe_exit(result: Result<Result<(), String>, i32>) -> ProbeExitAction {
    match result {
        Err(code) => ProbeExitAction {
            code,
            message: Some("probe env incomplete: config or tokenizer env var missing".into()),
        },
        Ok(Ok(())) => ProbeExitAction {
            code: 0,
            message: None,
        },
        Ok(Err(reason)) => ProbeExitAction {
            code: 1,
            message: Some(reason),
        },
    }
}

/// Execute a probe dispatch: emit ACK, load model, exit with appropriate code.
///
/// Always terminates the process via [`std::process::exit`].
fn dispatch_probe<P, E: fmt::Display>(
    result: Result<P, i32>,
    load: impl FnOnce(P) -> Result<(), E>,
) -> ! {
    emit_ack();
    let outcome = match result {
        Err(code) => Err(code),
        Ok(candidate) => Ok(load(candidate).map_err(|e| e.to_string())),
    };
    let action = compute_probe_exit(outcome);
    if let Some(ref msg) = action.message {
        let _ = write!(io::stderr(), "{msg}");
    }
    process::exit(action.code);
}

fn emit_ack() {
    let _ = writeln!(io::stdout(), "{PROBE_ACK}");
    let _ = io::stdout().flush();
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

    let mut cmd = Command::new(exe);
    for &(key, value) in env_pairs {
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

fn drain_pipe(pipe: Option<impl Read>, label: &str) -> Vec<u8> {
    pipe.map_or_else(Vec::new, |mut s| {
        let mut buf = Vec::new();
        if let Err(e) = s.read_to_end(&mut buf) {
            log::warn!("probe: failed to drain child {label}: {e}");
        }
        buf
    })
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
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = drain_pipe(child.stdout.take(), "stdout");
                let stderr = drain_pipe(child.stderr.take(), "stderr");
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    log::warn!(
                        "probe subprocess timed out after {}s, killing",
                        timeout.as_secs()
                    );
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(build_timeout_output());
                }
                thread::sleep(Duration::from_millis(100));
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
#[cfg(unix)]
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
/// - Exit 0 with ACK → [`ProbeStatus::Available`]
/// - Exit non-zero with ACK → [`ProbeError::ModelLoadFailed`]
/// - Killed by signal with ACK → [`ProbeStatus::BackendUnavailable`]
/// - No ACK in stdout → [`ProbeError::HandlerNotInstalled`]
pub(crate) fn interpret_probe_output(output: &Output) -> Result<ProbeStatus, ProbeError> {
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
        Some(_) => {
            let reason = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            Err(ProbeError::ModelLoadFailed {
                reason: if reason.is_empty() {
                    "model load failed".into()
                } else {
                    reason
                },
            })
        }
        None => Ok(ProbeStatus::BackendUnavailable),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let output = build_timeout_output();
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
}
