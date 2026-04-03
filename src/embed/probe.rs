use super::{EmbedError, Embedder, ModelPaths, ProbeStatus};

const PROBE_ENV_MODEL: &str = "__RURICO_PROBE_MODEL";
const PROBE_ENV_CONFIG: &str = "__RURICO_PROBE_CONFIG";
const PROBE_ENV_TOKENIZER: &str = "__RURICO_PROBE_TOKENIZER";
pub(crate) const PROBE_ACK: &str = "RURICO_PROBE_OK";

/// Resolve probe env vars into `ModelPaths`. Returns `None` if this is not a
/// probe invocation, `Some(Ok(paths))` if all vars are set, or `Some(Err(3))`
/// if the model var is set but config or tokenizer is missing.
pub(crate) fn probe_env_to_paths(
    model: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
) -> Option<Result<ModelPaths, i32>> {
    let model = model?;
    Some(match (config, tokenizer) {
        (Some(config), Some(tokenizer)) => Ok(ModelPaths {
            model: model.into(),
            config: config.into(),
            tokenizer: tokenizer.into(),
        }),
        _ => Err(3),
    })
}

/// Call at the start of `main()` in binaries that use [`Embedder::probe`].
///
/// When the process was re-invoked as a probe subprocess, this function writes
/// a handshake token to stdout, attempts to load the model, and exits with
/// code 0 (success) or non-zero (failure). Otherwise it returns immediately.
pub fn handle_probe_if_needed() {
    let Some(result) = probe_env_to_paths(
        std::env::var(PROBE_ENV_MODEL).ok(),
        std::env::var(PROBE_ENV_CONFIG).ok(),
        std::env::var(PROBE_ENV_TOKENIZER).ok(),
    ) else {
        return;
    };

    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{PROBE_ACK}");
    let _ = std::io::stdout().flush();

    match result {
        Err(code) => std::process::exit(code),
        Ok(paths) => match Embedder::new(&paths) {
            Ok(_) => std::process::exit(0),
            Err(e) => {
                let _ = write!(std::io::stderr(), "{e}");
                std::process::exit(1);
            }
        },
    }
}

/// Timeout for the probe subprocess (seconds).
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Re-exec the current binary as a probe subprocess.
///
/// Uses `Command` (fork+exec internally) so the child gets a clean process
/// state, avoiding the async-signal-safety issues of bare fork().
///
/// - Exit 0 → [`ProbeStatus::Available`]
/// - Exit non-zero → [`EmbedError::ModelCorrupt`] (reason from stderr)
/// - Killed by signal / timeout / no exit code → [`ProbeStatus::BackendUnavailable`]
pub(super) fn probe_via_subprocess(paths: &ModelPaths) -> Result<ProbeStatus, EmbedError> {
    let exe = std::env::current_exe()
        .map_err(|e| EmbedError::inference(format!("cannot locate executable: {e}")))?;

    let mut child = std::process::Command::new(exe)
        .env(PROBE_ENV_MODEL, &paths.model)
        .env(PROBE_ENV_CONFIG, &paths.config)
        .env(PROBE_ENV_TOKENIZER, &paths.tokenizer)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| EmbedError::inference(format!("probe spawn failed: {e}")))?;

    let output = wait_with_timeout(&mut child, PROBE_TIMEOUT)?;
    interpret_probe_output(&output)
}

/// Wait for a child process with a timeout. Kill and return
/// [`ProbeStatus::BackendUnavailable`] if the deadline is exceeded.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Result<std::process::Output, EmbedError> {
    let deadline = std::time::Instant::now() + timeout;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child.stdout.take().map_or_else(Vec::new, |mut s| {
                    let mut buf = Vec::new();
                    let _ = std::io::Read::read_to_end(&mut s, &mut buf);
                    buf
                });
                let stderr = child.stderr.take().map_or_else(Vec::new, |mut s| {
                    let mut buf = Vec::new();
                    let _ = std::io::Read::read_to_end(&mut s, &mut buf);
                    buf
                });
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    log::warn!(
                        "probe subprocess timed out after {}s, killing",
                        timeout.as_secs()
                    );
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(build_timeout_output());
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                return Err(EmbedError::inference(format!("probe wait failed: {e}")));
            }
        }
    }
}

/// Build a synthetic Output that `interpret_probe_output` maps to
/// `BackendUnavailable` (no exit code, PROBE_ACK present so it doesn't
/// trigger the "handler not installed" error).
fn build_timeout_output() -> std::process::Output {
    use std::os::unix::process::ExitStatusExt;
    std::process::Output {
        status: std::process::ExitStatus::from_raw(9), // SIGKILL
        stdout: format!("{PROBE_ACK}\n").into_bytes(),
        stderr: b"probe timed out".to_vec(),
    }
}

pub(crate) fn interpret_probe_output(
    output: &std::process::Output,
) -> Result<ProbeStatus, EmbedError> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.starts_with(PROBE_ACK) {
        return Err(EmbedError::inference(
            "probe handler not installed; call rurico::embed::handle_probe_if_needed() \
             at the start of main()",
        ));
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
            let reason = String::from_utf8_lossy(&output.stderr).trim().to_string();
            Err(EmbedError::ModelCorrupt {
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
