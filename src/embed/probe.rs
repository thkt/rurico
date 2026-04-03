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

/// Re-exec the current binary as a probe subprocess.
///
/// Uses `Command` (fork+exec internally) so the child gets a clean process
/// state, avoiding the async-signal-safety issues of bare fork().
///
/// - Exit 0 → [`ProbeStatus::Available`]
/// - Exit non-zero → [`EmbedError::ModelCorrupt`] (reason from stderr)
/// - Killed by signal / no exit code → [`ProbeStatus::BackendUnavailable`]
pub(super) fn probe_via_subprocess(paths: &ModelPaths) -> Result<ProbeStatus, EmbedError> {
    let exe = std::env::current_exe()
        .map_err(|e| EmbedError::inference(format!("cannot locate executable: {e}")))?;

    let output = std::process::Command::new(exe)
        .env(PROBE_ENV_MODEL, &paths.model)
        .env(PROBE_ENV_CONFIG, &paths.config)
        .env(PROBE_ENV_TOKENIZER, &paths.tokenizer)
        .output()
        .map_err(|e| EmbedError::inference(format!("probe spawn failed: {e}")))?;

    interpret_probe_output(&output)
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
