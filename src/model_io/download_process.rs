//! Download worker wire protocol and parent-side lifetime management.

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::process::{self, Command};
use std::time::Duration;

use super::{DOWNLOAD_TIMEOUT, ModelArtifact, ModelIoError, ModelPaths, hf_backend};
use crate::{dispatch, owned_process};

pub(crate) const REQUEST_ENV: &str = "__RURICO_DOWNLOAD_REQUEST";
const ACK: &str = "RURICO_DOWNLOAD_OK\n";
const POLL: Duration = Duration::from_millis(10);

pub(super) fn download<Id: ModelArtifact>(model: Id) -> Result<ModelPaths, ModelIoError> {
    ensure_installed(dispatch::installed(), env::var_os(REQUEST_ENV).is_some())?;
    let exe = env::current_exe().map_err(download_error)?;
    let request =
        serde_json::to_string(&(model.repo_id(), model.revision())).map_err(download_error)?;
    let mut cmd = Command::new(exe);
    // HF/cache/auth/proxy settings retain the caller's environment, as they
    // did for the former in-process download. REQUEST_ENV selects dispatch
    // before probe handling, so inherited probe variables cannot divert it.
    cmd.env(REQUEST_ENV, request);
    download_with_command(&mut cmd, DOWNLOAD_TIMEOUT)
}

fn ensure_installed(installed: bool, in_worker: bool) -> Result<(), ModelIoError> {
    if !installed || in_worker {
        return Err(ModelIoError::Download(
            "download dispatcher not installed: call rurico::handle_probe_if_needed() at the start of main() before download".into(),
        ));
    }
    Ok(())
}

fn download_error(error: impl fmt::Display) -> ModelIoError {
    ModelIoError::Download(error.to_string())
}

fn download_with_command(cmd: &mut Command, timeout: Duration) -> Result<ModelPaths, ModelIoError> {
    let mut child = owned_process::spawn(cmd).map_err(download_error)?;
    let collected = owned_process::collect(&mut child, timeout, POLL, ACK.as_bytes())
        .map_err(download_error)?;
    if collected.timed_out {
        return Err(ModelIoError::Download(format!(
            "download exceeded {} second timeout; worker group stopped",
            timeout.as_secs_f64()
        )));
    }
    if !collected.ack {
        return Err(ModelIoError::Download("download dispatcher not installed in child: call rurico::handle_probe_if_needed() at the start of main()".into()));
    }
    if !collected.output.status.success() {
        return Err(ModelIoError::Download(format!(
            "download worker {}: {}",
            collected.output.status,
            String::from_utf8_lossy(&collected.output.stderr).trim()
        )));
    }
    if collected.stdout_truncated {
        return Err(ModelIoError::Download(
            "download worker result exceeded 256 KiB output limit".into(),
        ));
    }
    let begin = owned_process::ack_payload_start(&collected.output.stdout, ACK.as_bytes())
        .ok_or_else(|| download_error("invalid download worker result prefix"))?;
    let payload = &collected.output.stdout[begin..];
    let [model, config, tokenizer] =
        serde_json::from_slice::<[Vec<u8>; 3]>(payload).map_err(download_error)?;
    let path = |bytes| PathBuf::from(OsString::from_vec(bytes));
    Ok(ModelPaths {
        model: path(model),
        config: path(config),
        tokenizer: path(tokenizer),
    })
}

pub(crate) fn dispatch_download(request: &str) -> ! {
    dispatch_download_with(request, hf_backend::download_in_worker)
}

fn dispatch_download_with(
    request: &str,
    download: impl FnOnce(&str, &str) -> Result<ModelPaths, ModelIoError>,
) -> ! {
    let mut stdout = io::stdout().lock();
    if stdout
        .write_all(ACK.as_bytes())
        .and_then(|()| stdout.flush())
        .is_err()
    {
        process::exit(1);
    }
    let result = serde_json::from_str::<(String, String)>(request)
        .map_err(download_error)
        .and_then(|(repo, revision)| download(&repo, &revision));
    let code = match result {
        Ok(paths) => {
            // Preserve Unix path bytes, including a non-UTF-8 HF cache root.
            let paths = [
                paths.model.as_os_str().as_bytes(),
                paths.config.as_os_str().as_bytes(),
                paths.tokenizer.as_os_str().as_bytes(),
            ];
            if serde_json::to_writer(&mut stdout, &paths).is_ok() && stdout.flush().is_ok() {
                0
            } else {
                1
            }
        }
        Err(error) => {
            let _ = writeln!(io::stderr(), "{error}");
            let _ = io::stderr().flush();
            1
        }
    };
    process::exit(code);
}

#[cfg(test)]
mod tests;
