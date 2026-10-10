use super::*;
use crate::embed::ModelId;
use crate::owned_process::tests::{reexec_for_thread_observation, thread_count};
use std::fs::{self, OpenOptions};
use std::path::Path;
use std::thread;
use std::time::Instant;

const TEST_MODE: &str = "RURICO_FAKE_DOWNLOAD_MODE";
const TEST_DIR: &str = "RURICO_FAKE_DOWNLOAD_DIR";

fn worker(mode: &str, dir: &Path) -> Command {
    let mut cmd = Command::new(env::current_exe().unwrap());
    cmd.args([
        "--exact",
        "model_io::download_process::tests::fake_worker_entry",
        "--nocapture",
    ])
    .env(TEST_MODE, mode)
    .env(TEST_DIR, dir)
    .env(REQUEST_ENV, r#"["fixture/model","fixed-revision"]"#);
    cmd
}

// Re-exec the test binary, but run the production dispatch/result and collection
// paths. The fake closure replaces only HF network I/O inside the child.
#[test]
fn fake_worker_entry() {
    let Ok(mode) = env::var(TEST_MODE) else {
        return;
    };
    if mode == "missing" {
        let error = download(ModelId::DEFAULT).unwrap_err();
        assert!(error.to_string().contains("dispatcher not installed"));
        return;
    }
    if mode == "dispatch" {
        crate::handle_probe_if_needed();
        panic!("download dispatcher must exit");
    }
    if matches!(mode.as_str(), "slow" | "hang" | "cache-race") {
        // A broken cleanup assertion must not leave the test fixture alive
        // indefinitely. This watchdog is absent from the production worker.
        thread::spawn(|| {
            thread::sleep(Duration::from_secs(10));
            process::exit(99);
        });
    }
    if mode == "startup" {
        println!("startup marker: RURICO_DOWNLOAD_OK");
    }
    if mode == "overflow" {
        io::stdout()
            .write_all(&vec![b'x'; owned_process::OUTPUT_LIMIT + 8192])
            .unwrap();
        println!();
    }
    let dir = PathBuf::from(env::var_os(TEST_DIR).unwrap());
    dispatch_download_with(&env::var(REQUEST_ENV).unwrap(), |repo, revision| {
        assert_eq!(repo, "fixture/model");
        assert_eq!(revision, "fixed-revision");
        if mode == "slow" || mode == "hang" {
            fs::write(dir.join("worker.pid"), process::id().to_string()).unwrap();
        }
        match mode.as_str() {
            "success" | "startup" | "overflow" => Ok(ModelPaths::from_dir(&dir)),
            "error" => {
                fs::write(dir.join("blob.incomplete"), b"partial").unwrap();
                Err(download_error("simulated mid-download I/O failure"))
            }
            "slow" => {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("blob.incomplete"))
                    .unwrap();
                loop {
                    file.write_all(b"chunk").unwrap();
                    file.flush().unwrap();
                    thread::sleep(Duration::from_millis(2));
                }
            }
            #[cfg(feature = "test-support")]
            "cache-race" => {
                assert!(
                    !dir.join("snapshot").exists(),
                    "worker must observe cache miss first"
                );
                fs::write(dir.join("cache-miss"), process::id().to_string()).unwrap();
                while !dir.join("publish-go").exists() {
                    thread::sleep(Duration::from_millis(1));
                }
                hf_hub::rurico_test_publish_pointer_with(
                    Path::new("normal.blob"),
                    &dir.join("snapshot"),
                    || {
                        fs::write(dir.join("publication-boundary"), b"ready").unwrap();
                        loop {
                            thread::park();
                        }
                    },
                )
                .unwrap();
                unreachable!()
            }
            "hang" => loop {
                thread::park();
            },
            _ => panic!("unknown fake mode"),
        }
    });
}

#[test]
fn download_worker_routes_paths_and_io_failure_without_cache_deletion() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("normal.blob"), b"cached").unwrap();
    fs::write(dir.path().join("other.incomplete"), b"other consumer").unwrap();
    let cache = dir.path().join(OsString::from_vec(vec![b'c', 0xff]));
    let paths =
        download_with_command(&mut worker("success", &cache), Duration::from_secs(5)).unwrap();
    assert_eq!(paths.model, cache.join("model.safetensors"));
    assert_eq!(paths.config, cache.join("config.json"));
    assert_eq!(paths.tokenizer, cache.join("tokenizer.json"));
    let startup_paths =
        download_with_command(&mut worker("startup", &cache), Duration::from_secs(5)).unwrap();
    assert_eq!(startup_paths.tokenizer, paths.tokenizer);
    let error =
        download_with_command(&mut worker("overflow", &cache), Duration::from_secs(5)).unwrap_err();
    assert!(
        error.to_string().contains("exceeded 256 KiB output limit"),
        "{error}"
    );
    let error = download_with_command(&mut worker("error", dir.path()), Duration::from_secs(5))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("simulated mid-download I/O failure"),
        "{error}"
    );
    assert_eq!(
        fs::read(dir.path().join("blob.incomplete")).unwrap(),
        b"partial"
    );
    assert_eq!(fs::read(dir.path().join("normal.blob")).unwrap(), b"cached");
    assert_eq!(
        fs::read(dir.path().join("other.incomplete")).unwrap(),
        b"other consumer"
    );
}

#[test]
fn download_dispatch_contract_rejects_missing_registration_and_bad_request() {
    assert!(
        ensure_installed(false, false)
            .unwrap_err()
            .to_string()
            .contains("dispatcher not installed")
    );
    assert!(ensure_installed(true, true).is_err());
    let dir = tempfile::tempdir().unwrap();
    let error = download_with_command(&mut worker("missing", dir.path()), Duration::from_secs(5))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("dispatcher not installed in child"),
        "{error}"
    );
    let mut cmd = worker("dispatch", dir.path());
    cmd.env(REQUEST_ENV, "invalid request");
    let error = download_with_command(&mut cmd, Duration::from_secs(5)).unwrap_err();
    assert!(error.to_string().contains("expected value"), "{error}");
}

#[test]
fn repeated_download_timeouts_stop_writes_and_reap_workers() {
    if reexec_for_thread_observation(
        "model_io::download_process::tests::repeated_download_timeouts_stop_writes_and_reap_workers",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("normal.blob"), b"cached").unwrap();
    fs::write(dir.path().join("other.incomplete"), b"other consumer").unwrap();
    let before = thread_count();
    for mode in ["slow", "hang", "slow", "hang", "slow"] {
        // Each attempt owns fresh observations; an old PID/file cannot pass.
        let attempt = tempfile::tempdir_in(dir.path()).unwrap();
        let mut cmd = worker(mode, attempt.path());
        let start = Instant::now();
        let error = download_with_command(&mut cmd, Duration::from_millis(300)).unwrap_err();
        assert!(
            matches!(error, ModelIoError::Download(ref message) if message.contains("timeout")),
            "{error}"
        );
        assert!(start.elapsed() < Duration::from_secs(3));
        let pid: u32 = fs::read_to_string(attempt.path().join("worker.pid"))
            .expect("fixture failed to enter download worker")
            .parse()
            .unwrap();
        assert!(!rurico_ffi::process_group_exists(pid).unwrap());
        let path = attempt.path().join("blob.incomplete");
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if mode == "slow" {
            assert!(size > 0, "this attempt never wrote before timeout");
        }
        thread::sleep(Duration::from_millis(30));
        assert_eq!(
            fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            size,
            "writes survived timeout"
        );
    }
    assert_eq!(fs::read(dir.path().join("normal.blob")).unwrap(), b"cached");
    assert_eq!(
        fs::read(dir.path().join("other.incomplete")).unwrap(),
        b"other consumer"
    );
    assert_eq!(thread_count(), before, "parent threads accumulated");
}

// Interrupt the actual HF publication primitive after another consumer has
// filled the same immutable snapshot. The barrier is at the former unlink /
// symlink window; the production collector performs the timeout and recovery.
#[cfg(feature = "test-support")]
#[test]
fn timeout_at_hf_publication_preserves_concurrently_published_snapshot() {
    use std::os::unix::fs::{MetadataExt, symlink};
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("normal.blob"), b"cached").unwrap();
    fs::write(dir.path().join("other.incomplete"), b"other consumer").unwrap();
    let pointer = dir.path().join("snapshot");
    let mut child = owned_process::spawn(&mut worker("cache-race", dir.path())).unwrap();
    let wait_for = |name: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !dir.path().join(name).exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        dir.path().join(name).exists()
    };
    let entered = wait_for("cache-miss");
    symlink("normal.blob", &pointer).unwrap(); // Independent consumer publication.
    let inode = fs::symlink_metadata(&pointer).unwrap().ino();
    fs::write(dir.path().join("publish-go"), b"go").unwrap();
    let at_boundary = wait_for("publication-boundary");
    let result = owned_process::collect(&mut child, Duration::ZERO, POLL, ACK.as_bytes()).unwrap();
    assert!(
        entered && at_boundary,
        "fixture did not reach this publication attempt"
    );
    assert!(result.timed_out && result.ack);
    assert!(!rurico_ffi::process_group_exists(child.id()).unwrap());
    assert_eq!(fs::read(&pointer).unwrap(), b"cached");
    assert_eq!(fs::symlink_metadata(&pointer).unwrap().ino(), inode);
    // Successful finalize must also reuse the existing link, never replace it.
    hf_hub::rurico_test_publish_pointer_with(Path::new("normal.blob"), &pointer, || {}).unwrap();
    assert_eq!(fs::symlink_metadata(&pointer).unwrap().ino(), inode);
    assert_eq!(
        fs::read(dir.path().join("other.incomplete")).unwrap(),
        b"other consumer"
    );
    let fresh = dir.path().join("fresh-snapshot");
    hf_hub::rurico_test_publish_pointer_with(Path::new("normal.blob"), &fresh, || {}).unwrap();
    assert_eq!(fs::read(&fresh).unwrap(), b"cached");
    let broken = dir.path().join("broken-snapshot");
    symlink("missing.blob", &broken).unwrap();
    assert!(
        hf_hub::rurico_test_publish_pointer_with(Path::new("normal.blob"), &broken, || {}).is_err()
    );
    assert_eq!(fs::read_link(&broken).unwrap(), Path::new("missing.blob"));
}
