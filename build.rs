// Preserve Cargo build conditions for offline smoke revision comparisons.
// Library-only consumers need neither the provenance commands nor this record.
use std::env;
use std::fmt::Write;
use std::fs;
use std::io::{ErrorKind, Write as IoWrite};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn append(signature: &mut String, key: &str, bytes: &[u8]) {
    signature.push_str(key);
    signature.push('=');
    for byte in bytes {
        write!(signature, "{byte:02x}").expect("write to String");
    }
    signature.push(';');
}

fn content_hash(bytes: &[u8]) -> Vec<u8> {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("build provenance shasum");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(bytes)
        .expect("hash build configuration");
    let out = child.wait_with_output().expect("build configuration hash");
    assert!(out.status.success(), "build configuration hash failed");
    out.stdout
        .split(u8::is_ascii_whitespace)
        .next()
        .unwrap()
        .to_vec()
}

fn main() {
    if env::var_os("CARGO_FEATURE_SMOKE").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=RURICO_SMOKE_BUILD_INVOCATION");
    if env::var("RURICO_SMOKE_BUILD_INVOCATION").as_deref() == Ok("locked-release-smoke-v1") {
        println!("cargo:rustc-env=RURICO_BUILD_INVOCATION=locked-release-smoke-v1");
    }
    let mut signature = String::new();
    for key in [
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "TARGET",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        append(
            &mut signature,
            key,
            env::var(key).expect("Cargo build condition").as_bytes(),
        );
    }
    let mut conditions: Vec<_> = env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_PROFILE_")
                || key.starts_with("CARGO_FEATURE_")
                || key.starts_with("CARGO_CFG_")
        })
        .collect();
    conditions.sort_unstable();
    for (key, value) in conditions {
        println!("cargo:rerun-if-env-changed={key}");
        append(&mut signature, &key, value.as_bytes());
    }
    let rustc = Command::new(env::var_os("RUSTC").expect("Cargo compiler"))
        .args(["-Vv"])
        .output()
        .expect("build compiler version");
    assert!(rustc.status.success(), "build compiler version failed");
    append(&mut signature, "BUILD_RUSTC", &rustc.stdout);
    println!("cargo:rerun-if-changed=Cargo.toml");
    append(
        &mut signature,
        "MANIFEST_SHA256",
        &content_hash(&fs::read("Cargo.toml").expect("Cargo manifest")),
    );

    // Record configuration contents, never local paths or credentials. Cargo
    // searches the checkout's ancestors and CARGO_HOME for these config files.
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let mut directories: Vec<_> = root.ancestors().map(|p| p.join(".cargo")).collect();
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("HOME").expect("home directory")).join(".cargo")
        });
    directories.push(cargo_home);
    let mut config_index = 0;
    for directory in &directories {
        for name in ["config", "config.toml"] {
            let path = directory.join(name);
            println!("cargo:rerun-if-changed={}", path.display());
            let hash = match fs::read(&path) {
                Ok(bytes) => content_hash(&bytes),
                Err(e) if e.kind() == ErrorKind::NotFound => continue,
                Err(e) => panic!("read build configuration: {e}"),
            };
            append(&mut signature, &format!("CONFIG_{config_index}"), &hash);
            config_index += 1;
        }
    }
    println!("cargo:rustc-env=RURICO_BUILD_FLAGS={signature}");
}
