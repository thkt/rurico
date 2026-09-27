//! Versioned raw records. Only this executable collects experiment provenance.
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::process::{Command, Stdio};
use std::str;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rurico::embed::{
    ChunkedEmbedding, Embed, EmbedOptions, InferenceMetrics, ModelId, fixtures,
    workloads::{workload_w1, workload_w2, workload_w3},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Batch,
    Sequential,
}

pub fn order(repeat: usize) -> [Method; 2] {
    if repeat.is_multiple_of(2) {
        [Method::Batch, Method::Sequential]
    } else {
        [Method::Sequential, Method::Batch]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub schema: u32,
    pub context: Value,
    pub sequence: usize,
    pub workload: String,
    pub input_sha256: Option<String>,
    pub options: EmbedOptions,
    pub method: Method,
    pub measured: bool,
    pub state: String,
    pub repeat: Option<usize>,
    /// External timer around the whole call sequence; excludes JSON and parity checks.
    pub wall: Duration,
    /// One entry per API invocation. Sequential uses singleton *options* calls.
    pub calls: Vec<Option<InferenceMetrics>>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub n: usize,
    pub min: Duration,
    pub median: Duration,
    pub max: Duration,
}

pub fn distribution(mut values: Vec<Duration>) -> Distribution {
    assert!(!values.is_empty());
    values.sort_unstable();
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[middle - 1] + (values[middle] - values[middle - 1]) / 2
    } else {
        values[middle]
    };
    Distribution {
        n: values.len(),
        min: values[0],
        median,
        max: values[values.len() - 1],
    }
}

fn command(program: &str, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status.success().then_some(out.stdout)
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    String::from_utf8(command(program, args)?)
        .ok()
        .map(|s| s.trim().to_owned())
}

fn sha256(bytes: &[u8]) -> Option<String> {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let written = child.stdin.take()?.write_all(bytes);
    let out = child.wait_with_output().ok()?;
    written.ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()?
        .split_whitespace()
        .next()
        .map(str::to_owned)
}

// Hash names and contents of non-ignored untracked files without publishing paths.
fn untracked_hash() -> Option<String> {
    let names = command("git", &["ls-files", "--others", "--exclude-standard", "-z"])?;
    let mut bytes = Vec::new();
    for name in names.split(|b| *b == 0).filter(|n| !n.is_empty()) {
        let name_str = str::from_utf8(name).ok()?;
        let data = fs::read(name_str).ok()?;
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&data);
    }
    sha256(&bytes)
}

// Metal's verbose version includes InstalledDir, an ephemeral local path.
// Save only version/target lines; unknown formats stay explicitly unavailable.
fn metal_version(output: &str) -> Option<String> {
    let lines: Vec<_> = output
        .lines()
        .filter(|line| line.starts_with("Apple metal version ") || line.starts_with("Target: "))
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

pub fn context() -> Value {
    let model = ModelId::DEFAULT;
    json!({
        "started_unix": SystemTime::now().duration_since(UNIX_EPOCH).ok(),
        "commit": command_text("git", &["rev-parse", "HEAD"]),
        "tracked_diff_sha256": command("git", &["diff", "--binary", "HEAD", "--"]).and_then(|b| sha256(&b)),
        "untracked_sha256": untracked_hash(),
        "dirty": command("git", &["status", "--porcelain", "-z"]).map(|b| !b.is_empty()),
        "lockfile_sha256": fs::read("Cargo.lock").ok().and_then(|b| sha256(&b)),
        "executable_sha256": env::current_exe().ok().and_then(|p| fs::read(p).ok()).and_then(|b| sha256(&b)),
        "model": model.repo_id(), "model_revision": model.revision(),
        "tokenizer_revision": model.revision(), "revision_source": "pinned cached_artifacts lookup; local content modification not checked",
        "machine": command_text("sysctl", &["-n", "hw.model"]),
        "chip": command_text("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "os": command_text("sw_vers", &[]),
        "rustc": command_text("rustc", &["-Vv"]),
        "cargo": command_text("cargo", &["-V"]),
        "xcode": command_text("xcodebuild", &["-version"]),
        "metal": command_text("xcrun", &["metal", "--version"]).and_then(|s| metal_version(&s)),
        "debug_assertions": cfg!(debug_assertions),
        "build_flags": null,
        "toolchain_scope": "commands available at measurement time; rebuild before measuring",
        "os_cache_cleared": false,
        "inference_cache_policy": "buffer and current-thread compile cache cleanup after every forward, including warm-up",
        "rss": null, "metal_memory": null,
        "unavailable": "null means unavailable or not collected; no estimates"
    })
}

fn emit(value: &impl Serialize) {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value).expect("write JSON record");
    writeln!(stdout).expect("write record newline");
    stdout.flush().expect("flush record");
}

pub fn emit_load(context: &Value, elapsed: Duration) {
    emit(
        &json!({"schema": 1, "event": "model_load", "context": context, "wall": elapsed,
        "scope": "Embedder::new only; cache lookup/tokenizer verification excluded; OS cache untouched"}),
    );
}

fn verify_metrics(m: &InferenceMetrics, options: &EmbedOptions, chunks: usize) {
    assert_eq!(
        m.pause_count,
        if options.forward_pause.is_some() {
            m.forwards.len()
        } else {
            0
        }
    );
    assert_eq!(m.tokenize, None);
    assert_eq!(m.num_chunks, chunks);
    assert_eq!(
        m.forwards.iter().map(|s| s.batch_size).sum::<usize>(),
        chunks
    );
    assert_eq!(
        m.forwards
            .iter()
            .map(|s| s.batch_size * s.sequence_length)
            .sum::<usize>(),
        m.padded_tokens
    );
    for shape in &m.forwards {
        assert!([128, 512, 2048, 8192].contains(&shape.sequence_length));
        if let Some(budget) = options.token_budget {
            assert!(shape.batch_size <= (budget / shape.sequence_length).max(1));
        }
    }
    if options.forward_pause.is_none() {
        assert_eq!(m.pause, Duration::ZERO);
    }
    // sleep guarantees at least the requested duration. Check only that floor,
    // never equality or an upper bound affected by host scheduling jitter.
    if let Some(requested) = options.forward_pause {
        let count = u32::try_from(m.pause_count).expect("pause count fits u32");
        assert!(m.pause >= requested * count);
    }
}

fn invoke(
    provider: &dyn Embed,
    texts: &[&str],
    options: &EmbedOptions,
    measured: bool,
) -> (Vec<ChunkedEmbedding>, Option<InferenceMetrics>) {
    if measured {
        let output = provider
            .embed_documents_batch_with_options_and_metrics(texts, options)
            .expect("measured inference");
        (output.embeddings, output.metrics)
    } else {
        (
            provider
                .embed_documents_batch_with_options(texts, options)
                .expect("ordinary inference"),
            None,
        )
    }
}

struct Trial<'a> {
    provider: &'a dyn Embed,
    context: &'a Value,
    workload: &'a str,
    texts: &'a [&'a str],
    input_sha256: Option<String>,
    options: EmbedOptions,
    expected: &'a [ChunkedEmbedding],
}

impl Trial<'_> {
    fn run(
        &self,
        sequence: usize,
        method: Method,
        measured: bool,
        state: &str,
        repeat: Option<usize>,
    ) -> Record {
        let mut output = Vec::new();
        let mut calls = Vec::new();
        let start = Instant::now();
        match method {
            Method::Batch => {
                let (docs, metrics) = invoke(self.provider, self.texts, &self.options, measured);
                output = docs;
                calls.push(metrics);
            }
            Method::Sequential => {
                for text in self.texts {
                    let (docs, metrics) = invoke(self.provider, &[*text], &self.options, measured);
                    output.extend(docs);
                    calls.push(metrics);
                }
            }
        }
        let wall = start.elapsed();
        // Outside timed region, reuse existing parity tolerances and fixtures.
        let diff =
            fixtures::compare(self.expected, &output).expect("input order/chunk shape parity");
        assert!(diff.cosine_min >= fixtures::DEFAULT_COSINE_MIN);
        assert!(diff.max_abs_diff <= fixtures::DEFAULT_MAX_ABS_DIFF);
        if measured {
            match method {
                Method::Batch => verify_metrics(
                    calls[0].as_ref().expect("MLX metrics"),
                    &self.options,
                    output.iter().map(|d| d.chunks().len()).sum(),
                ),
                Method::Sequential => {
                    for (m, d) in calls.iter().zip(&output) {
                        verify_metrics(
                            m.as_ref().expect("MLX metrics"),
                            &self.options,
                            d.chunks().len(),
                        );
                    }
                }
            }
        }
        Record {
            schema: 1,
            context: self.context.clone(),
            sequence,
            workload: self.workload.to_owned(),
            input_sha256: self.input_sha256.clone(),
            options: self.options,
            method,
            measured,
            state: state.to_owned(),
            repeat,
            wall,
            calls,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Baseline,
    Records,
    /// All API/options comparisons on the first three existing W2 documents.
    Overhead,
}

/// Run one process: model load is emitted by main; first inference and warm-up
/// records are retained but excluded from warm summary groups.
pub fn run(provider: &dyn Embed, context: &Value, mode: Mode) -> Vec<Record> {
    let include_options_and_overhead = mode != Mode::Baseline;
    let mut records = Vec::new();
    let mut profiles = vec![EmbedOptions::default()];
    if include_options_and_overhead {
        profiles.push(EmbedOptions {
            token_budget: Some(256),
            forward_pause: Some(Duration::from_millis(1)),
        });
    }
    for options in profiles {
        for (name, generate) in [
            ("w1", workload_w1 as fn() -> Vec<String>),
            ("w2", workload_w2),
            ("w3", workload_w3),
        ] {
            if mode == Mode::Overhead && name != "w2" {
                continue;
            }
            let mut texts = generate();
            if mode == Mode::Overhead {
                texts.truncate(3);
            }
            let refs: Vec<_> = texts.iter().map(String::as_str).collect();
            let mut expected = fixtures::load(
                &mut File::open(format!("tests/fixtures/phase2_baseline/{name}.bin"))
                    .expect("fixture file"),
            )
            .expect("fixture read");
            if mode == Mode::Overhead {
                expected.truncate(3);
            }
            let trial = Trial {
                provider,
                context,
                workload: if mode == Mode::Overhead {
                    "w2_first3"
                } else {
                    name
                },
                texts: &refs,
                input_sha256: sha256(&serde_json::to_vec(&texts).expect("encode input for hash")),
                options,
                expected: &expected,
            };
            let variants: &[bool] = if include_options_and_overhead {
                &[true, false]
            } else {
                &[true]
            };
            for &measured in variants {
                for method in order(0) {
                    let state = if records.is_empty() {
                        "first_inference"
                    } else {
                        "warmup"
                    };
                    let record = trial.run(records.len(), method, measured, state, None);
                    emit(&record);
                    records.push(record);
                }
            }
            for repeat in 0_usize..3 {
                // Alternate both method and instrumentation ordering across repeats.
                let variants: Vec<_> = if repeat.is_multiple_of(2) {
                    variants.to_vec()
                } else {
                    variants.iter().rev().copied().collect()
                };
                for measured in variants {
                    for method in order(repeat) {
                        let record =
                            trial.run(records.len(), method, measured, "warm", Some(repeat));
                        if name == "w2"
                            && options.token_budget == Some(256)
                            && measured
                            && method == Method::Batch
                        {
                            assert!(record.calls[0].as_ref().unwrap().forwards.len() > 1);
                        }
                        emit(&record);
                        records.push(record);
                    }
                }
            }
        }
    }
    let start = Instant::now();
    let empty = provider
        .embed_documents_batch_with_options_and_metrics(&[], &EmbedOptions::default())
        .expect("empty measured batch");
    let wall = start.elapsed();
    assert!(empty.embeddings.is_empty());
    verify_metrics(
        empty.metrics.as_ref().expect("empty MLX metrics"),
        &EmbedOptions::default(),
        0,
    );
    let empty_record = Record {
        schema: 1,
        context: context.clone(),
        sequence: records.len(),
        workload: "empty".to_owned(),
        input_sha256: sha256(b"[]"),
        options: EmbedOptions::default(),
        method: Method::Batch,
        measured: true,
        state: "validation".to_owned(),
        repeat: None,
        wall,
        calls: vec![empty.metrics],
    };
    emit(&empty_record);
    records.push(empty_record);
    emit_summary(&records);
    records
}

pub fn emit_summary(records: &[Record]) {
    for summary in summaries(records) {
        emit(&summary);
    }
}

fn summaries(records: &[Record]) -> Vec<Value> {
    let mut groups: Vec<Vec<&Record>> = Vec::new();
    for r in records.iter().filter(|r| r.state == "warm") {
        if let Some(group) = groups.iter_mut().find(|g| {
            let first = g[0];
            first.context == r.context
                && first.workload == r.workload
                && first.input_sha256 == r.input_sha256
                && first.options == r.options
                && first.method == r.method
                && first.measured == r.measured
        }) {
            group.push(r);
        } else {
            groups.push(vec![r]);
        }
    }
    groups
        .into_iter()
        .map(|group| {
            let r = group[0];
            json!({"schema": 1, "event": "summary", "context": r.context,
            "workload": r.workload, "input_sha256": r.input_sha256, "options": r.options,
            "method": r.method, "measured": r.measured,
            "sequences": group.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            "wall": distribution(group.iter().map(|r| r.wall).collect())})
        })
        .collect()
}

pub fn summarize_file(path: &str) {
    let input = io::BufReader::new(File::open(path).expect("open JSONL"));
    let mut records = Vec::new();
    for line in input.lines() {
        let value: Value = serde_json::from_str(&line.expect("read JSONL")).expect("parse JSONL");
        assert_eq!(value["schema"], 1, "unsupported record schema");
        if value.get("event").is_none() {
            records.push(serde_json::from_value::<Record>(value).expect("inference record"));
        }
    }
    assert!(!records.is_empty(), "no inference records");
    emit_summary(&records);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(sequence: usize, ns: u64) -> Record {
        Record {
            schema: 1,
            context: json!({"synthetic": true}),
            sequence,
            workload: "synthetic".to_owned(),
            input_sha256: None,
            options: EmbedOptions::default(),
            method: Method::Batch,
            measured: true,
            state: "warm".to_owned(),
            repeat: Some(sequence),
            wall: Duration::from_nanos(ns),
            calls: vec![Some(InferenceMetrics {
                forward_eval: Duration::from_nanos(ns / 2),
                tokenize: None,
                ..InferenceMetrics::default()
            })],
        }
    }

    #[test]
    fn records_roundtrip_and_summary_preserve_raw_trials_and_conditions() {
        let mut records = vec![record(0, 900), record(1, 100), record(2, 500)];
        let mut excluded = record(3, 100_000);
        excluded.state = "warmup".to_owned();
        records.push(excluded);
        let mut other = record(4, 700);
        other.options.token_budget = Some(256);
        records.push(other);
        let mut ordinary = record(5, 800);
        ordinary.measured = false;
        ordinary.calls = vec![None];
        records.push(ordinary);
        let mut sequential = record(6, 600);
        sequential.method = Method::Sequential;
        records.push(sequential);
        let encoded = serde_json::to_string(&records).unwrap();
        let read: Vec<Record> = serde_json::from_str(&encoded).unwrap();
        let summaries = summaries(&read);
        assert_eq!(summaries.len(), 4);
        let stats: Distribution = serde_json::from_value(summaries[0]["wall"].clone()).unwrap();
        assert_eq!(
            stats,
            Distribution {
                n: 3,
                min: Duration::from_nanos(100),
                median: Duration::from_nanos(500),
                max: Duration::from_nanos(900)
            }
        );
        assert_eq!(summaries[0]["sequences"], json!([0, 1, 2]));
        // Aggregation never overwrites a raw phase with another trial's median.
        assert_eq!(serde_json::to_string(&read).unwrap(), encoded);
        assert_eq!(
            read[0].calls[0].as_ref().unwrap().forward_eval.as_nanos(),
            450
        );
        assert!(read[0].calls[0].as_ref().unwrap().tokenize.is_none());
        assert!(read[5].calls[0].is_none());
        assert_eq!(
            distribution(vec![Duration::from_nanos(100), Duration::from_nanos(300)]).median,
            Duration::from_nanos(200)
        );
    }

    #[test]
    fn metal_version_omits_local_installation_paths() {
        let output = "Apple metal version 40000.1\nTarget: air64-apple-darwin\nInstalledDir: /private/local/toolchain/bin\n";
        assert_eq!(
            metal_version(output).unwrap(),
            "Apple metal version 40000.1\nTarget: air64-apple-darwin"
        );
        assert_eq!(metal_version("InstalledDir: /private/local/bin"), None);
    }

    #[test]
    fn scheduling_alternates_batch_and_sequential() {
        assert_eq!(
            (0..3).map(order).collect::<Vec<_>>(),
            [
                [Method::Batch, Method::Sequential],
                [Method::Sequential, Method::Batch],
                [Method::Batch, Method::Sequential]
            ]
        );
    }
}
