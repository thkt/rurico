//! MLX smoke-test binary — subprocess-isolated model verification.
//!
//! Invoked by integration tests via `Command` so MLX FFI crashes
//! (SIGABRT) are contained without killing the test runner.
//!
//! Loads the default embed model from the local HF Hub cache. The model
//! must be downloaded before running smoke tests.
//!
//! # Modes
//!
//! - default (no args): the legacy smoke assertions that other integration
//!   tests rely on. Running bare `mlx_smoke` keeps the contract with
//!   `tests/mlx_smoke.rs`.
//! - `capture-fixture`: run W1/W2/W3 and write the current-branch output to
//!   `tests/fixtures/phase2_baseline/w{1,2,3}.bin` so later PRs can compare
//!   bucket-batched output against today's main-branch baseline.
//! - `measure-baseline`: run W1/W2/W3 timing `embed_documents_batch` and a
//!   sequential equivalent, emitting one `baseline[wN] ...` line per workload
//!   so the numbers can be copied into `docs/benchmarks/phase1_baseline.md`.

use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

use rurico::embed::{self, Embed, fixtures};
use rurico::model_probe;
use rurico::sandbox;

/// Minimal `log` crate subscriber that writes every record to stderr.
///
/// Kept inside the smoke binary so that debug logs emitted by the library are
/// visible when this binary runs, without forcing a logging backend on
/// library consumers.
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!(
                "[{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }
    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

fn init_logger() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);
}

fn main() {
    init_logger();

    // Also acts as a probe subprocess when probe env vars are set.
    model_probe::handle_probe_if_needed();

    sandbox::exit_if_seatbelt(env!("CARGO_BIN_NAME"));

    let artifacts = embed::cached_artifacts(embed::ModelId::default())
        .expect("cache lookup failed")
        .expect("model not cached; run download first");

    let embedder = embed::Embedder::new(&artifacts).expect("model load");

    let mode = env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "capture-fixture" => run_capture_fixture(&embedder),
        "measure-baseline" => run_measure_baseline(&embedder),
        _ => run_assertions(&embedder),
    }
}

// ── Legacy smoke assertions ──────────────────────────────────────────────────

fn run_assertions(embedder: &embed::Embedder) {
    let dims = embedder.embedding_dims();

    let q = embedder.embed_query("authentication logic").expect("query");
    assert_eq!(q.len(), dims, "query dims");

    let q2 = embedder
        .embed_query("authentication logic")
        .expect("query2");
    assert_eq!(q, q2, "deterministic");

    let d = embedder
        .embed_document("function useAuth() { return user; }")
        .expect("short doc");
    assert_eq!(d.chunks.len(), 1, "short doc: 1 chunk");
    assert_eq!(d.chunks[0].len(), dims, "short doc dims");

    let batch = embedder
        .embed_documents_batch(&[
            "function useAuth() { return user; }",
            "function Button() { return <div/>; }",
        ])
        .expect("batch");
    assert_eq!(batch.len(), 2, "batch count");

    let sentence = "apple pie is a traditional dessert enjoyed around the world. ";
    let long_text = sentence.repeat(800);
    let ld = embedder.embed_document(&long_text).expect("long doc");
    assert!(ld.chunks.len() >= 2, "long doc: ≥2 chunks");
    for (i, chunk) in ld.chunks.iter().enumerate() {
        assert_eq!(chunk.len(), dims, "long doc chunk {i} dims");
    }

    for text in ["apple pie", "the cat", "Rust", "This is a test"] {
        let r = embedder.embed_document(text).expect(text);
        assert_eq!(r.chunks.len(), 1, "'{text}': 1 chunk");
        assert_eq!(r.chunks[0].len(), dims, "'{text}' dims");
    }

    eprintln!("smoke: all checks passed");
}

// ── Phase 2 workloads ────────────────────────────────────────────────────────

fn workload_w1() -> Vec<String> {
    vec![
        "apple pie is a traditional dessert enjoyed around the world. ".repeat(800),
        "the rain in Spain falls mainly on the plain. ".repeat(500),
    ]
}

fn workload_w2() -> Vec<String> {
    (0..100)
        .map(|i| format!("short text number {i} for benchmarking W2 workload"))
        .collect()
}

fn workload_w3() -> Vec<String> {
    (0..5)
        .flat_map(|i| {
            vec![
                "benchmarking long text for W3 workload. ".repeat(100 + i * 10),
                format!("short text {i}"),
            ]
        })
        .collect()
}

// ── capture-fixture mode ─────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from("tests/fixtures/phase2_baseline")
}

fn as_refs(texts: &[String]) -> Vec<&str> {
    texts.iter().map(String::as_str).collect()
}

fn run_capture_fixture(embedder: &embed::Embedder) {
    let dir = fixture_dir();
    fs::create_dir_all(&dir).expect("create fixture dir");

    for (name, texts) in [
        ("w1", workload_w1()),
        ("w2", workload_w2()),
        ("w3", workload_w3()),
    ] {
        let refs = as_refs(&texts);
        let out = embedder
            .embed_documents_batch(&refs)
            .unwrap_or_else(|e| panic!("embed {name}: {e}"));
        let path = dir.join(format!("{name}.bin"));
        let file = File::create(&path).expect("create fixture file");
        let mut w = BufWriter::new(file);
        fixtures::save(&mut w, &out).expect("save fixture");
        eprintln!(
            "capture[{name}] wrote {} docs to {}",
            out.len(),
            path.display()
        );
    }
    eprintln!("capture-fixture: done");
}

// ── measure-baseline mode ────────────────────────────────────────────────────

fn run_measure_baseline(embedder: &embed::Embedder) {
    // warm-up: one full W1 batch before measuring so MLX compile cache is hot
    let warm = workload_w1();
    let refs = as_refs(&warm);
    let _ = embedder
        .embed_documents_batch(&refs)
        .expect("warm-up batch");

    for (name, texts) in [
        ("w1", workload_w1()),
        ("w2", workload_w2()),
        ("w3", workload_w3()),
    ] {
        let refs = as_refs(&texts);

        let t0 = Instant::now();
        let _batch = embedder.embed_documents_batch(&refs).expect("batch embed");
        let batch_ms = t0.elapsed().as_millis();

        let t1 = Instant::now();
        for text in &refs {
            let _ = embedder.embed_document(text).expect("sequential embed");
        }
        let sequential_ms = t1.elapsed().as_millis();

        let ratio = if sequential_ms > 0 {
            batch_ms as f64 / sequential_ms as f64
        } else {
            0.0
        };

        eprintln!(
            "baseline[{name}] num_texts={} batch_ms={batch_ms} sequential_ms={sequential_ms} \
             ratio={ratio:.3}",
            refs.len()
        );
    }
    eprintln!("measure-baseline: done");
}
