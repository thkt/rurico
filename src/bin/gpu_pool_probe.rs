//! Phase 3a precursor probe bin (FR-004 / FR-004a / NFR-003).
//!
//! Loads the cached `cl-nagoya/ruri-v3-310m` model, tokenizes the first W1
//! long document with `DOCUMENT_PREFIX`, takes the first `MAX_SEQ_LEN` tokens
//! (`seq_len = 8192`), and runs one forward pass. The resulting
//! `[1, 8192, 768]` hidden-states tensor is fed to both
//! [`gpu_pool_and_normalize`] and the CPU reference [`postprocess_embedding`]
//! using the same input, so any divergence is reduction-order drift rather
//! than input drift.
//!
//! On stderr the probe emits `max_abs_diff=<f32>`, `cosine_sim=<f32>`, and a
//! `margin_10x: PASS|FAIL (threshold=1e-6)` banner. Exit code is `0` on
//! `max_abs_diff ≤ 1e-6`, `1` on the margin violation, and `2` on setup
//! failure (missing cache, tokenize error, etc.). These three exit classes
//! let integration tests in `tests/gpu_pool_probe.rs` distinguish expected
//! margin failure from environmental failure.
//!
//! # Flags
//!
//! - `--force-fail`: perturbs the CPU reference vector by `+1e-3` on its
//!   first element so the `margin_10x: FAIL` branch can be exercised without
//!   injecting reduction-order drift into the real backend. Used by T-013.
//!
//! # Requires
//!
//! - `cl-nagoya/ruri-v3-310m` downloaded and cached under the local HF Hub
//!   cache (run `rurico` or `mlx_smoke` once to populate it).
//! - Unsandboxed MLX runtime (Apple Silicon). The `exit_if_seatbelt` guard
//!   early-returns `SEATBELT_SKIP_EXIT` inside the Codex sandbox.

use std::env;
use std::error::Error;
use std::process;

use mlx_rs::Array;
use rurico::embed::{
    self, DOCUMENT_PREFIX, MAX_SEQ_LEN, cached_artifacts, gpu_pool_and_normalize,
    postprocess_embedding, tokenize_with_prefix, workloads::workload_w1,
};
use rurico::modernbert::ModernBert;
use rurico::sandbox;

/// NFR-003 precision margin: 10x over NFR-001's `1e-5` fixture tolerance.
const MARGIN_10X: f32 = 1e-6;

struct ProbeReport {
    max_abs_diff: f32,
    cosine_sim: f32,
}

impl ProbeReport {
    fn passes_margin(&self) -> bool {
        self.max_abs_diff <= MARGIN_10X
    }
}

fn main() {
    sandbox::exit_if_seatbelt(env!("CARGO_BIN_NAME"));

    let force_fail = env::args().any(|a| a == "--force-fail");

    match run(force_fail) {
        Ok(report) => {
            eprintln!("max_abs_diff={:.9e}", report.max_abs_diff);
            eprintln!("cosine_sim={:.9}", report.cosine_sim);
            if report.passes_margin() {
                eprintln!("margin_10x: PASS (threshold=1e-6)");
                process::exit(0);
            } else {
                eprintln!("margin_10x: FAIL (threshold=1e-6)");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("gpu_pool_probe: setup error: {e}");
            process::exit(2);
        }
    }
}

fn run(force_fail: bool) -> Result<ProbeReport, Box<dyn Error>> {
    let artifacts = cached_artifacts(embed::ModelId::default())?
        .ok_or("ruri-v3-310m not cached; download the embed model before running the probe")?;
    let mut model = ModernBert::load(&artifacts.paths().model, artifacts.config())?;

    let w1 = workload_w1();
    let text = &w1[0];
    let tokenized = tokenize_with_prefix(artifacts.tokenizer(), text, DOCUMENT_PREFIX)?;
    if tokenized.seq_len <= MAX_SEQ_LEN {
        return Err(format!(
            "W1[0] tokenized seq_len={} ≤ MAX_SEQ_LEN={}; probe expects a long chunk",
            tokenized.seq_len, MAX_SEQ_LEN
        )
        .into());
    }

    let seq_len = MAX_SEQ_LEN;
    let ids: Vec<u32> = tokenized.input_ids[..seq_len].to_vec();
    let mask: Vec<u32> = vec![1; seq_len];

    let seq_i32 = i32::try_from(seq_len)?;
    let hidden = model.forward(&ids, &mask, 1, seq_i32)?;
    hidden.eval()?;

    // CPU reference first: snapshot the flat hidden buffer before the GPU
    // path consumes `hidden`. The GPU path is value-consuming by design
    // (NFR-005 / ADR 0002 sub-decision 2), so this ordering is mandatory.
    let hidden_flat: &[f32] = hidden.as_slice();
    let mut cpu_pooled = postprocess_embedding(hidden_flat, seq_len, &mask)?;
    if force_fail && let Some(v) = cpu_pooled.first_mut() {
        *v += 1e-3;
    }

    let mask_arr = Array::from_slice(&mask, &[1, seq_i32]);
    let gpu_pooled = gpu_pool_and_normalize(hidden, &mask_arr)?;
    gpu_pooled.eval()?;
    let gpu_flat: &[f32] = gpu_pooled.as_slice();

    if gpu_flat.len() != cpu_pooled.len() {
        return Err(format!(
            "gpu_flat.len()={} != cpu_pooled.len()={}",
            gpu_flat.len(),
            cpu_pooled.len()
        )
        .into());
    }

    let max_abs_diff = gpu_flat
        .iter()
        .zip(cpu_pooled.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Both vectors are L2-normalized, so cosine similarity reduces to the
    // dot product. The force-fail perturbation breaks unit-norm on the CPU
    // side, but the probe gate reports on `max_abs_diff` — `cosine_sim` is
    // a diagnostic aid, not the threshold.
    let cosine_sim: f32 = gpu_flat
        .iter()
        .zip(cpu_pooled.iter())
        .map(|(a, b)| a * b)
        .sum();

    Ok(ProbeReport {
        max_abs_diff,
        cosine_sim,
    })
}
