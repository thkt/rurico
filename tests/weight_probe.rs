//! The real probe dispatcher must reject incomplete checkpoints before MLX
//! allocation. These tests require neither cached models nor GPU execution.
#![cfg(feature = "test-support")]

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;

use rurico::{embed, reranker};
use serde_json::Value;
use tokenizers::{Tokenizer, models::bpe::BPE};

#[test]
fn constructors_and_probe_children_reject_missing_required_weight() {
    for (kind, binary, config, header, missing) in [
        (
            "embed",
            env!("CARGO_BIN_EXE_probe_embed_smoke"),
            include_str!("fixtures/modernbert_configs/ruri-v3-310m.json"),
            include_str!("fixtures/modernbert_weights/ruri-v3-310m.header.json"),
            "final_norm.weight",
        ),
        (
            "reranker",
            env!("CARGO_BIN_EXE_probe_reranker_smoke"),
            include_str!("fixtures/modernbert_configs/ruri-v3-reranker-310m.json"),
            include_str!("fixtures/modernbert_weights/ruri-v3-reranker-310m.header.json"),
            "classifier.bias",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut header: Value = serde_json::from_str(header).unwrap();
        let payload_len = header
            .as_object()
            .unwrap()
            .values()
            .filter_map(|t| t["data_offsets"][1].as_u64())
            .max()
            .unwrap();
        header.as_object_mut().unwrap().remove(missing).unwrap();
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut model = File::create(dir.path().join("model.safetensors")).unwrap();
        model
            .write_all(&(bytes.len() as u64).to_le_bytes())
            .unwrap();
        model.write_all(&bytes).unwrap();
        model.set_len(8 + bytes.len() as u64 + payload_len).unwrap();
        fs::write(dir.path().join("config.json"), config).unwrap();
        Tokenizer::new(BPE::default())
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        let reason = format!("weights: missing key: {missing}");
        let (prefix, error) = if kind == "embed" {
            let artifacts = embed::CandidateArtifacts::from_dir(dir.path())
                .verify()
                .unwrap();
            (
                "__RURICO_PROBE",
                embed::Embedder::new(&artifacts).unwrap_err(),
            )
        } else {
            let artifacts = reranker::CandidateArtifacts::from_dir(dir.path())
                .verify()
                .unwrap();
            (
                "__RURICO_RERANKER_PROBE",
                reranker::Reranker::new(&artifacts).unwrap_err(),
            )
        };
        assert!(error.to_string().contains(&reason), "{error}");
        let output = Command::new(binary)
            .env_remove("__RURICO_PROBE_MODEL")
            .env_remove("__RURICO_RERANKER_PROBE_MODEL")
            .env("HF_HUB_CACHE", dir.path())
            .env(
                format!("{prefix}_MODEL"),
                dir.path().join("model.safetensors"),
            )
            .env(format!("{prefix}_CONFIG"), dir.path().join("config.json"))
            .env(
                format!("{prefix}_TOKENIZER"),
                dir.path().join("tokenizer.json"),
            )
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{kind}: {stderr}");
        assert!(stderr.contains(&reason), "{kind}: {stderr}");
    }
}
