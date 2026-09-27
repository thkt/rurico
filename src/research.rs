//! Issue #307 host-only observations. No production API or inference changes.
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::embed::research::capture as capture_embedding;
use crate::model_io::{BUCKET_BOUNDS, ModelPaths, assign_bucket, pad_sequences};
use crate::reranker::capture_research;
use crate::sandbox::require_unsandboxed_mlx_runtime;

pub(crate) fn verify_paths(paths: &ModelPaths) {
    let root =
        PathBuf::from(env::var("RURICO_307_MODEL_DIR").expect("set pinned snapshot directory"));
    for (actual, name) in [
        (&paths.model, "model.safetensors"),
        (&paths.config, "config.json"),
        (&paths.tokenizer, "tokenizer.json"),
    ] {
        assert_eq!(
            fs::canonicalize(actual).unwrap(),
            fs::canonicalize(root.join(name)).unwrap(),
            "Rust and Python must read the same verified files"
        );
    }
}

#[derive(Deserialize)]
pub(crate) struct Input {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub query: String,
    pub repeat: usize,
}
impl Input {
    pub fn text(&self) -> String {
        self.text.repeat(self.repeat)
    }
}

#[derive(Deserialize)]
struct Inputs {
    embedding: Vec<Input>,
    reranker: Vec<Input>,
}

#[derive(Serialize)]
pub(crate) struct Batch {
    pub id: String,
    pub conditions: Vec<&'static str>,
    pub keys: Vec<String>,
    pub ids: Vec<Vec<u32>>,
    pub mask: Vec<Vec<u32>>,
    pub values: Vec<Vec<f32>>,
    pub hidden_probes: Vec<Vec<Vec<f32>>>,
}

/// Compare exact length and rounded bucket; mixed short/long rows expose masking
/// and row-restoration bugs without batching multiple 8192-token CPU references.
pub(crate) fn batches(rows: &[(String, Vec<u32>, Vec<u32>)]) -> Vec<Batch> {
    let mut result = Vec::new();
    for (key, ids, mask) in rows {
        let bucket = BUCKET_BOUNDS[assign_bucket(ids.len())];
        let mut exact = batch(
            format!("{key}/exact"),
            &[(key, ids, mask)],
            ids.len(),
            vec!["exact"],
        );
        if ids.len() == bucket {
            exact.conditions.push("bucket");
        }
        result.push(exact);
        if ids.len() != bucket {
            result.push(batch(
                format!("{key}/bucket"),
                &[(key, ids, mask)],
                bucket,
                vec!["bucket"],
            ));
        }
    }
    let mixed: Vec<_> = rows
        .iter()
        .filter(|(_, ids, _)| ids.len() <= 512)
        .take(5)
        .map(|(key, ids, mask)| (key, ids, mask))
        .collect();
    if mixed.len() >= 2 {
        let max_len = mixed.iter().map(|(_, ids, _)| ids.len()).max().unwrap();
        result.push(batch(
            "mixed/bucket".into(),
            &mixed,
            BUCKET_BOUNDS[assign_bucket(max_len)],
            vec!["bucket"],
        ));
    }
    result
}

fn batch(
    id: String,
    rows: &[(&String, &Vec<u32>, &Vec<u32>)],
    len: usize,
    conditions: Vec<&'static str>,
) -> Batch {
    let ids: Vec<_> = rows.iter().map(|(_, ids, _)| (*ids).clone()).collect();
    let masks: Vec<_> = rows.iter().map(|(_, _, mask)| (*mask).clone()).collect();
    let (ids, mask, _, _) = pad_sequences(&ids, Some(&masks), Some(len));
    Batch {
        id,
        conditions,
        keys: rows.iter().map(|(key, _, _)| (*key).clone()).collect(),
        ids: ids.chunks(len).map(<[u32]>::to_vec).collect(),
        mask: mask.chunks(len).map(<[u32]>::to_vec).collect(),
        values: Vec::new(),
        hidden_probes: Vec::new(),
    }
}

pub(crate) fn shape(batch: &Batch) -> (Vec<u32>, Vec<u32>, i32, i32) {
    (
        batch.ids.concat(),
        batch.mask.concat(),
        i32::try_from(batch.ids.len()).unwrap(),
        i32::try_from(batch.ids[0].len()).unwrap(),
    )
}

pub(crate) fn save(kind: &str, tokens: &[Value], batches: &[Batch], public: &Value) {
    let out = env::var("RURICO_307_OUTPUT").expect("set RURICO_307_OUTPUT to a new JSON file");
    let value = json!({"schema":2,"producer":"rurico","kind":kind,
        "tokenization":tokens,"batches":batches,"public":public});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .unwrap();
    serde_json::to_writer(&mut file, &value).unwrap();
    file.write_all(b"\n").unwrap();
}

fn inputs() -> Inputs {
    let path = env::var("RURICO_307_INPUTS").expect("set RURICO_307_INPUTS");
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn batches_share_only_equal_shapes_within_a_row() {
    let rows = vec![
        ("short".into(), vec![7; 127], vec![1; 127]),
        ("boundary".into(), vec![8; 128], vec![1; 128]),
        ("long".into(), vec![9; 8192], vec![1; 8192]),
        ("other-long".into(), vec![9; 8192], vec![1; 8192]),
    ];
    let records = batches(&rows);
    assert_eq!(
        records.iter().map(|b| b.id.as_str()).collect::<Vec<_>>(),
        [
            "short/exact",
            "short/bucket",
            "boundary/exact",
            "long/exact",
            "other-long/exact",
            "mixed/bucket"
        ]
    );
    assert_eq!(records[0].ids[0], vec![7; 127]);
    assert_eq!(records[1].ids[0][127], 0);
    assert_eq!(records[1].mask[0][127], 0);
    assert_eq!(records[2].ids[0], vec![8; 128]);
    assert_eq!(records[3].ids[0], vec![9; 8192]);
    assert_eq!(records[0].conditions, ["exact"]);
    assert_eq!(records[1].conditions, ["bucket"]);
    for record in &records[2..5] {
        assert_eq!(record.conditions, ["exact", "bucket"]);
    }
    assert_eq!(records[5].conditions, ["bucket"]);
    assert_eq!(records[5].keys, ["short", "boundary"]);
    assert_eq!(records[5].ids[0], records[1].ids[0]);
    assert_eq!(records[5].mask[0], records[1].mask[0]);
}

#[test]
#[ignore = "Issue #307: fixed cached models, unsandboxed Metal, explicit output required"]
fn official_comparison_embedding() {
    require_unsandboxed_mlx_runtime();
    capture_embedding(&inputs().embedding);
}

#[test]
#[ignore = "Issue #307: fixed cached models, unsandboxed Metal, explicit output required"]
fn official_comparison_reranker() {
    require_unsandboxed_mlx_runtime();
    capture_research(&inputs().reranker);
}
