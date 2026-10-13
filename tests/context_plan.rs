//! Issue #316: exercise the research CLI with an isolated, synthetic tokenizer
//! and header-only cache. These are context/record contracts, not model scores.
#![cfg(feature = "smoke")]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{Value, json};
use tokenizers::{
    Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
    processors::template::TemplateProcessing,
};

const REVISION: &str = "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3";
const STRATEGIES: [&str; 4] = ["baseline", "query-prefix", "chunk-neighbors", "windows"];

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let snapshot = root.path().join(format!(
            "hub/models--cl-nagoya--ruri-v3-reranker-310m/snapshots/{REVISION}"
        ));
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("config.json"),
            include_str!("fixtures/modernbert_configs/ruri-v3-reranker-310m.json"),
        )
        .unwrap();
        // Kind verification reads only this header. No load/forward is allowed
        // in plan mode: these deliberately are not inference-capable weights.
        let header = serde_json::to_vec(&json!({
            "layers.0.attn.Wo.weight": {}, "classifier.weight": {},
            "head.dense.weight": {}, "head.norm.weight": {}
        }))
        .unwrap();
        let mut weights = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        weights.extend(header);
        fs::write(snapshot.join("model.safetensors"), weights).unwrap();
        let vocab = ["[UNK]", "[BOS]", "[EOS]", "[SEP]", "猫", "答", "fn", "7"]
            .into_iter()
            .enumerate()
            .map(|(i, word)| (word.to_owned(), u32::try_from(i).unwrap()))
            .collect();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[BOS] $A [EOS]")
                .unwrap()
                .try_pair("[BOS] $A [SEP] $B:1 [EOS]")
                .unwrap()
                .special_tokens(vec![("[BOS]", 1), ("[EOS]", 2), ("[SEP]", 3)])
                .build()
                .unwrap(),
        ));
        tokenizer
            .save(snapshot.join("tokenizer.json"), false)
            .unwrap();
        Self { root }
    }

    fn run(&self, inputs: &Value, strategy: &str, name: &str) -> (Output, PathBuf) {
        let input = self.root.path().join("input.json");
        fs::write(&input, serde_json::to_vec(inputs).unwrap()).unwrap();
        let output = self.root.path().join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_context_spike"))
            .args(["plan"])
            .arg(input)
            .arg(strategy)
            .arg(&output)
            .env("HF_HOME", self.root.path())
            .env("HF_HUB_CACHE", self.root.path().join("hub"))
            .env("HF_HUB_OFFLINE", "1")
            .env(
                "RURICO_316_MODEL_DIR",
                self.root.path().join(format!(
                    "hub/models--cl-nagoya--ruri-v3-reranker-310m/snapshots/{REVISION}"
                )),
            )
            .output()
            .unwrap();
        (result, output)
    }

    fn records(&self, inputs: &Value, strategy: &str) -> Vec<Value> {
        let (result, path) = self.run(inputs, strategy, &format!("{strategy}.jsonl"));
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let rows: Vec<Value> = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), inputs["queries"].as_array().unwrap().len() + 1);
        assert_eq!(rows[0]["mode"], "plan");
        for (row, query) in rows[1..].iter().zip(inputs["queries"].as_array().unwrap()) {
            assert_eq!(row["query"], query["id"]);
            assert_eq!(row["trial"], 0);
            assert_eq!(row["forwards"], json!([]));
            assert!(row["metal_peak_bytes"].is_null());
            let documents = row["documents"].as_array().unwrap();
            assert_eq!(
                documents.len(),
                inputs["documents"].as_array().unwrap().len()
            );
            for (record, document) in documents
                .iter()
                .zip(inputs["documents"].as_array().unwrap())
            {
                assert_eq!(record["document"], document["id"]);
                for window in record["windows"].as_array().unwrap() {
                    let ids = window["ids"].as_array().unwrap();
                    assert!(!ids.is_empty() && ids.len() <= 8192);
                    for field in [
                        "mask",
                        "sequence_ids",
                        "offsets_bytes",
                        "special_tokens_mask",
                    ] {
                        assert_eq!(window[field].as_array().unwrap().len(), ids.len());
                    }
                    assert!(window["mask"].as_array().unwrap().iter().all(|v| v == 1));
                    assert!(window.get("score").is_none());
                    if strategy != "baseline" {
                        assert_eq!(ids.first(), Some(&json!(1)));
                        assert_eq!(ids.last(), Some(&json!(2)));
                        assert_eq!(ids.iter().filter(|id| **id == 3).count(), 1);
                        assert_eq!(
                            window["special_tokens_mask"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .filter(|v| **v == 1)
                                .count(),
                            3
                        );
                    }
                }
            }
        }
        rows
    }
}

#[test]
fn short_japanese_and_code_pairs_preserve_boundaries_offsets_and_context() {
    let input = json!({"queries":[{"id":"ja","text":"答"},{"id":"code","text":"fn 7"}],
        "documents":[{"id":"ja","text":"猫 答 猫","evidence_char_span":[2,3]},
            {"id":"code","text":"fn 7","evidence_char_span":[3,4]},
            {"id":"empty","text":""}, {"id":"irrelevant","text":"猫"}]});
    for strategy in STRATEGIES {
        let rows = Fixture::new().records(&input, strategy);
        for row in &rows[1..] {
            for record in row["documents"].as_array().unwrap() {
                let windows = record["windows"].as_array().unwrap();
                assert_eq!(windows.len(), 1);
                assert_eq!(windows[0]["ids"], record["raw_ids"]);
                assert_eq!(windows[0]["sequence_ids"], record["raw_sequence_ids"]);
                assert_eq!(windows[0]["offsets_bytes"], record["raw_offsets_bytes"]);
                let expected = matches!(record["document"].as_str().unwrap(), "ja" | "code");
                assert_eq!(windows[0]["evidence_retained"], expected);
            }
            // Unicode code-point span [2,3] denotes 答, at byte span [4,7].
            let ja = &row["documents"][0]["windows"][0];
            assert_eq!(
                ja["sequence_ids"],
                if row["query"] == "ja" {
                    json!([null, 0, null, 1, 1, 1, null])
                } else {
                    json!([null, 0, 0, null, 1, 1, 1, null])
                }
            );
            let position = ja["ids"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .find(|(i, id)| **id == 5 && ja["sequence_ids"][*i] == 1)
                .unwrap()
                .0;
            assert_eq!(ja["offsets_bytes"][position], json!([4, 7]));
            assert_eq!(ja["evidence_token_count"], 1);
        }
    }
}

#[test]
fn long_pairs_expose_lost_evidence_and_windows_cover_the_same_candidates() {
    let documents: Vec<_> = [("head", 0), ("middle", 4500), ("tail", 9000)]
        .into_iter()
        .map(|(id, before)| {
            let text = "猫 ".repeat(before) + "答 " + &"猫 ".repeat(9000 - before);
            json!({"id":id,"text":text,"evidence_char_span":[before*2,before*2+1]})
        })
        .collect();
    let input = json!({"documents":documents,"queries":[
        {"id":"short","text":"答"},{"id":"long","text":long_query()}]});
    for strategy in STRATEGIES {
        let rows = Fixture::new().records(&input, strategy);
        for (q, row) in rows[1..].iter().enumerate() {
            for (d, record) in row["documents"].as_array().unwrap().iter().enumerate() {
                assert_eq!(record["raw_query_tokens"], if q == 0 { 1 } else { 8300 });
                assert_eq!(record["raw_document_tokens"], 9001);
                let windows = record["windows"].as_array().unwrap();
                if strategy == "baseline" {
                    let window = &windows[0];
                    assert_eq!(window["ids"].as_array().unwrap().len(), 8192);
                    assert_eq!(window["ids"][8191], 2);
                    assert_eq!(window["sequence_ids"][8191], Value::Null);
                    assert_eq!(window["offsets_bytes"][8191], json!([0, 0]));
                    assert_eq!(window["special_tokens_mask"][8191], 1);
                    assert_eq!(window["evidence_retained"], q == 0 && d != 2);
                    if q == 1 {
                        assert_eq!(window["document_tokens_retained"], 0);
                        assert_eq!(window["query_tokens_retained"], 8190);
                        let mut expected = vec![json!(1)];
                        expected.extend(vec![json!(6); 7276]);
                        expected.extend((0..914).map(|i| json!(if i % 2 == 0 { 5 } else { 7 })));
                        expected.push(json!(2));
                        assert_eq!(window["ids"], json!(expected));
                    }
                } else {
                    // Independent, hand-derived ranges for this input, not the
                    // ranges reported by the planner under test. See #316 README
                    // for comparison parameters (not an adopted product policy).
                    let expected_ranges = match strategy {
                        "query-prefix" => vec![(0, if q == 0 { 8188 } else { 7165 })],
                        "chunk-neighbors" => vec![[(0, 1024), (3584, 5120), (8192, 9001)][d]],
                        "windows" if q == 0 => vec![(0, 8188), (8060, 9001)],
                        "windows" => vec![(0, 7165), (7037, 9001)],
                        _ => unreachable!(),
                    };
                    assert_eq!(windows.len(), expected_ranges.len());
                    for (window, &(start, end)) in windows.iter().zip(&expected_ranges) {
                        assert_eq!(window["document_token_range"], json!([start, end]));
                        assert_query_selection(window, q == 1);
                        assert_eq!(window["document_tokens_retained"], end - start);
                        assert!(window["document_tokens_retained"].as_u64().unwrap() > 0);
                        let a = window["document_token_range"][0].as_u64().unwrap();
                        let b = window["document_token_range"][1].as_u64().unwrap();
                        assert!(a < b && b <= 9001);
                        let actual: Vec<_> = window["ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| window["sequence_ids"][*i] == 1)
                            .map(|(_, id)| id.as_u64().unwrap())
                            .collect();
                        let evidence_position = [0, 4500, 9000][d];
                        let expected: Vec<_> = (a..b)
                            .map(|i| if i == evidence_position { 5 } else { 4 })
                            .collect();
                        assert_eq!(actual, expected);
                        assert_eq!(
                            window["evidence_retained"],
                            a <= evidence_position && evidence_position < b
                        );
                    }
                    if strategy == "windows" {
                        let mut covered = vec![false; 9001];
                        let mut previous_end = 0;
                        for window in windows {
                            let start = usize::try_from(
                                window["document_token_range"][0].as_u64().unwrap(),
                            )
                            .unwrap();
                            let end = usize::try_from(
                                window["document_token_range"][1].as_u64().unwrap(),
                            )
                            .unwrap();
                            assert!(start <= previous_end && end > previous_end);
                            covered[start..end].fill(true);
                            previous_end = end;
                        }
                        assert!(covered.into_iter().all(|v| v));
                        assert!(windows.iter().any(|w| w["evidence_retained"] == true));
                    }
                }
            }
        }
    }
}

// The discarded prefix and retained tail have different token IDs. Alternating
// Japanese/code tokens in the tail exposes reordering and duplicate loss too.
fn long_query() -> String {
    "fn ".repeat(7276) + &"答 7 ".repeat(512)
}

fn assert_query_selection(window: &Value, long: bool) {
    let expected: Vec<u64> = if long { [5, 7].repeat(512) } else { vec![5] };
    let positions: Vec<_> = window["sequence_ids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, sequence)| **sequence == 0)
        .map(|(i, _)| i)
        .collect();
    let actual: Vec<_> = positions
        .iter()
        .map(|&i| window["ids"][i].as_u64().unwrap())
        .collect();
    assert_eq!(window["query_tokens_retained"], expected.len());
    assert_eq!(actual, expected);
    // Offsets identify the suffix in the original input even with repeated IDs.
    for (index, &position) in positions.iter().enumerate() {
        let offset = if long {
            let base = 7276 * 3 + (index / 2) * 6;
            if index % 2 == 0 {
                [base, base + 3]
            } else {
                [base + 4, base + 5]
            }
        } else {
            [0, 3]
        };
        assert_eq!(window["offsets_bytes"][position], json!(offset));
    }
}

#[test]
fn chunk_selection_counts_occurrences_ignores_discarded_query_and_breaks_ties_first() {
    let mut tie = vec!["猫"; 4096];
    tie[512] = "答";
    tie[2048] = "答";
    let mut frequency = vec!["猫"; 4096];
    // The prefix-only query token must not influence selection.
    frequency[..512].fill("fn");
    frequency[512] = "答";
    frequency[513] = "7";
    frequency[2048..2051].fill("答");
    let none = vec!["猫"; 4096];
    let tokens = [tie, frequency, none];
    let documents: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(i, words)| json!({"id":format!("d{i}"),"text":words.join(" ")}))
        .collect();
    let input = json!({"documents":documents,"queries":[{"id":"long","text":long_query()}]});
    let rows = Fixture::new().records(&input, "chunk-neighbors");
    // Equal maxima choose chunk 1, 3 occurrences beat 2 distinct matches,
    // and zero matches choose chunk 0. Include immediate neighbors only.
    for (i, &(start, end)) in [(0, 1536), (1536, 3072), (0, 1024)].iter().enumerate() {
        let windows = rows[1]["documents"][i]["windows"].as_array().unwrap();
        assert_eq!(windows.len(), 1);
        let window = &windows[0];
        assert_eq!(window["document_token_range"], json!([start, end]));
        assert_eq!(window["document_tokens_retained"], end - start);
        assert_query_selection(window, true);
        let expected: Vec<_> = tokens[i][start..end]
            .iter()
            .map(|word| match *word {
                "猫" => 4,
                "答" => 5,
                "7" => 7,
                "fn" => 6,
                _ => unreachable!(),
            })
            .collect();
        let actual: Vec<_> = window["ids"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(index, _)| window["sequence_ids"][*index] == 1)
            .map(|(_, id)| id.as_u64().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn partial_evidence_in_each_window_is_not_complete_full_document_evidence() {
    // Long-query document capacity: 8192 - BOS/SEP/EOS(3) - 1024 = 7165.
    // Windows [0,7165), [7037,9001) cover the document, but neither
    // contains [7000,7300). Same text/query/candidates give positive controls
    // at the overlap and at each exact edge, without using reported ranges.
    let spans: [(usize, usize); 4] = [(7000, 7300), (7037, 7165), (7000, 7165), (7037, 7300)];
    let documents: Vec<_> = spans
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            json!({"id":format!("span{i}"),"text":"猫 ".repeat(9001),
            "evidence_char_span":[start*2,end*2-1]})
        })
        .collect();
    let input = json!({"documents":documents,"queries":[
        {"id":"short","text":"答"},{"id":"long","text":long_query()}]});
    let rows = Fixture::new().records(&input, "windows");
    for (q, row) in rows[1..].iter().enumerate() {
        let expected_ranges = if q == 0 {
            [(0, 8188), (8060, 9001)]
        } else {
            [(0, 7165), (7037, 9001)]
        };
        for (d, &(evidence_start, evidence_end)) in spans.iter().enumerate() {
            let windows = row["documents"][d]["windows"].as_array().unwrap();
            assert_eq!(windows.len(), 2);
            for (window, &(start, end)) in windows.iter().zip(&expected_ranges) {
                assert_eq!(window["document_token_range"], json!([start, end]));
                assert_query_selection(window, q == 1);
                assert_eq!(window["document_tokens_retained"], end - start);
                assert_eq!(
                    window["evidence_token_count"],
                    evidence_end
                        .min(end)
                        .saturating_sub(evidence_start.max(start))
                );
                assert_eq!(
                    window["evidence_retained"],
                    start <= evidence_start && evidence_end <= end
                );
            }
        }
    }
}

#[test]
fn invalid_spans_fail_and_existing_records_are_preserved() {
    let fixture = Fixture::new();
    let valid = json!({"queries":[{"id":"q","text":"答"}],
        "documents":[{"id":"d","text":"答","evidence_char_span":[0,1]}]});
    fixture.records(&valid, "windows");
    let saved = fixture.root.path().join("windows.jsonl");
    let original = fs::read(&saved).unwrap();
    let (result, _) = fixture.run(&valid, "windows", "windows.jsonl");
    assert!(!result.status.success());
    assert_eq!(fs::read(saved).unwrap(), original);
    for span in [[0, 0], [0, 2]] {
        let mut invalid = valid.clone();
        invalid["documents"][0]["evidence_char_span"] = json!(span);
        let (result, _) = fixture.run(&invalid, "windows", &format!("invalid-{}.jsonl", span[1]));
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("a < b"));
    }
}
