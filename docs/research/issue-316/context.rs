//! Research-only context selection with the existing tokenizer/forward path.
use std::{cmp::Reverse, collections::HashSet, env, fs, io::Write, path::PathBuf, time::Instant};

use mlx_rs::memory;
use serde::Deserialize;
use serde_json::{Value, json};
use tokenizers::{Encoding, Tokenizer};

use super::RerankerInner;
use crate::model_io::{BUCKET_BOUNDS, MAX_SEQ_LEN, assign_bucket};
use crate::reranker::processing::truncate_pair;
use crate::reranker::{RerankerModelId, cached_artifacts};
use crate::sandbox::require_unsandboxed_mlx_runtime;

const QUERY_CAP: usize = 1024;
const CHUNK: usize = 512;
const OVERLAP: usize = 128;

#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
    evidence_char_span: Option<[usize; 2]>,
}
#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
}
#[derive(Deserialize)]
struct Inputs {
    documents: Vec<Document>,
    queries: Vec<Query>,
}

fn slice(encoded: &Encoding, start: usize, end: usize) -> Encoding {
    // Do not carry the standalone sequence-0 range into pair postprocessing:
    // set_sequence_id(1) adds a range without removing the old sequence-0 range.
    // Preserve token/offset data; the postprocessor assigns fresh pair ranges.
    Encoding::new(
        encoded.get_ids()[start..end].to_vec(),
        encoded.get_type_ids()[start..end].to_vec(),
        encoded.get_tokens()[start..end].to_vec(),
        encoded.get_word_ids()[start..end].to_vec(),
        encoded.get_offsets()[start..end].to_vec(),
        encoded.get_special_tokens_mask()[start..end].to_vec(),
        encoded.get_attention_mask()[start..end].to_vec(),
        Vec::new(),
        Default::default(),
    )
}

fn ranges(length: usize, capacity: usize, overlap: usize) -> Vec<(usize, usize)> {
    assert!(capacity > overlap);
    let mut result = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + capacity).min(length);
        result.push((start, end));
        if end == length {
            break;
        }
        start = end - overlap;
    }
    result
}

fn char_span(document: &Document) -> Option<[usize; 2]> {
    document.evidence_char_span.map(|[a, b]| {
        assert!(a < b && b <= document.text.chars().count());
        let byte_at = |n| {
            document
                .text
                .char_indices()
                .nth(n)
                .map_or(document.text.len(), |(i, _)| i)
        };
        [byte_at(a), byte_at(b)]
    })
}

fn observe(
    encoding: &Encoding,
    sequences: &[Option<usize>],
    ids: &[u32],
    mask: &[u32],
    evidence: Option<[usize; 2]>,
) -> Value {
    let retained = ids.len();
    let offsets = encoding.get_offsets();
    let specials = encoding.get_special_tokens_mask();
    let evidence_indices: Vec<_> = evidence.map_or_else(Vec::new, |[a, b]| {
        (0..encoding.len())
            .filter(|&i| {
                sequences[i] == Some(1) && specials[i] == 0 && offsets[i].0 < b && offsets[i].1 > a
            })
            .collect()
    });
    // Baseline truncate overwrites the final retained position with EOS.
    let is_retained = |i: usize| i < retained && ids[i] == encoding.get_ids()[i];
    let mut model_sequences = sequences[..retained].to_vec();
    let mut model_offsets = offsets[..retained].to_vec();
    let mut model_specials = specials[..retained].to_vec();
    for i in 0..retained {
        if !is_retained(i) {
            model_sequences[i] = None;
            model_offsets[i] = (0, 0);
            model_specials[i] = 1;
        }
    }
    json!({"ids":ids,"mask":mask,
        "sequence_ids":model_sequences,
        "offsets_bytes":model_offsets,
        "special_tokens_mask":model_specials,
        "evidence_present":evidence.is_some(),
        "evidence_token_count":evidence_indices.len(),
        "evidence_retained":evidence.is_some() && !evidence_indices.is_empty() && evidence_indices.iter().copied().all(is_retained)})
}

/// Invoked only by the explicit smoke-feature research binary.
pub fn run() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "context_spike plan|model INPUTS STRATEGY OUTPUT"
    );
    let mode = &args[1];
    assert!(matches!(mode.as_str(), "plan" | "model"));
    let strategy = &args[3];
    assert!(matches!(
        strategy.as_str(),
        "baseline" | "query-prefix" | "chunk-neighbors" | "windows"
    ));
    if mode == "model" {
        require_unsandboxed_mlx_runtime();
    }
    let input: Inputs = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    assert!(!input.documents.is_empty() && !input.queries.is_empty());
    let artifacts = cached_artifacts(RerankerModelId::RuriV3Reranker310m)
        .unwrap()
        .expect("fixed cached model required; no download from this entry");
    let snapshot = PathBuf::from(env::var("RURICO_316_MODEL_DIR").unwrap());
    assert_eq!(
        snapshot.file_name().unwrap(),
        RerankerModelId::RuriV3Reranker310m.revision()
    );
    for (actual, name) in [
        (&artifacts.paths.model, "model.safetensors"),
        (&artifacts.paths.config, "config.json"),
        (&artifacts.paths.tokenizer, "tokenizer.json"),
    ] {
        assert_eq!(
            fs::canonicalize(actual).unwrap(),
            fs::canonicalize(snapshot.join(name)).unwrap()
        );
    }
    let tokenizer: Tokenizer = artifacts.tokenizer.clone();
    assert!(tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none());
    let empty = tokenizer.encode("", false).unwrap();
    let overhead = tokenizer
        .post_process(empty.clone(), Some(empty), true)
        .unwrap()
        .len();
    assert!(overhead > 0 && overhead < MAX_SEQ_LEN - QUERY_CAP);
    let load = Instant::now();
    let mut model = (mode == "model").then(|| RerankerInner::new(&artifacts).unwrap());
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[4])
        .unwrap();
    writeln!(output, "{}", json!({"event":"context","mode":mode,"strategy":strategy,
        "query_cap":QUERY_CAP,"query_selection":"tail","chunk_tokens":CHUNK,
        "window_overlap":OVERLAP,"special_overhead":overhead,
        "load_ns":load.elapsed().as_nanos(),"padding":"fixed bucket, singleton; unchanged forward/cleanup"})).unwrap();
    for query in &input.queries {
        // First call and three warm calls share identical input and policy.
        let trials = if mode == "model" { 4 } else { 1 };
        for trial in 0..trials {
            if model.is_some() {
                memory::reset_peak_memory().unwrap();
            }
            let start = Instant::now();
            let raw_query = tokenizer.encode(query.text.as_str(), false).unwrap();
            let selected_query = slice(
                &raw_query,
                raw_query.len().saturating_sub(QUERY_CAP),
                raw_query.len(),
            );
            let capacity = MAX_SEQ_LEN - overhead - selected_query.len();
            let query_ids: HashSet<_> = selected_query.get_ids().iter().copied().collect();
            let mut records = Vec::new();
            let mut forwards = Vec::new();
            for document in &input.documents {
                let raw = tokenizer
                    .encode((query.text.as_str(), document.text.as_str()), true)
                    .unwrap();
                let doc = tokenizer.encode(document.text.as_str(), false).unwrap();
                let evidence = char_span(document);
                // The original document and evidence span are unchanged across windows.
                let full_evidence_indices: Vec<_> = evidence.map_or_else(Vec::new, |[x, y]| {
                    doc.get_offsets()
                        .iter()
                        .enumerate()
                        .filter(|&(_, &(s, e))| s < y && e > x)
                        .map(|(i, _)| i)
                        .collect()
                });
                let selections = match strategy.as_str() {
                    "baseline" => vec![(0, doc.len())],
                    "query-prefix" => vec![(0, capacity.min(doc.len()))],
                    "windows" => ranges(doc.len(), capacity, OVERLAP),
                    "chunk-neighbors" => {
                        let chunks = ranges(doc.len(), CHUNK, 0);
                        let best = chunks
                            .iter()
                            .enumerate()
                            .max_by_key(|&(i, &(a, b))| {
                                let matches = doc.get_ids()[a..b]
                                    .iter()
                                    .filter(|id| query_ids.contains(id))
                                    .count();
                                (matches, Reverse(i))
                            })
                            .unwrap()
                            .0;
                        let a = chunks[best.saturating_sub(1)].0;
                        let b = chunks[(best + 1).min(chunks.len() - 1)].1;
                        assert!(b - a <= capacity);
                        vec![(a, b)]
                    }
                    _ => unreachable!(),
                };
                let mut windows = Vec::new();
                for &(a, b) in &selections {
                    let encoded = if strategy == "baseline" {
                        raw.clone()
                    } else {
                        tokenizer
                            .post_process(selected_query.clone(), Some(slice(&doc, a, b)), true)
                            .unwrap()
                    };
                    let mut ids = encoded.get_ids().to_vec();
                    let mut mask = encoded.get_attention_mask().to_vec();
                    if strategy == "baseline" {
                        truncate_pair(&mut ids, &mut mask, MAX_SEQ_LEN, records.len());
                    }
                    assert!(!ids.is_empty() && ids.len() == mask.len() && ids.len() <= MAX_SEQ_LEN);
                    let sequences = encoded.get_sequence_ids();
                    let mut observation = observe(&encoded, &sequences, &ids, &mask, evidence);
                    for (sequence, text_field, count_field) in [
                        (0, "retained_query_text", "query_tokens_retained"),
                        (1, "retained_document_text", "document_tokens_retained"),
                    ] {
                        let text_ids: Vec<_> = ids
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|&(i, id)| {
                                sequences[i] == Some(sequence) && encoded.get_ids()[i] == id
                            })
                            .map(|(_, id)| id)
                            .collect();
                        observation[count_field] = json!(text_ids.len());
                        observation[text_field] =
                            json!(tokenizer.decode(&text_ids, false).unwrap());
                    }
                    // Local window evidence is complete only if it covers all full-document evidence tokens.
                    if evidence.is_some() {
                        observation["evidence_retained"] = json!(
                            observation["evidence_retained"] == true
                                && full_evidence_indices.iter().all(|&i| a <= i && i < b)
                        );
                    }
                    observation["document_token_range"] = json!([a, b]);
                    if let Some(model) = &mut model {
                        let bucket = BUCKET_BOUNDS[assign_bucket(ids.len())];
                        let clock = Instant::now();
                        let scores = model.forward_sub_batch(&[ids], &[mask], bucket).unwrap();
                        assert_eq!(scores.len(), 1);
                        assert!(scores[0].is_finite());
                        observation["score"] = json!(scores[0]);
                        forwards.push(json!({"document":document.id,"range":[a,b],"rows":1,
                            "bucket":bucket,"host_forward_ns":clock.elapsed().as_nanos()}));
                    }
                    windows.push(observation);
                }
                records.push(json!({"document":document.id,"raw_ids":raw.get_ids(),
                    "raw_mask":raw.get_attention_mask(),"raw_special_tokens_mask":raw.get_special_tokens_mask(),
                    "raw_sequence_ids":raw.get_sequence_ids(),"raw_offsets_bytes":raw.get_offsets(),
                    "raw_query_tokens":raw_query.len(),"raw_document_tokens":doc.len(),"windows":windows}));
            }
            let wall_ns = start.elapsed().as_nanos();
            let peak = model.as_ref().map(|_| memory::peak_memory().unwrap());
            writeln!(
                output,
                "{}",
                json!({"event":"call","query":query.id,"trial":trial,
                "cold":trial==0,"wall_ns":wall_ns,"metal_peak_bytes":peak,
                "forwards":forwards,"documents":records})
            )
            .unwrap();
            output.flush().unwrap();
        }
    }
}
