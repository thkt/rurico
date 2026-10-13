//! Measurement protocol shared by the fixed-model lane and model-free boundary tests.
use crate::embed::query317::{self, Queries, QueryBatch};
use crate::embed::{self, Embed, EmbedError, embedder};
use crate::reranker::{RankedResult, RerankerError, pairs317};
use crate::{reranker::mlx, sandbox::require_unsandboxed_mlx_runtime};
use serde_json::{Value, json};
use std::{env, fs::OpenOptions, io::Write, path::Path, time::Instant};
use tokenizers::Tokenizer;

fn record_at(path: &Path, value: &Value) {
    let mut file = OpenOptions::new().append(true).open(path).unwrap();
    writeln!(file, "{value}").unwrap();
}

pub(crate) struct Legacy<'a>(pub(crate) &'a dyn Embed);
impl Embed for Legacy<'_> {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.0.embed_query(text)
    }
    fn embed_document(&self, text: &str) -> Result<embed::ChunkedEmbedding, EmbedError> {
        self.0.embed_document(text)
    }
    fn embed_documents_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<embed::ChunkedEmbedding>, EmbedError> {
        self.0.embed_documents_batch(texts)
    }
    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.0.embed_text(text, prefix)
    }
}
impl QueryBatch for Legacy<'_> {}

pub(crate) fn query_workload(case: &str) -> Vec<String> {
    match case {
        "single" => vec!["図書館では日本語の本を借りられます。".into()],
        "multiple" => [
            "猫。",
            "",
            "図書館では日本語の本を借りられます。",
            "fn square(x: i32) -> i32 { x * x }",
            "猫。",
        ]
        .iter()
        .cycle()
        .take(32)
        .map(|s| (*s).into())
        .collect(),
        "boundary" => vec![
            "猫 ".repeat(4094),
            "短い query".into(),
            "猫 ".repeat(62),
            String::new(),
        ],
        _ => panic!("unknown query workload"),
    }
}
pub(crate) fn pair_workload(case: &str) -> Vec<(String, String)> {
    let input: Value = serde_json::from_str(include_str!("../issue-307/inputs.json")).unwrap();
    let rows = input["reranker"].as_array().unwrap();
    let row = |i: usize| {
        (
            rows[i]["query"].as_str().unwrap().to_owned(),
            rows[i]["text"]
                .as_str()
                .unwrap()
                .repeat(usize::try_from(rows[i]["repeat"].as_u64().unwrap()).unwrap()),
        )
    };
    match case {
        "single" => vec![row(0)],
        "multiple" => (0..32)
            .map(|i| {
                let mut p = row(i % 4);
                if i % 2 == 1 {
                    p.0 = "コードの例".into();
                }
                p
            })
            .collect(),
        "common" => {
            let query = row(0).0;
            (0..256).map(|i| (query.clone(), row(i % 4).1)).collect()
        }
        "boundary" => vec![row(5), (String::new(), String::new()), row(4), row(0)],
        _ => panic!("unknown pair workload"),
    }
}
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> Value {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f64;
    let mut min_cos = 1.0f64;
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.len(), b.len());
        assert!(!a.is_empty());
        assert!(a.iter().chain(b).all(|x| x.is_finite()));
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let aa: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let bb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        assert!(aa > 0. && bb > 0.);
        min_cos = min_cos.min(dot / (aa * bb).sqrt());
        for (&x, &y) in a.iter().zip(b) {
            max_abs = max_abs.max((f64::from(x) - f64::from(y)).abs());
        }
    }
    // Issue #307 same-token diagnostic, not a new product tolerance.
    assert!(
        min_cos >= 0.99999 && max_abs <= 1e-5,
        "cos={min_cos}, abs={max_abs}"
    );
    json!({"max_abs":max_abs,"min_cosine":min_cos})
}

pub(crate) fn observe_queries(
    tokenizer: &Tokenizer,
    refs: &[&str],
    candidate: bool,
    dims: usize,
    mut run: impl FnMut(bool, &[&str]) -> Result<Queries, EmbedError>,
    memory: (&mut dyn FnMut(), &mut dyn FnMut() -> usize),
    mut record: impl FnMut(&Value),
) {
    let (reset_peak, peak) = memory;
    let expected: Vec<_> = refs
        .iter()
        .map(|text| {
            let tokens = embed::tokenize_with_prefix(tokenizer, text, embed::QUERY_PREFIX).unwrap();
            embed::truncate_for_query(tokens.input_ids, tokens.attention_mask, embed::MAX_SEQ_LEN).0
        })
        .collect();
    assert_eq!(query317::prepare(tokenizer, refs).unwrap(), expected);
    assert!(run(candidate, &[]).unwrap().values.is_empty());
    let mut outputs = Vec::new();
    for trial in 0..4 {
        if !candidate {
            query317::begin_singles();
        }
        reset_peak();
        let start = Instant::now();
        let Queries { values, metrics } = run(candidate, refs).unwrap();
        let wall_ns = start.elapsed().as_nanos();
        let peak = peak();
        assert_eq!(values.len(), refs.len());
        assert!(values.iter().all(|v| v.len() == dims));
        let singles = query317::take_singles();
        let (forwards, tokenize_ns) = if let Some(m) = &metrics {
            assert!(singles.is_empty());
            (
                serde_json::to_value(&m.forwards).unwrap(),
                m.tokenize.unwrap().as_nanos(),
            )
        } else {
            assert_eq!(singles.len(), refs.len());
            (
                json!(singles.iter().map(|(_, shape)| shape).collect::<Vec<_>>()),
                singles.iter().map(|(t, _)| t.as_nanos()).sum(),
            )
        };
        record(
            &json!({"event":"call","trial":trial,"cold":trial==0,"values":values,"wall_ns":wall_ns,"tokenize_ns":tokenize_ns,"metrics":metrics,"forwards":forwards,"metal_peak_bytes":peak}),
        );
        outputs.push(values);
    }
    for values in outputs.iter().skip(1) {
        compare(&outputs[0], values);
    }
    let oracle = run(!candidate, refs).unwrap();
    record(&json!({"event":"comparison","diagnostic":compare(&outputs[0],&oracle.values)}));
}

pub(crate) fn observe_pairs(
    tokenizer: &Tokenizer,
    pairs: &[(&str, &str)],
    candidate: bool,
    mut run: impl FnMut(&[(&str, &str)], bool) -> Result<(Vec<f32>, Value), RerankerError>,
    mut reset_peak: impl FnMut(),
    mut peak: impl FnMut() -> usize,
    mut record: impl FnMut(&Value),
) -> Vec<f32> {
    {
        let candidate_encodings = pairs317::prepare(tokenizer, pairs, true).unwrap();
        let baseline_encodings = pairs317::prepare(tokenizer, pairs, false).unwrap();
        assert_eq!(candidate_encodings.len(), pairs.len());
        assert_eq!(baseline_encodings.len(), pairs.len());
        for (actual, expected) in candidate_encodings.iter().zip(&baseline_encodings) {
            pairs317::assert_encoding(actual, expected);
        }
    }
    assert!(run(&[], candidate).unwrap().0.is_empty());
    let mut first = None;
    for trial in 0..4 {
        reset_peak();
        let (scores, metrics) = run(pairs, candidate).unwrap();
        assert_eq!(scores.len(), pairs.len());
        assert!(scores.iter().all(|x| x.is_finite()));
        if let Some(expected) = &first {
            assert_eq!(&scores, expected);
        } else {
            first = Some(scores.clone());
        }
        record(
            &json!({"event":"call","trial":trial,"cold":trial==0,"scores":scores,"metrics":metrics,"metal_peak_bytes":peak()}),
        );
    }
    first.unwrap()
}

pub(crate) fn compare_public(
    case: &str,
    pairs: &[(&str, &str)],
    original: &[f32],
    score: impl FnOnce(&[(&str, &str)]) -> Result<Vec<f32>, RerankerError>,
    rerank: impl FnOnce(&str, &[&str]) -> Result<Vec<RankedResult>, RerankerError>,
) -> Value {
    assert_eq!(original.len(), pairs.len());
    let mut order: Vec<_> = (0..original.len()).collect();
    order.sort_by(|&a, &b| original[b].total_cmp(&original[a]).then(a.cmp(&b)));
    if case == "common" || case == "single" {
        let docs: Vec<_> = pairs.iter().map(|(_, d)| *d).collect();
        let ranked = rerank(pairs[0].0, &docs).unwrap();
        assert_eq!(ranked.len(), original.len());
        assert_eq!(ranked.iter().map(|r| r.index).collect::<Vec<_>>(), order);
        for result in ranked {
            assert_eq!(result.score, original[result.index]);
        }
    } else {
        assert_eq!(score(pairs).unwrap(), original);
    }
    json!({"event":"comparison","exact_scores":true,"order":order})
}

fn dispatch(
    domain: &str,
    case: &str,
    mode: &str,
    record: impl FnMut(&Value),
    query: impl FnOnce(&str, bool, &mut dyn FnMut(&Value)),
    pair: impl FnOnce(&str, bool, &mut dyn FnMut(&Value)),
) {
    let candidate = match mode {
        "baseline" => false,
        "candidate" => true,
        _ => panic!("invalid mode"),
    };
    let mut record = record;
    record(&json!({"event":"context","domain":domain,"case":case,"mode":mode}));
    match domain {
        "query" => query(case, candidate, &mut record),
        "pair" => pair(case, candidate, &mut record),
        _ => panic!("invalid domain"),
    }
}

#[test]
#[ignore = "Issue #317 fixed models on unsandboxed Apple Silicon; explicit runner"]
fn observe_query317() {
    require_unsandboxed_mlx_runtime();
    let path = env::var("RURICO_317_OUTPUT").expect("explicit output required");
    dispatch(
        &env::var("RURICO_317_DOMAIN").unwrap(),
        &env::var("RURICO_317_CASE").unwrap(),
        &env::var("RURICO_317_MODE").unwrap(),
        |value| record_at(Path::new(&path), value),
        embedder::observe317,
        mlx::observe317,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{ChunkedEmbedding, ForwardShape, InferenceMetrics};
    use std::{
        cell::Cell,
        fs::read_to_string,
        panic::{AssertUnwindSafe, catch_unwind},
        time::Duration,
    };
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    fn tokenizer() -> Tokenizer {
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(
                    [
                        ("[UNK]".into(), 0),
                        ("検索クエリ".into(), 3),
                        (":".into(), 4),
                        ("猫".into(), 5),
                    ]
                    .into_iter()
                    .collect(),
                )
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    struct Identifiable;
    impl Embed for Identifiable {
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            // Different input values; these are protocol fixtures, not model outputs.
            Ok(vec![
                text.len() as f32 + 1.,
                text.bytes().map(u64::from).sum::<u64>() as f32 + 1.,
            ])
        }
        fn embed_document(&self, _: &str) -> Result<ChunkedEmbedding, EmbedError> {
            panic!("wrong entry")
        }
        fn embed_documents_batch(&self, _: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            panic!("wrong entry")
        }
        fn embed_text(&self, _: &str, _: &str) -> Result<Vec<f32>, EmbedError> {
            panic!("wrong entry")
        }
    }

    #[test]
    fn measurement_protocol_records_trials_shapes_and_real_workload_contracts() {
        let tokenizer = tokenizer();
        let model = Identifiable;
        let legacy = Legacy(&model);
        for case in ["single", "multiple", "boundary"] {
            let owned = query_workload(case);
            let refs: Vec<_> = owned.iter().map(String::as_str).collect();
            let expected = serde_json::to_value(legacy.queries(&refs).unwrap().values).unwrap();
            for candidate in [false, true] {
                let resets = Cell::new(0);
                let mut calls = Vec::new();
                let mut rows = Vec::new();
                observe_queries(
                    &tokenizer,
                    &refs,
                    candidate,
                    2,
                    |batch, inputs| {
                        calls.push((batch, inputs.len()));
                        let mut result = legacy.queries(inputs)?;
                        if batch {
                            result.metrics = Some(InferenceMetrics {
                                tokenize: Some(Duration::from_nanos(11)),
                                forwards: vec![ForwardShape {
                                    batch_size: inputs.len(),
                                    sequence_length: 128,
                                }],
                                ..InferenceMetrics::default()
                            });
                        } else {
                            for _ in inputs {
                                query317::observe_single(Duration::from_nanos(7), 128);
                            }
                        }
                        Ok(result)
                    },
                    (&mut || resets.set(resets.get() + 1), &mut || {
                        assert!(resets.get() > 0);
                        321
                    }),
                    |row| rows.push(row.clone()),
                );
                assert_eq!(resets.get(), 4);
                assert_eq!(
                    calls,
                    [
                        (candidate, 0),
                        (candidate, refs.len()),
                        (candidate, refs.len()),
                        (candidate, refs.len()),
                        (candidate, refs.len()),
                        (!candidate, refs.len())
                    ]
                );
                assert_eq!(rows.len(), 5);
                for (trial, row) in rows[..4].iter().enumerate() {
                    assert_eq!(row["trial"], trial);
                    assert_eq!(row["cold"], trial == 0);
                    assert_eq!(row["metal_peak_bytes"], 321);
                    assert_eq!(
                        row["forwards"].as_array().unwrap().len(),
                        if candidate { 1 } else { refs.len() }
                    );
                    assert_eq!(
                        row["tokenize_ns"],
                        if candidate { 11 } else { 7 * refs.len() }
                    );
                    assert_eq!(row["values"], expected);
                }
                assert_eq!(rows[4]["event"], "comparison");
            }
        }
        for case in ["single", "multiple", "common", "boundary"] {
            let owned = pair_workload(case);
            let pairs: Vec<_> = owned
                .iter()
                .map(|(q, d)| (q.as_str(), d.as_str()))
                .collect();
            if case == "common" {
                assert_eq!(pairs.len(), 256);
                assert!(pairs.iter().all(|p| p.0 == pairs[0].0));
            }
            let expected_scores: Vec<f32> = pairs
                .iter()
                .map(|(q, d)| (q.len() + d.len()) as f32)
                .collect();
            let expected_record = json!(expected_scores);
            for candidate in [false, true] {
                let mut rows = Vec::new();
                let mut calls = Vec::new();
                let resets = Cell::new(0);
                let scores = observe_pairs(
                    &tokenizer,
                    &pairs,
                    candidate,
                    |inputs, reuse| {
                        assert_eq!(inputs, if calls.is_empty() { &[][..] } else { &pairs });
                        calls.push((reuse, inputs.len()));
                        Ok((
                            if inputs.is_empty() {
                                Vec::new()
                            } else {
                                expected_scores.clone()
                            },
                            json!({"source":"fixture"}),
                        ))
                    },
                    || resets.set(resets.get() + 1),
                    || {
                        assert!(resets.get() > 0);
                        123
                    },
                    |row| rows.push(row.clone()),
                );
                assert_eq!(scores, expected_scores);
                assert_eq!(resets.get(), 4);
                assert_eq!(
                    calls,
                    [
                        (candidate, 0),
                        (candidate, pairs.len()),
                        (candidate, pairs.len()),
                        (candidate, pairs.len()),
                        (candidate, pairs.len())
                    ]
                );
                assert_eq!(rows.len(), 4);
                for (trial, row) in rows.iter().enumerate() {
                    assert_eq!(row["trial"], trial);
                    assert_eq!(row["cold"], trial == 0);
                    assert_eq!(row["scores"], expected_record);
                    assert_eq!(row["metal_peak_bytes"], 123);
                    assert_eq!(row["metrics"]["source"], "fixture");
                }
            }
        }
    }

    #[test]
    fn public_comparison_uses_one_entry_and_rejects_wrong_scores_or_order() {
        let pairs = [("q", "first"), ("q", "second"), ("q", "third")];
        let scores = [0.2, 0.7, 0.7];
        for case in ["single", "common", "multiple", "boundary"] {
            let calls = Cell::new(0);
            let ranked = || {
                vec![
                    RankedResult {
                        index: 1,
                        score: 0.7,
                    },
                    RankedResult {
                        index: 2,
                        score: 0.7,
                    },
                    RankedResult {
                        index: 0,
                        score: 0.2,
                    },
                ]
            };
            let result = compare_public(
                case,
                &pairs,
                &scores,
                |inputs| {
                    assert!(case == "multiple" || case == "boundary");
                    assert_eq!(inputs, pairs);
                    calls.set(calls.get() + 1);
                    Ok(scores.to_vec())
                },
                |query, docs| {
                    assert!(case == "single" || case == "common");
                    assert_eq!(query, "q");
                    assert_eq!(docs, ["first", "second", "third"]);
                    calls.set(calls.get() + 1);
                    Ok(ranked())
                },
            );
            assert_eq!(calls.get(), 1);
            assert_eq!(result["order"], json!([1, 2, 0]));
        }
        for bad in [
            vec![
                RankedResult {
                    index: 1,
                    score: 0.2,
                },
                RankedResult {
                    index: 2,
                    score: 0.7,
                },
                RankedResult {
                    index: 0,
                    score: 0.2,
                },
            ],
            vec![
                RankedResult {
                    index: 2,
                    score: 0.7,
                },
                RankedResult {
                    index: 1,
                    score: 0.7,
                },
                RankedResult {
                    index: 0,
                    score: 0.2,
                },
            ],
        ] {
            assert!(
                catch_unwind(|| compare_public(
                    "common",
                    &pairs,
                    &scores,
                    |_| panic!("extra scoring"),
                    |_, _| Ok(bad)
                ))
                .is_err()
            );
        }
        assert!(
            catch_unwind(|| compare(&[vec![1., 2.], vec![3., 4.]], &[vec![3., 4.], vec![1., 2.]]))
                .is_err()
        );
        assert!(catch_unwind(|| compare(&[vec![1.]], &[vec![f32::NAN]])).is_err());
    }

    #[test]
    fn dispatch_and_append_preserve_context_and_reject_invalid_modes() {
        let file = tempfile::NamedTempFile::new().unwrap();
        for domain in ["query", "pair"] {
            for mode in ["baseline", "candidate"] {
                dispatch(
                    domain,
                    "single",
                    mode,
                    |value| record_at(file.path(), value),
                    |case, batch, record| {
                        assert_eq!(domain, "query");
                        assert_eq!(case, "single");
                        assert_eq!(batch, mode == "candidate");
                        record(&json!({"event":"query"}));
                    },
                    |case, batch, record| {
                        assert_eq!(domain, "pair");
                        assert_eq!(case, "single");
                        assert_eq!(batch, mode == "candidate");
                        record(&json!({"event":"pair"}));
                    },
                );
            }
        }
        let rows: Vec<Value> = read_to_string(file.path())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 8);
        for pair in rows.as_chunks::<2>().0 {
            assert_eq!(pair[0]["event"], "context");
            assert_eq!(pair[0]["domain"], pair[1]["event"]);
        }
        for (domain, mode, expected) in [
            ("other", "baseline", "invalid domain"),
            ("query", "other", "invalid mode"),
        ] {
            let calls = Cell::new(0);
            let rejection = catch_unwind(AssertUnwindSafe(|| {
                dispatch(
                    domain,
                    "single",
                    mode,
                    |_| {},
                    |_, _, _| calls.set(calls.get() + 1),
                    |_, _, _| calls.set(calls.get() + 1),
                );
            }));
            assert_eq!(calls.get(), 0);
            let panic = rejection.expect_err("dispatch must reject invalid input");
            assert_eq!(panic.downcast_ref::<&str>(), Some(&expected));
        }
    }
}
