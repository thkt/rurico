use std::panic::catch_unwind;

use mlx_rs::{Array, ops::indexing::IndexOp};
use serde_json::{Value, json};

use super::processing::plan_document_chunks;
use super::{
    ChunkedEmbedding, DOCUMENT_PREFIX, Embed, EmbedOptions, Embedder, MAX_SEQ_LEN, ModelId,
    cached_artifacts, extract_prefix_tokens, gpu_pool_and_normalize, max_content,
    tokenize_with_prefix, truncate_for_query,
};
use crate::mlx_cache::{Component, clear_inference_cache};
use crate::modernbert::ModernBert;
use crate::research::{Input, batches, save, shape, verify_paths};

pub(crate) fn capture(inputs: &[Input], model: ModelId, dimensions: Option<usize>) {
    let artifacts = cached_artifacts(model)
        .unwrap()
        .expect("fixed embed cache required");
    verify_paths(&artifacts.paths);
    let _ = verified_dimensions(artifacts.config.hidden_size, dimensions);
    let tokenizer = &artifacts.tokenizer;
    let prefix = extract_prefix_tokens(tokenizer, DOCUMENT_PREFIX).unwrap();
    let mut tokens = Vec::new();
    let mut rows = Vec::new();
    for input in inputs {
        let text = input.text();
        let raw = tokenize_with_prefix(tokenizer, &text, &input.prefix).unwrap();
        let (ids, mask, _) = truncate_for_query(
            raw.input_ids.clone(),
            raw.attention_mask.clone(),
            MAX_SEQ_LEN,
        );
        rows.push((format!("{}/text", input.id), ids.clone(), mask.clone()));
        let chunks = if input.prefix == DOCUMENT_PREFIX {
            plan_document_chunks(tokenizer, &[&text], &prefix, max_content(prefix.len()))
                .unwrap()
                .0
        } else {
            Vec::new()
        };
        // Short documents already have the same model tokens as /text.
        if chunks.len() > 1 {
            for (i, chunk) in chunks.iter().enumerate() {
                rows.push((
                    format!("{}/chunk-{i}", input.id),
                    chunk.clone(),
                    vec![1; chunk.len()],
                ));
            }
        }
        tokens.push(
            json!({"id":input.id,"raw_ids":raw.input_ids,"raw_mask":raw.attention_mask,
            "model_ids":ids,"model_mask":mask,"chunks":chunks}),
        );
    }
    let mut observations = batches(&rows);
    // Drop each model after each shape: exact-length research must not grow the
    // production model's four-bucket mask cache across unbounded shapes.
    for batch in &mut observations {
        let mut model = ModernBert::load(&artifacts.paths.model, &artifacts.config).unwrap();
        let (ids, mask, b, s) = shape(batch);
        let hidden = model.forward(&ids, &mask, b, s).unwrap();
        let probes: Vec<_> = batch
            .mask
            .iter()
            .enumerate()
            .map(|(row, mask)| {
                let last = mask.iter().rposition(|&x| x == 1).unwrap();
                [0, last / 2, last]
                    .iter()
                    .map(|&pos| {
                        let sample = hidden
                            .index((i32::try_from(row).unwrap(), i32::try_from(pos).unwrap()));
                        sample.eval().unwrap();
                        sample.as_slice::<f32>().to_vec()
                    })
                    .collect()
            })
            .collect();
        let pooled = gpu_pool_and_normalize(hidden, &Array::from_slice(&mask, &[b, s])).unwrap();
        pooled.eval().unwrap();
        batch.values = pooled
            .as_slice::<f32>()
            .chunks(artifacts.config.hidden_size)
            .map(<[f32]>::to_vec)
            .collect();
        batch.hidden_probes = probes;
        drop(pooled);
        drop(model);
        clear_inference_cache(Component::Embed);
    }
    let embedder = Embedder::new(&artifacts).unwrap();
    let dimensions = verified_dimensions(embedder.embedding_dims(), dimensions);
    let public = observe_public(&embedder, inputs, &tokens, model, dimensions);
    save("embedding", &tokens, &observations, &public);
}

fn verified_dimensions(actual: usize, expected: Option<usize>) -> Option<(usize, usize)> {
    expected.map(|expected| {
        assert_eq!(actual, expected, "fixed model dimension");
        (actual, expected)
    })
}

// Uses the same public calls as the concrete Embedder; model-free providers can
// exercise record selection and delegation without replacing the model oracle.
fn observe_public(
    embedder: &dyn Embed,
    inputs: &[Input],
    tokens: &[Value],
    model: ModelId,
    dimensions: Option<(usize, usize)>,
) -> Value {
    let text_values: Vec<_> = inputs
        .iter()
        .map(|input| {
            eprintln!("research embedding public text {}", input.id);
            json!({"id":input.id,
        "value":embedder.embed_text(&input.text(), &input.prefix).unwrap()})
        })
        .collect();
    let docs: Vec<_> = inputs
        .iter()
        .filter(|input| input.prefix == DOCUMENT_PREFIX)
        .collect();
    let texts: Vec<_> = docs.iter().map(|input| input.text()).collect();
    let refs: Vec<_> = texts.iter().map(String::as_str).collect();
    eprintln!("research embedding public documents batch start");
    let options = EmbedOptions {
        token_budget: dimensions.map(|_| 8192),
        ..EmbedOptions::default()
    };
    let measured = embedder
        .embed_documents_batch_with_options_and_metrics(&refs, &options)
        .unwrap();
    let documents = document_records(&docs, &measured.embeddings);
    eprintln!("research embedding public documents batch complete");
    let metrics = measured.metrics.unwrap();
    let shapes: Vec<_> = metrics
        .forwards
        .iter()
        .map(|x| json!({"batch_size":x.batch_size,"seq_len":x.sequence_length}))
        .collect();
    let mut public = json!({"text":text_values,"documents":documents,"forwards":shapes});
    if let Some((concrete_dimensions, dimensions)) = dimensions {
        let short_docs: Vec<_> = docs
            .iter()
            .copied()
            .filter(|input| {
                let record = tokens
                    .iter()
                    .find(|record| record["id"] == input.id)
                    .unwrap();
                record["chunks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|chunk| chunk.as_array().unwrap().len() <= 512)
            })
            .collect();
        let short_texts: Vec<_> = short_docs.iter().map(|input| input.text()).collect();
        let short_refs: Vec<_> = short_texts.iter().map(String::as_str).collect();
        eprintln!("research embedding public default short documents batch start");
        let short_outputs = embedder.embed_documents_batch(&short_refs).unwrap();
        public["default_short_documents"] = json!(document_records(&short_docs, &short_outputs));
        public["document_token_budget"] = json!(options.token_budget);
        let singles: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                eprintln!("research embedding public singleton {}", docs[index].id);
                embedder.embed_document(text).unwrap()
            })
            .collect();
        public["singleton_documents"] = json!(document_records(&docs, &singles));
        let queries: Vec<_> = inputs
            .iter()
            .filter(|input| input.prefix == super::QUERY_PREFIX)
            .map(|input| {
                eprintln!("research embedding public query {}", input.id);
                json!({"id":input.id,"value":embedder.embed_query(&input.text()).unwrap()})
            })
            .collect();
        assert!(embedder.embed_documents_batch(&[]).unwrap().is_empty());
        public["queries"] = json!(queries);
        public["empty_batch"] = json!([]);
        public["model"] = json!({"repository":model.repo_id(),"revision":model.revision(),
            "concrete_dimensions":concrete_dimensions,"expected_dimensions":dimensions});
    }
    public
}

fn document_records(docs: &[&Input], embeddings: &[ChunkedEmbedding]) -> Vec<Value> {
    assert_eq!(embeddings.len(), docs.len(), "public document count");
    docs.iter()
        .zip(embeddings)
        .map(|(input, out)| json!({"id":input.id,"chunks":out.chunks()}))
        .collect()
}

#[test]
fn document_records_reject_count_mismatch_before_zip() {
    let inputs: Vec<Input> = serde_json::from_value(json!([
        {"id":"first", "text":"a", "repeat":1},
        {"id":"second", "text":"b", "repeat":1}
    ]))
    .unwrap();
    let docs: Vec<_> = inputs.iter().collect();
    let mut outputs = vec![
        ChunkedEmbedding::try_new(vec![vec![1.0, 0.0]]).unwrap(),
        ChunkedEmbedding::try_new(vec![vec![0.0, 1.0]]).unwrap(),
    ];
    assert_eq!(
        document_records(&docs, &outputs),
        vec![
            json!({"id":"first", "chunks":[[1.0,0.0]]}),
            json!({"id":"second", "chunks":[[0.0,1.0]]})
        ]
    );
    // An invalid extra output must not disappear at the serialization boundary.
    outputs.push(ChunkedEmbedding::try_new(vec![vec![f32::NAN]]).unwrap());
    for count in [1, 3] {
        let error = catch_unwind(|| document_records(&docs, &outputs[..count]))
            .expect_err("missing or excess documents must be rejected before recording");
        let message = error.downcast_ref::<String>().unwrap();
        assert!(message.contains("public document count"), "{message}");
    }
}

#[test]
fn public_observation_preserves_calls_and_records() {
    use super::{EmbedError, ForwardShape, InferenceMetrics, MeasuredEmbedding, QUERY_PREFIX};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Provider(Mutex<Vec<Value>>);
    impl Provider {
        fn chunks(text: &str) -> ChunkedEmbedding {
            let chunks = match text {
                "aa" => vec![vec![1.0], vec![2.0]],
                "b" => vec![vec![3.0]],
                _ => vec![vec![4.0]],
            };
            ChunkedEmbedding::try_new(chunks).unwrap()
        }
        fn record(&self, call: Value) {
            self.0.lock().unwrap().push(call);
        }
    }
    impl Embed for Provider {
        fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
            self.record(json!(["text", text, prefix]));
            Ok(vec![10.0])
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            self.record(json!(["query", text]));
            Ok(vec![20.0])
        }
        fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
            self.record(json!(["single", text]));
            Ok(Self::chunks(text))
        }
        fn embed_documents_batch(
            &self,
            texts: &[&str],
        ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            self.record(json!(["default", texts]));
            Ok(texts.iter().map(|text| Self::chunks(text)).collect())
        }
        fn embed_documents_batch_with_options_and_metrics(
            &self,
            texts: &[&str],
            options: &EmbedOptions,
        ) -> Result<MeasuredEmbedding, EmbedError> {
            self.record(json!(["measured", texts, options.token_budget]));
            assert_eq!(options.forward_pause, None);
            Ok(MeasuredEmbedding {
                embeddings: texts.iter().map(|text| Self::chunks(text)).collect(),
                metrics: Some(InferenceMetrics {
                    forwards: vec![
                        ForwardShape {
                            batch_size: 1,
                            sequence_length: 8192,
                        },
                        ForwardShape {
                            batch_size: 2,
                            sequence_length: 512,
                        },
                    ],
                    ..InferenceMetrics::default()
                }),
            })
        }
    }
    let inputs: Vec<Input> = serde_json::from_value(json!([
        {"id":"long", "text":"a", "prefix":DOCUMENT_PREFIX, "repeat":2},
        {"id":"query", "text":"q", "prefix":QUERY_PREFIX, "repeat":2},
        {"id":"boundary", "text":"b", "prefix":DOCUMENT_PREFIX, "repeat":1},
        {"id":"empty", "text":"", "prefix":DOCUMENT_PREFIX, "repeat":1},
        {"id":"plain", "text":"c", "prefix":"", "repeat":1}
    ]))
    .unwrap();
    // Token records need not be in input order. Every chunk must fit the short
    // lane: a short first chunk alone cannot admit a later 513-token chunk.
    let tokens = vec![
        json!({"id":"empty", "chunks":[[1,2]]}),
        json!({"id":"boundary", "chunks":[vec![1;512]]}),
        json!({"id":"long", "chunks":[[1],vec![1;513]]}),
    ];
    let provider = Provider::default();
    let observed = observe_public(
        &provider,
        &inputs,
        &tokens,
        ModelId::RuriV3_30m,
        Some((256, 256)),
    );
    let documents = json!([
        {"id":"long", "chunks":[[1.0],[2.0]]},
        {"id":"boundary", "chunks":[[3.0]]},
        {"id":"empty", "chunks":[[4.0]]}
    ]);
    let legacy = json!({
        "text":[
            {"id":"long","value":[10.0]}, {"id":"query","value":[10.0]},
            {"id":"boundary","value":[10.0]}, {"id":"empty","value":[10.0]},
            {"id":"plain","value":[10.0]}
        ],
        "documents":documents,
        "forwards":[{"batch_size":1,"seq_len":8192},{"batch_size":2,"seq_len":512}]
    });
    let mut expected = legacy.clone();
    expected["singleton_documents"] = documents;
    expected["default_short_documents"] = json!([
        {"id":"boundary","chunks":[[3.0]]}, {"id":"empty","chunks":[[4.0]]}
    ]);
    expected["document_token_budget"] = json!(8192);
    expected["queries"] = json!([{"id":"query","value":[20.0]}]);
    expected["empty_batch"] = json!([]);
    expected["model"] = json!({"repository":"cl-nagoya/ruri-v3-30m",
        "revision":"24899e5de370b56d179604a007c0d727bf144504",
        "concrete_dimensions":256,"expected_dimensions":256});
    assert_eq!(observed, expected);
    let initial_calls = vec![
        json!(["text", "aa", DOCUMENT_PREFIX]),
        json!(["text", "qq", QUERY_PREFIX]),
        json!(["text", "b", DOCUMENT_PREFIX]),
        json!(["text", "", DOCUMENT_PREFIX]),
        json!(["text", "c", ""]),
    ];
    let mut expected_calls = initial_calls.clone();
    expected_calls.extend([
        json!(["measured", ["aa", "b", ""], 8192]),
        json!(["default", ["b", ""]]),
        json!(["single", "aa"]),
        json!(["single", "b"]),
        json!(["single", ""]),
        json!(["query", "qq"]),
        json!(["default", []]),
    ]);
    assert_eq!(*provider.0.lock().unwrap(), expected_calls);

    let provider = Provider::default();
    assert_eq!(
        observe_public(&provider, &inputs, &[], ModelId::RuriV3_310m, None),
        legacy
    );
    let mut expected_calls = initial_calls;
    expected_calls.push(json!(["measured", ["aa", "b", ""], null]));
    assert_eq!(*provider.0.lock().unwrap(), expected_calls);
}

#[test]
fn fixed_dimensions_reject_mislabeled_configuration() {
    assert_eq!(verified_dimensions(256, Some(256)), Some((256, 256)));
    assert_eq!(verified_dimensions(768, None), None);
    let error = catch_unwind(|| verified_dimensions(768, Some(256))).unwrap_err();
    assert!(
        error
            .downcast_ref::<String>()
            .unwrap()
            .contains("fixed model dimension")
    );
}
