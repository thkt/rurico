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

pub(crate) fn capture(inputs: &[Input]) {
    let artifacts = cached_artifacts(ModelId::RuriV3_310m)
        .unwrap()
        .expect("fixed embed cache required");
    verify_paths(&artifacts.paths);
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
    let text_values: Vec<_> = inputs
        .iter()
        .map(|input| {
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
    let measured = embedder
        .embed_documents_batch_with_options_and_metrics(&refs, &EmbedOptions::default())
        .unwrap();
    let documents = document_records(&docs, &measured.embeddings);
    let metrics = measured.metrics.unwrap();
    let shapes: Vec<_> = metrics
        .forwards
        .iter()
        .map(|x| json!({"batch_size":x.batch_size,"seq_len":x.sequence_length}))
        .collect();
    save(
        "embedding",
        &tokens,
        &observations,
        &json!({"text":text_values,"documents":documents,"forwards":shapes}),
    );
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
