use serde_json::json;

use super::RerankerModel;
use crate::mlx_cache::{Component, clear_inference_cache};
use crate::model_io::MAX_SEQ_LEN;
use crate::reranker::processing::{sigmoid, truncate_pair};
use crate::reranker::{Reranker, RerankerModelId, cached_artifacts};
use crate::research::{Input, batches, save, shape, verify_paths};

pub(crate) fn capture(inputs: &[Input]) {
    let artifacts = cached_artifacts(RerankerModelId::RuriV3Reranker310m)
        .unwrap()
        .expect("fixed reranker cache required");
    verify_paths(&artifacts.paths);
    let mut tokens = Vec::new();
    let mut rows = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let text = input.text();
        let raw = artifacts
            .tokenizer
            .encode((input.query.as_str(), text.as_str()), true)
            .unwrap();
        let mut ids = raw.get_ids().to_vec();
        let mut mask = raw.get_attention_mask().to_vec();
        truncate_pair(&mut ids, &mut mask, MAX_SEQ_LEN, i);
        tokens.push(
            json!({"id":input.id,"raw_ids":raw.get_ids(),"raw_mask":raw.get_attention_mask(),
            "model_ids":ids,"model_mask":mask}),
        );
        rows.push((format!("{}/text", input.id), ids, mask));
    }
    let mut observations = batches(&rows);
    for batch in &mut observations {
        let mut model = RerankerModel::load(&artifacts.paths.model, &artifacts.config).unwrap();
        let (ids, mask, b, s) = shape(batch);
        let logits = model.forward(&ids, &mask, b, s).unwrap();
        logits.eval().unwrap();
        batch.values = logits
            .as_slice::<f32>()
            .iter()
            .map(|&v| vec![v, sigmoid(v)])
            .collect();
        drop(logits);
        drop(model);
        clear_inference_cache(Component::Reranker);
    }
    let reranker = Reranker::new(&artifacts).unwrap();
    // Match the mixed diagnostic batch; long pairs are checked as singletons.
    let short: Vec<_> = inputs
        .iter()
        .take(5)
        .map(|x| (x.query.as_str(), x.text()))
        .collect();
    let pairs: Vec<_> = short.iter().map(|(q, d)| (*q, d.as_str())).collect();
    let scores = reranker.score_batch(&pairs).unwrap();
    let singleton: Vec<_> = inputs
        .iter()
        .map(|x| json!({"id":x.id,"score":reranker.score(&x.query, &x.text()).unwrap()}))
        .collect();
    let documents: Vec<_> = short.iter().take(4).map(|(_, d)| d.as_str()).collect();
    let ranks: Vec<_> = reranker
        .rerank(&inputs[0].query, &documents)
        .unwrap()
        .iter()
        .map(|r| json!({"index":r.index,"score":r.score}))
        .collect();
    save(
        "reranker",
        &tokens,
        &observations,
        &json!({"singleton":singleton,"mixed_scores":scores,"ranking":ranks}),
    );
}
