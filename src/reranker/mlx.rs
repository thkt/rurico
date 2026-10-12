#[cfg(test)]
use crate::mlx_cache::testing::{Stage, checkpoint};

use super::processing::{dispatch_sub_batch_size, scores_from_logits, truncate_pair};
use super::{Artifacts, ModelInitError, RerankerError};
use crate::mlx_cache::{Component, clear_inference_cache, run_inference};
use crate::model_io::{MAX_SEQ_LEN, pad_sequences};
use crate::modernbert::{
    Config, ModernBert, biasless_layer_norm, layer_norm_eps_f32,
    weights::{self, WeightKind},
};

use mlx_rs::{
    builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn,
    ops::indexing::IndexOp,
};

use std::path::Path;

type EncodedPairs = (Vec<Vec<u32>>, Vec<Vec<u32>>);

pub(crate) struct RerankerInner {
    model: RerankerModel,
    pub(crate) tokenizer: tokenizers::Tokenizer,
}

impl RerankerInner {
    pub(crate) fn new(artifacts: &Artifacts) -> Result<Self, ModelInitError> {
        let config = &artifacts.config;
        let tokenizer = artifacts.tokenizer.clone();

        let model =
            RerankerModel::load(&artifacts.paths.model, config).map_err(ModelInitError::backend)?;

        tracing::info!(
            hidden_size = config.hidden_size,
            model_path = ?artifacts.paths.model,
            "reranker: model loaded"
        );
        Ok(Self { model, tokenizer })
    }

    /// Retains all pair tokens and scores. The shared [`crate::model_io::TOKEN_BUDGET`]
    /// splits forward passes; it does not bound total CPU or GPU memory.
    pub(crate) fn score_batch(
        &mut self,
        pairs: &[(&str, &str)],
    ) -> Result<Vec<f32>, RerankerError> {
        let (all_ids, all_masks) = self.encode_pairs(pairs)?;

        let raw_max = all_ids.iter().map(Vec::len).max().unwrap_or(0);
        let total_pairs = pairs.len();
        let (bucket_len, sub_batch_size) = dispatch_sub_batch_size(raw_max, total_pairs);

        let mut all_scores = Vec::with_capacity(total_pairs);
        for (ids_chunk, masks_chunk) in all_ids
            .chunks(sub_batch_size)
            .zip(all_masks.chunks(sub_batch_size))
        {
            let sub_scores = self.forward_sub_batch(ids_chunk, masks_chunk, bucket_len)?;
            all_scores.extend(sub_scores);
        }
        Ok(all_scores)
    }

    fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<EncodedPairs, RerankerError> {
        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());
        let mut all_masks: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());

        for (pair_idx, &(query, doc)) in pairs.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode((query, doc), true)
                .map_err(RerankerError::tokenizer)?;
            let mut ids = encoding.get_ids().to_vec();
            let mut mask = encoding.get_attention_mask().to_vec();
            truncate_pair(&mut ids, &mut mask, MAX_SEQ_LEN, pair_idx);
            all_ids.push(ids);
            all_masks.push(mask);
        }

        Ok((all_ids, all_masks))
    }

    pub(crate) fn forward_sub_batch(
        &mut self,
        ids_chunk: &[Vec<u32>],
        masks_chunk: &[Vec<u32>],
        bucket_len: usize,
    ) -> Result<Vec<f32>, RerankerError> {
        self.forward_sub_batch_with(ids_chunk, masks_chunk, bucket_len, scores_from_logits)
    }

    fn forward_sub_batch_with(
        &mut self,
        ids_chunk: &[Vec<u32>],
        masks_chunk: &[Vec<u32>],
        bucket_len: usize,
        readback: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>, RerankerError>,
    ) -> Result<Vec<f32>, RerankerError> {
        let (flat_ids, flat_mask, batch_size, max_len) =
            pad_sequences(ids_chunk, Some(masks_chunk), Some(bucket_len));

        let batch_size_i32 = i32::try_from(batch_size).expect("batch_size fits in i32");
        let max_len_i32 = i32::try_from(max_len).expect("max_len fits in i32");
        run_inference(
            || {
                self.model
                    .forward(&flat_ids, &flat_mask, batch_size_i32, max_len_i32)
                    .map_err(RerankerError::inference)
            },
            |output| {
                output.eval().map_err(RerankerError::inference)?;
                #[cfg(test)]
                checkpoint(Stage::Eval).map_err(RerankerError::inference)?;
                let flat: &[f32] = output.as_slice();
                #[cfg(test)]
                checkpoint(Stage::Readback).map_err(RerankerError::inference)?;
                readback(flat, batch_size, bucket_len)
            },
            || clear_inference_cache(Component::Reranker),
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct PredictionHead {
    #[param]
    dense: nn::Linear,
    #[param]
    norm: nn::LayerNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
struct RerankerModel {
    #[param]
    model: ModernBert,
    #[param]
    head: PredictionHead,
    #[param]
    classifier: nn::Linear,
}

impl RerankerModel {
    fn new(config: &Config) -> Result<Self, Exception> {
        let model = ModernBert::new(config)?;
        let h = i32::try_from(config.hidden_size).expect("validated hidden_size");
        let eps = layer_norm_eps_f32(config);

        let dense = nn::LinearBuilder::new(h, h).bias(false).build()?;
        let norm = biasless_layer_norm(h, eps)?;
        let classifier = nn::LinearBuilder::new(h, 1).build()?;

        Ok(Self {
            model,
            head: PredictionHead { dense, norm },
            classifier,
        })
    }

    fn load(path: &Path, config: &Config) -> Result<Self, Exception> {
        weights::load(path, config, WeightKind::Reranker, Self::new)
    }

    fn forward(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[u32],
        batch_size: i32,
        seq_len: i32,
    ) -> Result<mlx_rs::Array, Exception> {
        let hidden = self
            .model
            .forward(input_ids, attention_mask, batch_size, seq_len)?;

        // CLS pooling: [batch, seq, hidden] -> [seq, batch, hidden] -> index(0) -> [batch, hidden]
        let cls = hidden.transpose_axes(&[1, 0, 2])?.index(0);

        #[cfg(test)]
        checkpoint(Stage::Pool)?;
        let x = self.head.dense.forward(&cls)?;
        let x = nn::gelu(&x)?;
        let x = self.head.norm.forward(&x)?;

        self.classifier.forward(&x)
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "smoke"))]
pub(super) mod research;

#[cfg(all(test, feature = "smoke"))]
mod bucket_probe {
    //! Issue #309 host measurement. Compiled only for tests + smoke.
    use std::{env, time::Instant};

    use mlx_rs::memory;
    use serde_json::{Value, json};

    use super::RerankerInner;
    use crate::model_io::{ModelArtifact, compute_sub_batch_size};
    use crate::reranker::bucket_spike::{DispatchError, dispatch};
    use crate::reranker::processing::{dispatch_sub_batch_size, sigmoid, validate_logits};
    use crate::reranker::{RerankerError, RerankerModelId, cached_artifacts, sort_results};
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    impl RerankerInner {
        fn observe_length_batch(
            &mut self,
            pairs: &[(&str, &str)],
            candidate: bool,
        ) -> Result<Value, RerankerError> {
            let start = Instant::now();
            let (ids, masks) = self.encode_pairs(pairs)?;
            let encode_ns = start.elapsed().as_nanos();
            let lengths: Vec<_> = ids.iter().map(Vec::len).collect();
            let mut forwards = Vec::new();
            let raw = |flat: &[f32], rows, bucket| {
                // Reuse production shape/finite validation before returning CPU data.
                validate_logits(flat, rows, bucket)?;
                Ok(flat.to_vec())
            };
            let logits = if candidate {
                dispatch(&lengths, |indices, bucket| {
                    // Account for gather copies in candidate wall and RSS. No claim
                    // that length grouping also bounds retained input tokens.
                    let chunk_ids: Vec<_> = indices.iter().map(|&i| ids[i].clone()).collect();
                    let chunk_masks: Vec<_> = indices.iter().map(|&i| masks[i].clone()).collect();
                    let forward_start = Instant::now();
                    let values = self.forward_sub_batch_with(&chunk_ids, &chunk_masks, bucket, raw)?;
                    let host_forward_ns = forward_start.elapsed().as_nanos();
                    forwards.push(json!({"indices":indices,"rows":indices.len(),"bucket":bucket,"host_forward_ns":host_forward_ns}));
                    Ok(values)
                })
                .map_err(|error| match error {
                    DispatchError::Forward(error) => error,
                    DispatchError::Shape { expected, actual } => RerankerError::inference_message(
                        format!("candidate: expected {expected} rows, got {actual}"),
                    ),
                })?
            } else {
                let (bucket, size) = dispatch_sub_batch_size(
                    lengths.iter().copied().max().unwrap_or(0),
                    pairs.len(),
                );
                let mut values = Vec::with_capacity(pairs.len());
                for (part, (chunk_ids, chunk_masks)) in
                    ids.chunks(size).zip(masks.chunks(size)).enumerate()
                {
                    let forward_start = Instant::now();
                    values.extend(self.forward_sub_batch_with(
                        chunk_ids,
                        chunk_masks,
                        bucket,
                        raw,
                    )?);
                    let host_forward_ns = forward_start.elapsed().as_nanos();
                    let indices: Vec<_> = (part * size..part * size + chunk_ids.len()).collect();
                    forwards
                        .push(json!({"indices":indices,"rows":chunk_ids.len(),"bucket":bucket,"host_forward_ns":host_forward_ns}));
                }
                values
            };
            let scores: Vec<_> = logits.iter().copied().map(sigmoid).collect();
            let wall_ns = start.elapsed().as_nanos();
            assert_eq!(scores.len(), pairs.len());
            let peak_bytes = memory::peak_memory().unwrap();
            let real_tokens: usize = lengths.iter().sum();
            let padded_tokens: usize = forwards
                .iter()
                .map(|f| {
                    usize::try_from(f["rows"].as_u64().unwrap() * f["bucket"].as_u64().unwrap())
                        .unwrap()
                })
                .sum();
            for forward in &forwards {
                assert!(
                    forward["rows"].as_u64().unwrap()
                        <= u64::try_from(compute_sub_batch_size(
                            usize::try_from(forward["bucket"].as_u64().unwrap()).unwrap(),
                            None,
                        ))
                        .unwrap()
                );
            }
            let order: Vec<_> = sort_results(&scores).iter().map(|r| r.index).collect();
            Ok(
                json!({"event":"call","candidate":candidate,"lengths":lengths,
            "logits":logits,"scores":scores,"order":order,"forwards":forwards,
            "real_tokens":real_tokens,"padded_tokens":padded_tokens,
            "padding_ratio":if real_tokens == 0 { 1.0 } else { padded_tokens as f64 / real_tokens as f64 },
            "encode_ns":encode_ns,"wall_ns":wall_ns,"mutex_wait_ns":null,"gpu_kernel_ns":null,"metal_peak_bytes":peak_bytes}),
            )
        }
    }

    fn workload(name: &str, tokenizer: &tokenizers::Tokenizer) -> Vec<(String, String)> {
        let input: Value =
            serde_json::from_str(include_str!("../../docs/research/issue-307/inputs.json"))
                .unwrap();
        let rows = input["reranker"].as_array().unwrap();
        let pair = |i: usize| {
            let row = &rows[i];
            (
                row["query"].as_str().unwrap().to_owned(),
                row["text"]
                    .as_str()
                    .unwrap()
                    .repeat(usize::try_from(row["repeat"].as_u64().unwrap()).unwrap()),
            )
        };
        match name {
            "same" => (0..50).map(|_| pair(0)).collect(),
            "mixed" => (0..50)
                .map(|i| pair(if i % 7 == 0 { 4 } else { i % 4 }))
                .collect(),
            "empty" => Vec::new(),
            "boundary128" | "boundary512" | "boundary1024" | "boundary2048" | "boundary4096"
            | "limit" => {
                let targets: &[usize] = match name {
                    "boundary128" => &[127, 128, 129],
                    "boundary512" => &[511, 512, 513],
                    "boundary1024" => &[1023, 1024, 1025],
                    "boundary2048" => &[2047, 2048, 2049],
                    "boundary4096" => &[4095, 4096, 4097],
                    _ => &[8191, 8192, 8193],
                };
                targets
                    .iter()
                    .copied()
                    .map(|target| {
                        let query = "猫について";
                        // Choose strings by actual pair encoding, not an assumed token
                        // count for repeated text. Both sides then encode these strings.
                        for n in target / 2 - 16..=target / 2 + 1 {
                            let text = "猫 ".repeat(n);
                            for doc in [text.as_str(), text.trim_end()] {
                                if tokenizer.encode((query, doc), true).unwrap().len() == target {
                                    return (query.to_owned(), doc.to_owned());
                                }
                            }
                        }
                        panic!("cannot construct pair boundary {target}");
                    })
                    .collect()
            }
            _ => panic!("unknown Issue #309 workload"),
        }
    }

    #[test]
    #[ignore = "Issue #309: cached fixed reranker, unsandboxed Metal, guarded host runner"]
    fn observe_length_batches() {
        require_unsandboxed_mlx_runtime();
        let mode = env::var("RURICO_309_MODE").unwrap();
        assert!(matches!(mode.as_str(), "reference" | "candidate"));
        let name = env::var("RURICO_309_WORKLOAD").unwrap();
        let id = RerankerModelId::default();
        let artifacts = cached_artifacts(id)
            .unwrap()
            .expect("fixed reranker cache required");
        let owned = workload(&name, &artifacts.tokenizer);
        let pairs: Vec<_> = owned
            .iter()
            .map(|(q, d)| (q.as_str(), d.as_str()))
            .collect();
        eprintln!(
            "bucket309 {}",
            json!({"event":"input","workload":name,"pairs":owned,
        "build_flags":env!("RURICO_BUILD_FLAGS"),
        "model":id.repo_id(),"revision":ModelArtifact::revision(id),"dtype":"F32","model_files":[artifacts.paths.model, artifacts.paths.config, artifacts.paths.tokenizer]})
        );
        let mut inner = RerankerInner::new(&artifacts).unwrap();
        let mut last_reference = None;
        let repeats =
            env::var("RURICO_308_REPEATS").map_or(4, |value| value.parse::<usize>().unwrap());
        assert!(repeats >= 4);
        for repeat in 0..repeats {
            memory::reset_peak_memory().unwrap();
            let mut row = inner
                .observe_length_batch(&pairs, mode == "candidate")
                .unwrap();
            row["repeat"] = json!(repeat);
            row["state"] = json!(if repeat == 0 { "cold" } else { "warm" });
            eprintln!("bucket309 {row}");
            if mode == "reference" && repeat == repeats - 1 {
                last_reference = Some(row);
            }
        }
        // Internal production score_batch parity, outside timed records.
        // This does not exercise the public Reranker wrapper or its mutex/empty-input handling.
        let observed =
            last_reference.unwrap_or_else(|| inner.observe_length_batch(&pairs, false).unwrap());
        let public = inner.score_batch(&pairs).unwrap();
        let reference: Vec<f32> = serde_json::from_value(observed["scores"].clone()).unwrap();
        assert_eq!(public.len(), reference.len());
        assert!(
            public
                .iter()
                .zip(reference)
                .all(|(a, b)| (a - b).abs() <= 2.5e-5)
        );
        assert_eq!(
            sort_results(&public)
                .iter()
                .map(|r| r.index)
                .collect::<Vec<_>>(),
            serde_json::from_value::<Vec<usize>>(observed["order"].clone()).unwrap()
        );
    }
}
