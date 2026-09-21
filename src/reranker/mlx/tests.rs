use super::*;

#[test]
fn load_rejects_incomplete_weights_before_mlx_allocation() {
    weights::tests::assert_loader_rejects(WeightKind::Reranker, |path, config| {
        RerankerModel::load(path, config).map(|_| ())
    });
}

#[cfg(feature = "test-mlx")]
mod runtime {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use mlx_rs::{
        Array, memory,
        module::{ModuleParameters as _, ModuleParametersExt},
        random,
    };
    use serial_test::serial;

    use super::*;
    use crate::model_io::ModelArtifact;
    use crate::reranker::{RerankerModelId, cached_artifacts, processing::sigmoid};
    use crate::sandbox::require_unsandboxed_mlx_runtime;

    // Chosen before GPU comparison. Both logits and sigmoid scores must satisfy
    // abs(a-b) <= 1e-6 + 1e-6 * abs(a); rankings must match exactly.
    const ATOL: f32 = 1e-6;
    const RTOL: f32 = 1e-6;
    const PAIRS: [(&str, &str); 4] = [
        ("東京の人口", "東京は日本の都市です。"),
        ("東京の人口", "京都も日本の都市です。"),
        ("東京の人口", "東京都の人口統計を調べます。"),
        ("東京の人口", "猫は窓辺で眠っています。"),
    ];

    fn rank(logits: &[f32]) -> Vec<usize> {
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        order
    }

    fn assert_close(reference: &[f32], actual: &[f32]) {
        assert_eq!(reference.len(), actual.len());
        for (&a, &b) in reference.iter().zip(actual) {
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() <= ATOL + RTOL * a.abs(), "{a} != {b}");
        }
        assert_eq!(rank(reference), rank(actual));
    }

    // Reconstruct the old head, preserving its initialization order. Backbone
    // LayerNorm biases in 1a53b0d were zero and layer-zero attn_norm was unused.
    // This isolates the removed random bias; it is not an official HF oracle.
    fn legacy_head_load(path: &Path, config: &Config) -> RerankerModel {
        let h = i32::try_from(config.hidden_size).unwrap();
        let model = ModernBert::new(config).unwrap();
        let dense = nn::LinearBuilder::new(h, h).build().unwrap();
        let norm = nn::LayerNormBuilder::new(h)
            .eps(layer_norm_eps_f32(config))
            .build()
            .unwrap();
        let classifier = nn::LinearBuilder::new(h, 1).build().unwrap();
        let mut model = RerankerModel {
            model,
            head: PredictionHead { dense, norm },
            classifier,
        };
        model.load_safetensors(path).unwrap();
        model
    }

    /// Host only; prints fixed revisions, inputs, pre/post logits/scores/ranks,
    /// load wall time and Metal peak bytes. Run alone with --nocapture.
    #[test]
    #[ignore = "requires cached official reranker and unsandboxed MLX"]
    #[serial]
    fn official_reranker_reload_contract() {
        require_unsandboxed_mlx_runtime();
        let id = RerankerModelId::default();
        assert_eq!(
            ModelArtifact::revision(id),
            "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"
        );
        let artifacts = cached_artifacts(id)
            .unwrap()
            .expect("cache fixed reranker revision");
        let config = &artifacts.config;
        let mut ids = Vec::new();
        let mut masks = Vec::new();
        for pair in PAIRS {
            let encoding = artifacts.tokenizer.encode(pair, true).unwrap();
            ids.push(encoding.get_ids().to_vec());
            masks.push(encoding.get_attention_mask().to_vec());
        }
        let (ids, masks, batch, seq) = pad_sequences(&ids, Some(&masks), Some(128));
        let batch = i32::try_from(batch).unwrap();
        let seq = i32::try_from(seq).unwrap();
        eprintln!(
            "reranker contract: model={} model/tokenizer_revision={} dtype=F32 pairs={PAIRS:?} batch={batch} seq={seq} atol={ATOL} rtol={RTOL} rank=exact",
            id.repo_id(),
            ModelArtifact::revision(id)
        );
        let mut reference: Option<Vec<f32>> = None;
        // Repeat seed 42 after another initialization; alternate load order
        // so the added validation cost is not always measured second.
        for (trial, seed) in [42, 7, 42].into_iter().enumerate() {
            let mut results = BTreeMap::new();
            let modes = if trial == 1 {
                ["checked", "unchecked", "legacy-head"]
            } else {
                ["legacy-head", "unchecked", "checked"]
            };
            for mode in modes {
                memory::clear_cache().unwrap();
                random::seed(seed).unwrap();
                memory::reset_peak_memory().unwrap();
                let start = Instant::now();
                let mut model = match mode {
                    "legacy-head" => legacy_head_load(&artifacts.paths.model, config),
                    "unchecked" => {
                        let mut model = RerankerModel::new(config).unwrap();
                        model.load_safetensors(&artifacts.paths.model).unwrap();
                        model
                    }
                    _ => RerankerModel::load(&artifacts.paths.model, config).unwrap(),
                };
                let load_ms = start.elapsed().as_secs_f64() * 1000.0;
                let peak = memory::peak_memory().unwrap();
                if mode != "legacy-head" {
                    let actual: BTreeMap<String, Vec<usize>> = model
                        .parameters()
                        .flatten()
                        .iter()
                        .map(|(key, value)| {
                            (
                                key.to_string(),
                                value
                                    .shape()
                                    .iter()
                                    .map(|&d| usize::try_from(d).unwrap())
                                    .collect(),
                            )
                        })
                        .collect();
                    assert_eq!(
                        actual,
                        weights::expected_shapes(config, WeightKind::Reranker)
                    );
                    assert!(model.head.dense.bias.is_none());
                    assert!(model.head.norm.bias.is_none());
                    assert!(model.classifier.bias.is_some());
                }
                let output = model.forward(&ids, &masks, batch, seq).unwrap();
                output.eval().unwrap();
                let logits = output.as_slice::<f32>().to_vec();
                assert_eq!(logits.len(), PAIRS.len());
                assert!(logits.iter().all(|x| x.is_finite()));
                let scores: Vec<_> = logits.iter().copied().map(sigmoid).collect();
                eprintln!(
                    "trial={trial} seed={seed} mode={mode} load_ms={load_ms:.3} metal_load_peak_bytes={peak} logits={logits:?} scores={scores:?} rank={:?}",
                    rank(&scores)
                );
                drop(output);
                if mode == "legacy-head" {
                    // Anchor the reconstructed legacy head to the actual
                    // 1a53b0d public-API run supplied by host verification.
                    let before: Vec<serde_json::Value> = serde_json::from_str(include_str!(
                        "../../../docs/benchmarks/issue-300-before-output.json"
                    ))
                    .unwrap();
                    let recorded = before
                        .iter()
                        .find(|sample| {
                            sample["result"]["kind"] == "reranker"
                                && sample["result"]["seed"].as_u64() == Some(seed)
                        })
                        .expect("recorded prechange reranker seed");
                    let recorded_scores: Vec<f32> =
                        serde_json::from_value(recorded["result"]["scores"].clone()).unwrap();
                    assert_close(&recorded_scores, &scores);
                }
                if mode == "checked" {
                    if let Some(reference) = &reference {
                        assert_close(reference, &logits);
                        assert_close(
                            &reference.iter().copied().map(sigmoid).collect::<Vec<_>>(),
                            &scores,
                        );
                    } else {
                        reference = Some(logits.clone());
                    }
                    // Final classifier bias must contribute to logits, not
                    // merely appear in the header or parameter inventory.
                    let bias = model
                        .classifier
                        .bias
                        .value
                        .as_ref()
                        .unwrap()
                        .add(Array::from(0.25_f32))
                        .unwrap();
                    *model.classifier.bias = Some(bias);
                    let shifted = model.forward(&ids, &masks, batch, seq).unwrap();
                    shifted.eval().unwrap();
                    assert_close(
                        &logits.iter().map(|v| v + 0.25).collect::<Vec<_>>(),
                        shifted.as_slice(),
                    );
                }
                results.insert(mode, logits);
                drop(model);
            }
            assert_close(&results["checked"], &results["unchecked"]);
            let differences: Vec<_> = results["checked"]
                .iter()
                .zip(&results["legacy-head"])
                .map(|(new, old)| new - old)
                .collect();
            let score_differences: Vec<_> = results["checked"]
                .iter()
                .zip(&results["legacy-head"])
                .map(|(&new, &old)| sigmoid(new) - sigmoid(old))
                .collect();
            eprintln!(
                "trial={trial} new_minus_legacy_logits={differences:?} new_minus_legacy_scores={score_differences:?} legacy_rank={:?} new_rank={:?}",
                rank(&results["legacy-head"]),
                rank(&results["checked"])
            );
        }
    }
}
