//! #318: isolated experiments over existing tokenization/forward boundaries.
use crate::research::runtime318::Observation;
use crate::sandbox::require_unsandboxed_mlx_runtime;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, mpsc};
use std::time::{Duration, Instant};
use std::{env, thread};

use mlx_rs::memory;
use serde_json::json;

use crate::model_io::truncate_with_eos;
use crate::model_io::{BUCKET_BOUNDS, MAX_SEQ_LEN, ModelArtifact};
use crate::reranker::mlx::RerankerInner;
use crate::reranker::{Artifacts, RerankerModelId, cached_artifacts};

use crate::runtime318_planner::{
    Limits, PreparedBatch, Rejected, Row, collect_complete, query_turn, stopped,
};

fn limits(rows: usize) -> Limits {
    Limits {
        rows,
        padded_tokens: rows * 128,
        bytes: rows * 4096,
    }
}

fn prepare(tokenizer: &tokenizers::Tokenizer, doc: usize) -> Row {
    let encoding = tokenizer.encode(("query", "doc"), true).unwrap();
    let mut ids = encoding.get_ids().to_vec();
    let mut mask = encoding.get_attention_mask().to_vec();
    truncate_with_eos(&mut ids, &mut mask, MAX_SEQ_LEN);
    Row {
        doc,
        chunk: 0,
        ids: ids.into_boxed_slice(),
        mask: mask.into_boxed_slice(),
    }
}

fn record(value: &serde_json::Value) {
    eprintln!("runtime318 {value}");
}

#[derive(Clone, Copy)]
enum Stop {
    Complete,
    AfterFirst,
    Expired,
    DuringForward,
}

fn forward(
    inner: &mut RerankerInner,
    batch: &PreparedBatch,
    deadline: Option<Instant>,
) -> Vec<(usize, usize, f32)> {
    // Existing forward takes Vec rows, so adapter copies are measured and
    // explicitly outside the prepared-payload cap.
    let ids: Vec<_> = batch.rows.iter().map(|r| r.ids.to_vec()).collect();
    let masks: Vec<_> = batch.rows.iter().map(|r| r.mask.to_vec()).collect();
    let start = Instant::now();
    if let Some(deadline) = deadline {
        assert!(
            start < deadline,
            "measurement deadline expired before forward"
        );
    }
    let values = inner.forward_sub_batch(&ids, &masks, batch.bucket).unwrap();
    let end = Instant::now();
    if let Some(deadline) = deadline {
        assert!(
            end >= deadline,
            "forward did not cross measurement deadline"
        );
        record(&json!({"event":"deadline_forward", "rows":batch.rows.len(),
            "remaining_at_start_ns":deadline.duration_since(start).as_nanos(),
            "overrun_at_end_ns":end.duration_since(deadline).as_nanos(),
            "forward_wall_ns":end.duration_since(start).as_nanos()}));
    }
    record(
        &json!({"event":"forward", "rows":batch.rows.len(), "bucket":batch.bucket,
        "ids_dtype":"u32", "output_dtype":"f32", "prepared_bytes":batch.bytes(),
        "wall_ns":start.elapsed().as_nanos(), "active_bytes":memory::active_memory().unwrap(),
        "cache_bytes":memory::cache_memory().unwrap(), "peak_bytes":memory::peak_memory().unwrap()}),
    );
    assert_eq!(values.len(), batch.rows.len());
    batch
        .rows
        .iter()
        .zip(values)
        .map(|(r, score)| (r.doc, r.chunk, score))
        .collect()
}

fn stream_forward(
    inner: &mut RerankerInner,
    batch: &PreparedBatch,
    stop: Stop,
    deadline: &mut Option<Instant>,
) -> Vec<(usize, usize, f32)> {
    let measured_deadline = if matches!(stop, Stop::DuringForward) {
        // Experiment only: arm once after preparation, never extend on later forwards.
        Some(*deadline.get_or_insert_with(|| Instant::now() + Duration::from_millis(10)))
    } else {
        None
    };
    forward(inner, batch, measured_deadline)
}

fn stream(
    inner: &mut RerankerInner,
    count: usize,
    cap: usize,
    overlap: bool,
    stop: Stop,
    mut sink: impl FnMut(Vec<(usize, usize, f32)>),
) -> usize {
    let cancel = AtomicBool::new(false);
    let mut deadline = matches!(stop, Stop::Expired).then(Instant::now);
    let tokenizer = inner.tokenizer.clone();
    let begin = Instant::now();
    let mut completed = 0;
    if overlap {
        thread::scope(|scope| {
            // One queued batch, one producer batch, one consumer batch.
            let (tx, rx) = mpsc::sync_channel(1);
            let cancel_ref = &cancel;
            // DuringForward arms in the consumer; the producer remains bounded
            // by the queue and is released by dropping rx at the deadline boundary.
            let producer_deadline = deadline;
            let producer = scope.spawn(move || {
                for start in (0..count).step_by(cap) {
                    if stopped(cancel_ref.load(Ordering::Relaxed), producer_deadline) { break; }
                    let preparing = Instant::now();
                    let mut batch = PreparedBatch::new(limits(cap));
                    for doc in start..(start + cap).min(count) {
                        batch.push(prepare(&tokenizer, doc)).unwrap();
                    }
                    let prepared = Instant::now();
                    let sent = tx.send(batch);
                    let delivered = Instant::now();
                    record(&json!({"event":"producer", "start_ns":preparing.duration_since(begin).as_nanos(),
                        "prepared_ns":prepared.duration_since(begin).as_nanos(),
                        "delivered_ns":delivered.duration_since(begin).as_nanos(), "send_ok":sent.is_ok()}));
                    if sent.is_err() { break; }
                }
            });
            for batch in &rx {
                if stopped(cancel.load(Ordering::Relaxed), deadline) {
                    break;
                }
                let start = Instant::now();
                let rows = stream_forward(inner, &batch, stop, &mut deadline);
                record(
                    &json!({"event":"consumer", "start_ns":start.duration_since(begin).as_nanos(),
                    "end_ns":begin.elapsed().as_nanos()}),
                );
                completed += rows.len();
                sink(rows);
                if matches!(stop, Stop::AfterFirst) {
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
            }
            // Drop before joining: release a producer blocked by backpressure.
            drop(rx);
            producer.join().unwrap();
        });
    } else {
        for start in (0..count).step_by(cap) {
            if stopped(cancel.load(Ordering::Relaxed), deadline) {
                break;
            }
            let mut batch = PreparedBatch::new(limits(cap));
            for doc in start..(start + cap).min(count) {
                batch.push(prepare(&tokenizer, doc)).unwrap();
            }
            let rows = stream_forward(inner, &batch, stop, &mut deadline);
            completed += rows.len();
            sink(rows);
            if matches!(stop, Stop::AfterFirst) {
                cancel.store(true, Ordering::Relaxed);
            }
        }
    }
    record(
        &json!({"event":"stream", "overlap":overlap, "requested":count,
        "completed":completed, "cancelled":cancel.load(Ordering::Relaxed),
        "wall_ns":begin.elapsed().as_nanos()}),
    );
    completed
}

fn compare(inner: &mut RerankerInner, cap: usize, overlap: bool, baseline: &[f32]) {
    let mut received = 0;
    let mut max_error = 0.0_f32;
    let completed = stream(
        inner,
        baseline.len(),
        cap,
        overlap,
        Stop::Complete,
        |rows| {
            for (doc, chunk, score) in rows {
                assert_eq!((doc, chunk), (received, 0));
                assert!(score.is_finite());
                max_error = max_error.max((score - baseline[received]).abs());
                received += 1;
            }
        },
    );
    assert_eq!(completed, baseline.len());
    assert_eq!(received, baseline.len());
    // #307 same-model diagnostic; not a newly adopted product quality tolerance.
    assert!(max_error <= 2.5e-5, "score difference: {max_error}");
    record(&json!({"event":"comparison", "count":received, "max_abs_score_difference":max_error}));
}

#[test]
#[ignore = "host-only #318 resource/overlap experiments; fixed cached models required"]
fn observe_runtime() {
    require_unsandboxed_mlx_runtime();
    let case = env::var("RURICO_318_CASE").expect("choose the documented experiment case");
    assert_eq!(BUCKET_BOUNDS, [128, 512, 2048, 8192]);
    // Embedding-only experiments must not prepare or retain a reranker tokenizer.
    if case == "fairness" {
        fairness();
        return;
    }
    if case == "ownership" {
        ownership();
        return;
    }
    let model = RerankerModelId::default();
    let artifacts = cached_artifacts(model)
        .unwrap()
        .expect("cache fixed reranker first; no download");
    record(
        &json!({"event":"context", "case":case, "model":model.repo_id(),
        "revision":ModelArtifact::revision(model), "hidden":artifacts.config.hidden_size,
        "intermediate":artifacts.config.intermediate_size}),
    );
    if case == "retention" {
        // Real tokenizer without loading GPU weights; measure entire input
        // storage independently of forward size. No 5000-pair GPU call.
        for count in [50, 500, 5000] {
            let mut ids = Vec::with_capacity(count);
            let mut masks = Vec::with_capacity(count);
            for doc in 0..count {
                let row = prepare(&artifacts.tokenizer, doc);
                ids.push(row.ids.into_vec());
                masks.push(row.mask.into_vec());
            }
            let direct = (ids.capacity() + masks.capacity()) * size_of::<Vec<u32>>()
                + ids
                    .iter()
                    .chain(&masks)
                    .map(|r| r.capacity() * 4)
                    .sum::<usize>();
            drop(ids);
            drop(masks);
            let mut batch = PreparedBatch::new(limits(10));
            let mut peak = 0;
            let mut completed = 0;
            for doc in 0..count {
                if let Err(Rejected::Full(row)) = batch.push(prepare(&artifacts.tokenizer, doc)) {
                    completed += batch.rows.len();
                    batch = PreparedBatch::new(limits(10));
                    batch.push(row).unwrap();
                }
                peak = peak.max(batch.bytes());
            }
            completed += batch.rows.len();
            assert_eq!(completed, count);
            record(
                &json!({"event":"retention", "inputs":count, "direct_retained_bytes":direct,
                "prepared_peak_bytes":peak, "row_cap":10, "scope":"requested payload, excludes tokenizer scratch/RSS/results"}),
            );
        }
        return;
    }
    if case == "mixed" {
        mixed_ownership(&artifacts);
        return;
    }
    let mut inner = RerankerInner::new(&artifacts).unwrap();
    if case == "shapes" {
        for cap in [1, 10, 25, 50] {
            let _observation = Observation::begin();
            let mut batch = PreparedBatch::new(limits(cap));
            for doc in 0..cap {
                batch.push(prepare(&inner.tokenizer, doc)).unwrap();
            }
            let rows = forward(&mut inner, &batch, None);
            assert_eq!(rows.len(), cap);
            record(&json!({"event":"shape_case", "rows":cap, "bucket":128}));
        }
        return;
    }
    // One public-equivalent direct call, at measurable 50-pair scale.
    let baseline = inner.score_batch(&vec![("query", "doc"); 50]).unwrap();
    for trial in 0..3 {
        if case == "forward" || case == "overlap" {
            memory::reset_peak_memory().unwrap();
            let start = Instant::now();
            let values = inner.score_batch(&vec![("query", "doc"); 50]).unwrap();
            assert_eq!(values.len(), baseline.len());
            assert!(
                values
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() <= 2.5e-5)
            );
            record(&json!({"event":"direct", "trial":trial, "rows":50,
                "wall_ns":start.elapsed().as_nanos(), "peak_bytes":memory::peak_memory().unwrap(),
                "active_bytes":memory::active_memory().unwrap(), "cache_bytes":memory::cache_memory().unwrap()}));
        }
        match case.as_str() {
            "deadline" => {
                for overlap in [false, true] {
                    record(&json!({"event":"trial", "trial":trial, "overlap":overlap}));
                    let mut partial = Vec::new();
                    let completed =
                        stream(&mut inner, 50, 10, overlap, Stop::DuringForward, |rows| {
                            partial.extend(rows)
                        });
                    assert_eq!(completed, 10);
                    assert_eq!(partial.len(), 10);
                    for (doc, &(id, chunk, score)) in partial.iter().enumerate() {
                        assert_eq!((id, chunk), (doc, 0));
                        assert!(score.is_finite() && (score - baseline[doc]).abs() <= 2.5e-5);
                    }
                    let rejected = collect_complete([Ok(partial)], 50);
                    assert_eq!(rejected, Err("incomplete stream"));
                    record(
                        &json!({"event":"deadline_result", "trial":trial, "overlap":overlap,
                        "completed":completed, "complete_vec_rejected":true,
                        "measurement_deadline_ms":10}),
                    );
                }
            }
            "forward" => {
                for cap in [1, 10, 25, 50] {
                    memory::reset_peak_memory().unwrap();
                    record(&json!({"event":"trial", "trial":trial, "cap":cap}));
                    compare(&mut inner, cap, false, &baseline);
                }
            }
            "overlap" => {
                for overlap in if trial % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    memory::reset_peak_memory().unwrap();
                    record(&json!({"event":"trial", "trial":trial, "overlap":overlap}));
                    compare(&mut inner, 10, overlap, &baseline);
                }
            }
            "control" => {
                let expired = stream(&mut inner, 50, 10, true, Stop::Expired, |_| {
                    panic!("expired stream emitted output")
                });
                assert_eq!(expired, 0);
                record(&json!({"event":"deadline", "completed":0, "expired_before_forward":true}));
                let mut partial = Vec::new();
                let completed = stream(&mut inner, 50, 10, true, Stop::AfterFirst, |rows| {
                    partial.extend(rows)
                });
                assert_eq!(completed, 10);
                assert_eq!(partial.len(), 10);
                assert_eq!(
                    collect_complete([Ok(partial)], 50),
                    Err("incomplete stream")
                );
            }
            _ => panic!("unknown experiment: {case}"),
        }
    }
}

fn mixed_ownership(rerank_artifacts: &Artifacts) {
    use crate::embed::{Embed, EmbedOptions, Embedder, ModelId, cached_artifacts as cached_embed};
    use crate::reranker::Reranker;
    let embed_artifacts = cached_embed(ModelId::DEFAULT)
        .unwrap()
        .expect("cache embed 310m first");
    let embedder = Embedder::new(&embed_artifacts).unwrap();
    let reranker = Reranker::new(rerank_artifacts).unwrap();
    let options = EmbedOptions {
        token_budget: Some(128),
        ..Default::default()
    };
    for trial in 0..3 {
        for parallel in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            memory::reset_peak_memory().unwrap();
            let begin = Instant::now();
            let embed = || {
                let measured = embedder
                    .embed_documents_batch_with_options_and_metrics(
                        &["東京は日本の都市です。"],
                        &options,
                    )
                    .unwrap();
                let metrics = measured.metrics.unwrap();
                record(
                    &json!({"event":"mixed", "model":"embed", "trial":trial, "parallel":parallel,
                    "wall_ns":metrics.wall.as_nanos(), "lock_wait_ns":metrics.lock_wait.as_nanos()}),
                );
            };
            let rerank = || {
                let start = Instant::now();
                let scores = reranker.score_batch(&[("query", "doc")]).unwrap();
                assert_eq!(scores.len(), 1);
                record(
                    &json!({"event":"mixed", "model":"reranker", "trial":trial, "parallel":parallel,
                    "wall_ns":start.elapsed().as_nanos(), "lock_wait_ns":null}),
                );
            };
            if parallel {
                let barrier = Barrier::new(2);
                thread::scope(|scope| {
                    scope.spawn(|| {
                        barrier.wait();
                        embed();
                    });
                    scope.spawn(|| {
                        barrier.wait();
                        rerank();
                    });
                });
            } else {
                embed();
                rerank();
            }
            record(
                &json!({"event":"mixed_peak", "trial":trial, "parallel":parallel,
                "wall_ns":begin.elapsed().as_nanos(), "active_bytes":memory::active_memory().unwrap(),
                "peak_bytes":memory::peak_memory().unwrap()}),
            );
        }
    }
}

fn fairness() {
    use crate::embed::{Embed, EmbedOptions, Embedder, ModelId, cached_artifacts};
    let model = ModelId::DEFAULT;
    let artifacts = cached_artifacts(model)
        .unwrap()
        .expect("cache embed 310m first");
    record(
        &json!({"event":"context", "case":"fairness", "model":model.repo_id(),
        "revision":ModelArtifact::revision(model), "hidden":artifacts.config.hidden_size,
        "intermediate":artifacts.config.intermediate_size}),
    );
    let embedder = Embedder::new(&artifacts).unwrap();
    let options = EmbedOptions {
        token_budget: Some(128),
        ..Default::default()
    };
    for trial in 0..3 {
        for priority in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let enqueued = Instant::now();
            let (mut queries, mut bulk, mut burst) = (12, 4, 0);
            while queries + bulk > 0 {
                let query = if priority {
                    query_turn(queries > 0, bulk > 0, burst)
                } else {
                    bulk == 0
                };
                let start = Instant::now();
                if query {
                    embedder.embed_query("東京の人口").unwrap();
                    queries -= 1;
                    burst += 1;
                } else {
                    embedder
                        .embed_documents_batch_with_options(
                            &["東京は日本の都市です。"; 2],
                            &options,
                        )
                        .unwrap();
                    bulk -= 1;
                    burst = 0;
                }
                record(
                    &json!({"event":"fairness", "trial":trial, "priority":priority, "query":query,
                    "queue_wait_ns":start.duration_since(enqueued).as_nanos(), "wall_ns":start.elapsed().as_nanos(),
                    "remaining_queries":queries, "remaining_bulk":bulk}),
                );
            }
        }
    }
}

fn ownership() {
    use crate::embed::{Embed, EmbedOptions, Embedder, ModelId, cached_artifacts};
    let model = ModelId::DEFAULT;
    let artifacts = cached_artifacts(model)
        .unwrap()
        .expect("cache fixed embed 310m first");
    record(
        &json!({"event":"context", "case":"ownership", "model":model.repo_id(),
        "revision":ModelArtifact::revision(model), "hidden":artifacts.config.hidden_size,
        "intermediate":artifacts.config.intermediate_size}),
    );
    let a = Embedder::new(&artifacts).unwrap();
    let b = Embedder::new(&artifacts).unwrap();
    let options = EmbedOptions {
        token_budget: Some(128),
        ..Default::default()
    };
    for trial in 0..3 {
        for shared in if trial % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            memory::reset_peak_memory().unwrap();
            let barrier = Barrier::new(2);
            thread::scope(|scope| {
                for (worker, model) in [(0, &a), (1, if shared { &a } else { &b })] {
                    let barrier = &barrier;
                    let options = &options;
                    scope.spawn(move || {
                        barrier.wait();
                        for turn in 0..4 {
                            let docs: &[&str] = if worker == 0 { &["query"] } else { &["doc", "doc"] };
                            let measured = model.embed_documents_batch_with_options_and_metrics(docs, options).unwrap();
                            assert_eq!(measured.embeddings.len(), docs.len());
                            let metrics = measured.metrics.unwrap();
                            record(&json!({"event":"ownership", "trial":trial, "shared":shared,
                                "worker":worker, "turn":turn, "lock_wait_ns":metrics.lock_wait.as_nanos(),
                                "wall_ns":metrics.wall.as_nanos(), "forwards":metrics.forwards}));
                        }
                    });
                }
            });
            record(
                &json!({"event":"ownership_peak", "shared":shared, "trial":trial,
                "active_bytes":memory::active_memory().unwrap(), "peak_bytes":memory::peak_memory().unwrap()}),
            );
        }
    }
}
