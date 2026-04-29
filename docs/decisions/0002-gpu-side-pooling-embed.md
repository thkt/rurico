# ADR 0002: GPU-side Pooling for the embed Pipeline

- Status: Proposed
- Date: 2026-04-24
- Confidence: medium. The mlx-rs 0.25.3 Array API surface for reduction + broadcasting divide is unconfirmed, and the empirical precision margin over NFR-001 is unmeasured pre-prototype.

## Context

Phase 2 (length-bucket batching, PR #55-#61) reduced W3 `padding_ratio` from 2.316 to 1.165 and brought W2 and W3 into the primary SLA + padding thresholds. W1 (long-document mix, all three chunks at `seq_len ≈ 8K` resolving to bucket 3) stayed at `padding_ratio = 1.474` and batch/sequential `ratio = 1.254`. Bucket batching is a mechanical no-op for single-bucket workloads, so closing W1 needs a different lever.

`docs/benchmarks/phase2_result.md` L42-47 names three attack paths for W1:

- Phase 3 (this ADR). GPU-side pooling to reduce the fixed readback cost that dominates when every forward pass is 8K × 3.
- Phase 5a. Mutex scope reduction to parallelise `tokenize` and `chunk_plan`.
- Sub-8K chunking. Out of scope.

Today's embed pipeline reads the full `[batch, seq, hidden]` tensor back to CPU after `model.forward()` and runs mask-weighted mean pooling + L2 normalize CPU-side:

- `src/embed/mlx.rs::forward_sub_batch` runs `output.eval()`, then `output.as_slice()`, then `unpack_batch_output` (per-chunk slicing), then `postprocess_embedding`.
- `src/embed/pooling.rs` implements `mean_pooling` + `l2_normalize` as sequential f32 accumulation over `seq_len × hidden_size`.

Three contract gaps this ADR resolves.

1. Readback volume is proportional to `seq × hidden` instead of `hidden`. For W1 (`8192 × 768 × 3` chunks, plus the sequential denominator's equivalent repeats), readback dominates `forward_eval_ms = 15,620`. The pooling math only needs `[batch, hidden]` out. Any post-check that triggers a synchronous `eval() + as_slice()` inside the pool function defeats this lever and is pushed to direct callers instead of the hot path.
2. `release_inference_output(output: Array)` has a one-value contract. `src/mlx_cache.rs:26-31` documents drop-before-clear ordering at compile time by consuming the Array by value. If GPU pooling introduces a second Array (`hidden` + `pooled`), the ordering is no longer compile-time guaranteed.
3. f32 accumulation order drifts between CPU and GPU reductions. CPU `mean_pooling` is strict left-to-right row-major accumulation. Any GPU reduction uses tree or warp-partial aggregation, which introduces per-element `~1e-6` drift that L2 normalize can amplify toward the NFR-001 `max_abs_diff ≤ 1e-5` boundary. Risk is highest at `seq_len = 8192` where 8192 terms are summed per hidden dim.

## Decision

Introduce GPU-side pooling via a new `src/embed/pooling.rs::gpu_pool_and_normalize` function, rewire `forward_sub_batch` and `embed_query_truncated` to use it, and gate the rewiring on an empirical precision probe.

The decision has three sub-decisions.

### Sub-decision 1: Layer placement in embed/pooling.rs

`ModernBert::forward` stays as the `[batch, seq, hidden]` backbone. The new pooling function lives in `src/embed/pooling.rs` alongside the existing CPU `mean_pooling` and `l2_normalize`.

- `ModernBert` is the token-to-hidden-states backbone. Pooling is post-processing specific to the embedding use case, not every consumer of this model.
- Phase 5a (mutex scope reduction) plans to release `tokenize` and `chunk_plan` from the `EmbedderInner` mutex. Keeping `ModernBert::forward` pure simplifies Phase 5a: the lock continues to wrap the GPU forward call, with pre-processing and post-processing running outside the lock.
- The reranker already follows this pattern. `src/reranker/mlx.rs::forward` does on-GPU CLS pooling (`transpose_axes + index(0)`) using the same `ModernBert` backbone unchanged.

### Sub-decision 2: Value-consuming signature for hidden

```rust
pub fn gpu_pool_and_normalize(
    hidden: mlx_rs::Array,
    mask: &mlx_rs::Array,
) -> Result<mlx_rs::Array, Exception>
```

`hidden` is consumed by value. The pooled Array is returned for the caller to pass to `release_inference_output`.

Visibility is `pub` (not `pub(super)`) so the Phase 3a precursor probe binary (`src/bin/gpu_pool_probe.rs`) can call it as a separate crate target. Internal callers (`src/embed/mlx.rs`) access the same symbol via the `pub use pooling::gpu_pool_and_normalize` re-export on `src/embed.rs`.

- Preserves the compile-time drop-before-clear contract declared in `src/mlx_cache.rs:26-31`. Only one Array (the pooled result) survives past this call.
- `mask` stays as `&Array` because the caller may hold it for other purposes (for example, in the query path) and pooling does not consume it.
- A unit test asserts that two live Arrays from the same forward call cannot both reach `release_inference_output`. Move semantics catch this at compile time.

### Sub-decision 3: Phase 3a precursor probe with 10x margin

Phase 3a adds `src/bin/gpu_pool_probe.rs`, a standalone binary that forwards a single W1 chunk (`seq_len = 8192`), runs both `gpu_pool_and_normalize` and the existing `postprocess_embedding` on the same hidden-states Array, and emits `max_abs_diff` and `cosine_sim`. Phase 3b rewiring is gated on the probe reporting `max_abs_diff ≤ 1e-6`, a 10x margin over NFR-001's `1e-5`.

- f32 reduction order drift is empirical, not derivable. Pre-committing to an NFR-001 margin without measurement is the same trap that Phase 2 W3 kernel-compile variance fell into.
- If the probe fails the 10x margin, Phase 3b rewiring blocks and the approach gets re-examined (for example, mixed-precision accumulation with f64 intermediate, or a CPU-hybrid with GPU-side mask multiply + CPU sum).
- The probe bin also serves as the mlx-rs 0.25.3 Array API surface primary-source check. `cargo build --bin gpu_pool_probe` fails fast if `sum(axis, keepdim)`, `sqrt`, or broadcasting `divide` are missing from the bindings.

## Options Considered

### Option A (chosen): GPU pool in embed/pooling.rs with a precursor probe

Pros:

- preserves `ModernBert::forward` as a pure backbone, keeping Phase 5a mutex scope work unaffected
- readback volume drops from `O(seq × hidden)` to `O(hidden)` per batch entry
- precision gate is empirical and fails fast before production rewiring

Cons:

- `src/embed/pooling.rs` grows during Phase 3b (both CPU and GPU variants live side by side until Phase 3c cleanup)
- adds a probe bin target that exists only until Phase 3c cleanup

### Option B: `ModernBert::forward_pooled` in model.rs

Pros:

- single call site in the embed forward path
- backbone consumer can opt into pooled output directly

Cons:

- mixes backbone responsibility with post-processing
- conflicts with Phase 5a mutex scope reduction, which expects `ModernBert::forward` to stay pure
- reranker would end up asymmetric: CLS pooling in `src/reranker/mlx.rs`, mean pooling inside `ModernBert`

### Option C: MLX compile cache for the pool graph

Pros:

- repeated same-shape calls amortise pool graph construction

Cons:

- secondary to readback shape reduction (the primary lever)
- compile cache policy is Phase 4 scope, mixing concerns
- adds shape-cache invalidation logic to Phase 3

## Consequences

Positive:

- W1 `forward_eval_ms + readback_pool_ms` drops. Exact magnitude is aspirational, measured post-prototype in `phase3_result.md`.
- `ModernBert::forward` remains a pure backbone. Phase 5a mutex scope work is unaffected.
- Issue #52 AC "CPU-side pooling と GPU-side pooling の結果差分が許容範囲内" is reached.
- NFR-001 regression detection is explicit via the probe bin. Phase 2 aspirational diagnostics continue as ongoing regression detectors.
- `gpu_pool_and_normalize` stays readback-free. The only NaN source (fully-masked row, `0/0` on divide) is already rejected upstream by `ModernBert::forward::validate_attention_mask`; the pool function relies on that invariant instead of re-scanning its output. Direct callers that bypass `ModernBert::forward` (the Phase 3a probe bin) add their own `is_finite` guard as a defensive check.

Negative:

- `src/embed/pooling.rs` grows from 85 lines to roughly 165 lines while both variants coexist. Phase 3c cleanup removes the CPU variant after rewiring stabilises, but Phase 3b to 3c leaves a dead-code window.
- Approach is conditional on mlx-rs 0.25.3 exposing reduction + broadcasting ops. If not, Phase 3 is deferred until a bindings update or an alternative (f64 intermediate, CPU-hybrid) is chosen.
- Probe bin is additional binary target surface that must be kept green until Phase 3c deletes it.
- FR-001a invariant is now distributed across `validate_attention_mask` (upstream rejection) and the probe bin (`is_finite` defensive guard). A regression that removes either location would leave a NaN-pathway silently open. Mitigation: T-011 is re-typed as an integration scenario on the probe bin; any replacement wiring must add an equivalent guard at the new direct-caller site.

## Migration Plan

1. Phase 3a. Add `src/bin/gpu_pool_probe.rs` and `gpu_pool_and_normalize` in `src/embed/pooling.rs`. Probe bin measures `max_abs_diff` and `cosine_sim` on W1 single chunk. Production paths unchanged.
2. Phase 3b. If probe passes the 10x margin, rewire `forward_sub_batch` and `embed_query_truncated`. Delete `unpack_batch_output`. CPU variants in `pooling.rs` become dead code but are not yet removed.
3. Phase 3c. Remove the CPU variants (`mean_pooling`, `l2_normalize`, `postprocess_embedding`), remove the probe bin, write `docs/benchmarks/phase3_result.md` with measured readback reduction and Phase 2 aspirational recovery status.

## Reassessment Triggers

- Phase 3a probe reports `max_abs_diff > 1e-6`. Approach is re-examined with f64 intermediate accumulation or CPU-hybrid fallback before Phase 3b.
- mlx-rs 0.25.3 bindings are missing `sum(axis, keepdim)`, `sqrt`, or broadcasting `divide`. Phase 3 is deferred or the binding gap is submitted upstream.
- Phase 5a lands first and reduces W1 ratio below 0.80 on its own. The readback attack retains its diagnostic value even if ratio is no longer the primary motivator.

## References

- Issue #52: `embed pipeline のスループット改善余地を計測して段階的に最適化する`
- `docs/benchmarks/phase2_result.md` L42-47 (Phase 3 / 5a attack paths)
- `src/embed/mlx.rs::forward_sub_batch:285-324` (rewire target)
- `src/embed/mlx.rs::embed_query_truncated:144-189` (rewire target)
- `src/embed/pooling.rs::postprocess_embedding:56-85` (CPU reference implementation)
- `src/mlx_cache.rs::release_inference_output:30-44` (drop-before-clear contract)
- `src/reranker/mlx.rs::forward:124-143` (existing on-GPU pooling pattern)
- `tests/fixtures/phase2_baseline/w{1,2,3}.bin` (NFR-001 reference fixtures, regeneration not required because fixture format stores the final pooled vector)
