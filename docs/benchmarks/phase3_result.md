# Phase 3 Result — embed pipeline

Phase 3 outcomes (issue [#52](https://github.com/thkt/rurico/issues/52)) for the GPU-side pooling effort, measured on Apple Silicon + cached `cl-nagoya/ruri-v3-310m` after PR #62 (Phase 3a probe) and PR #63 (Phase 3b production rewire) had both merged. Comparison target is [`phase2_result.md`](./phase2_result.md). Each timed number below is the median of `MEASURE_REPEATS = 3` runs per workload after warm-up (see `src/bin/mlx_smoke.rs::run_measure_baseline`).

Regenerate via `target/release/mlx_smoke measure-baseline` on any later Phase 3 commit. Numerical equivalence against Phase 1 fixtures continues to be enforced bit-near-exact through `mlx_smoke verify-fixture` (NFR-001).

## Headline result

ADR 0002 primary lever proven empirically: `readback_pool_ms = 0` on every workload, replacing the Phase 2 `O(seq × hidden)` readback with a single `[batch × hidden]` GPU-pooled tensor. Phase 1 fixture parity holds at `cosine = 1.000000` and `max_abs_diff` 13–34× under the `1e-5` threshold. The end-to-end `batch_ms` reduction lands at -6.9% (W1) / -5.8% (W2) / -50.6% (W3) median-of-3 (W3 caveat in [Aggregate semantics](#aggregate-semantics)).

## AC-2 / NFR-001 numerical equivalence

`mlx_smoke verify-fixture` (release build) compares Phase 3b GPU-pooled output against the committed `tests/fixtures/phase2_baseline/w{1,2,3}.bin` per chunk. Spec NFR-001 requires `cosine ≥ 0.99999 AND max_abs_diff ≤ 1e-5` on every chunk pair.

| Workload | `cosine_min` | `max_abs_diff` | Margin to 1e-5 |
| --- | --- | --- | --- |
| W1 | 1.000000 | 7.749e-7 | 12.9× |
| W2 | 1.000000 | 2.980e-7 | 33.5× |
| W3 | 1.000000 | 4.172e-7 | 24.0× |

All three workloads PASS NFR-001 with comfortable margin, validating that GPU-side `mean → divide → l2_normalize` (in `src/embed/pooling.rs::gpu_pool_and_normalize`) reproduces the Phase 1 CPU reference within ADR 0002 sub-decision 3's tolerance budget.

## NFR-002 readback elimination

`measure-baseline` emits a per-workload `readback_shape[wN]: hidden_size=H total_rows=R total_flat=F` banner where `total_flat == total_rows × hidden_size` (asserted in `tests/mlx_smoke.rs` T-006). Phase 2 readback was `O(seq × hidden)` per chunk; Phase 3b readback is `O(batch × hidden)` once.

| Workload | `total_rows` (= `batch × num_chunks`) | `total_flat` | `readback_pool_ms` | Phase 2 equivalent readback (estimated) |
| --- | --- | --- | --- | --- |
| W1 | 3 | 2,304 | 0 | ~18.9M elements (3 chunks × 8192 seq_len × 768 hidden) |
| W2 | 100 | 76,800 | 0 | ~1.46M elements (100 chunks × 19 seq_len × 768 hidden) |
| W3 | 10 | 7,680 | 0 | ~8.36M elements (10 chunks at bucket-padded seq_len × 768 hidden) |

`readback_pool_ms = 0` reflects that the `batch × 768` flat slice is small enough to be dominated by the surrounding async-eval handshake. The structural reduction stands regardless of whether the millisecond floor is 0 or sub-1 on a given run.

## NFR-004 batch-time reduction

### Aggregate semantics

Phase 2 timed `forward_eval_ms` to include the post-forward CPU `mean_pooling + l2_normalize` (since the readback was the last step inside the timed window). Phase 3b moves pooling onto the GPU, so the comparable aggregate is `forward_eval_ms + readback_pool_ms`, not either axis alone. A `forward_eval_ms` uptick on W1 (+1.0%) reflects the GPU pool work shifting into that bucket; the offsetting `readback_pool_ms` drop (Phase 2 was non-zero, Phase 3 is 0) wipes it out at the aggregate level.

### Per-workload median-of-3

| Workload | Phase 2 `batch_ms` | Phase 3 `batch_ms` | Δ `batch_ms` | Δ `forward_eval_ms` | Phase 3 `readback_pool_ms` |
| --- | --- | --- | --- | --- | --- |
| W1 | 17,024 | 15,852 | **-6.9%** | +1.0% | 0 |
| W2 | 565 | 532 | **-5.8%** | -3.3% | 0 |
| W3 | 5,150 | 2,544 | **-50.6%** | (W3 caveat below) | 0 |

W1 saturated diagnostic only (every chunk in bucket 3); the -6.9% reduction comes from readback elimination on a workload that bucket batching could not improve. W2 was already at primary PASS in Phase 2. The additional -5.8% comes from the GPU pool on a hot path that was already fast.

### W3 baseline caveat

Phase 2 W3 `batch_ms = 5,150` was a single-run value. Across four separate `measure-baseline` invocations during Phase 2 the W3 ratio spanned 0.87 / 0.97 / 1.08 / 0.97 (`phase2_result.md::Batch vs Sequential`), so the `5,150` cell sits at the high end of the observed variance band. Phase 3 W3 `batch_ms = 2,544` is itself median-of-3 from a single Phase 3 invocation. The headline `-50.6%` therefore mixes a Phase 2 outlier with a Phase 3 median; readers should treat the magnitude as "structural improvement, exact percentage variance-bound" rather than a tight bound. Repeated Phase 3 invocations would refine the central tendency.

## Workloads

Unchanged from Phase 1/2, defined in [`src/embed/workloads.rs`](../../src/embed/workloads.rs). Any workload edit invalidates the fixtures (`mlx_smoke capture-fixture`) and these numbers.

| ID | Shape | Characterisation |
| --- | --- | --- |
| W1 | 2 long texts (~48K + ~22K chars) | Long-document mix. 3 chunks total after splitting |
| W2 | 100 short texts (~55 chars each) | Short-text batch. One chunk per text, `max_seq_len=19` |
| W3 | 10 alternating long/short (5 + 5) | Long × short interleave. Heavy length dispersion |

## Phase 3 vs Phase 2 architectural delta

| Aspect | Phase 2 | Phase 3 |
| --- | --- | --- |
| Pooling location | CPU after readback | GPU before readback (`gpu_pool_and_normalize`) |
| Per-chunk readback | `seq_len × hidden_size` floats | `hidden_size` floats |
| Non-finite guard | `postprocess_embedding` post-pool CPU `is_finite` | `split_pooled` post-readback `is_finite` (same coverage, smaller buffer) |
| All-zero mask handling | CPU mask sum → 0 → unchanged vector | GPU `f32::MIN_POSITIVE` clamp + upstream `validate_attention_mask` rejects fully-masked rows |
| Drop-before-clear ordering | Implicit via `?`-exit | Compile-time enforced via `pool_output(Array, ...) -> Result<Array, _>` consume-by-value |

`postprocess_embedding`, `mean_pooling`, `l2_normalize`, and the `gpu_pool_probe` precursor binary are removed once the GPU path was validated.
