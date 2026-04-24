# Phase 1 Baseline — embed pipeline

Captured on `feat/embed-phase2-prep-52` at commit prior to any bucket-batching changes. Values are single-run measurements on Apple Silicon + cached `cl-nagoya/ruri-v3-310m`. All workloads are warmed up once per shape before timing (see `src/bin/mlx_smoke.rs::run_measure_baseline` and the 3-workload warmup loop).

These numbers are the comparison target for Phase 2 (issue [#52](https://github.com/thkt/rurico/issues/52)) and must be re-measured as `docs/benchmarks/phase2_result.md` after bucket batching lands.

## Workloads

Definitions live in [`src/embed/workloads.rs`](file:///Users/thkt/GitHub/cli/rurico/src/embed/workloads.rs). Both the library's `#[cfg(test)]` suite and the `mlx_smoke` binary import from there, so any workload edit invalidates the committed fixtures and must be accompanied by `mlx_smoke capture-fixture` + `mlx_smoke measure-baseline`.

| ID | Shape | Characterisation |
| --- | --- | --- |
| W1 | 2 long texts (~48K + ~22K chars) | Long-document mix. 3 chunks total after splitting. Phase 1's original smoke workload shape |
| W2 | 100 short texts (~55 chars each) | Short-text batch. One chunk per text, `max_seq_len=19` |
| W3 | 10 alternating long/short (5 + 5) | Long × short interleave. Single-chunk each but heavy length dispersion |

## Batch vs Sequential

`batch_ms` = wall-clock of one `embed_documents_batch(texts)` call. `sequential_ms` = sum of one `embed_document(text)` per input.

| Workload | `batch_ms` | `sequential_ms` | `ratio = batch / sequential` | Spec AC-2 target (`≤ 0.80`) |
| --- | --- | --- | --- | --- |
| W1 | 18,283 | 14,989 | **1.220** | FAILS — batch is 22% slower than sequential |
| W2 | 592 | 5,870 | **0.101** | PASSES — batch is 9.9× faster |
| W3 | 6,668 | 3,899 | **1.710** | FAILS — batch is 71% slower than sequential |

Bucket batching (Phase 2 core) is expected to bring W1 and W3 under `0.80`, where W3 has the largest absolute headroom (`padding_ratio=2.316`).

## Per-phase metrics (from `PhaseMetrics::log()`)

Values are from the same `measure-baseline` run, with each workload's shape warmed once before timing.

| Workload | `tokenize_ms` | `chunk_plan_ms` | `forward_eval_ms` | `readback_pool_ms` | `cache_clear_ms` | `real_tokens` | `padded_tokens` | `padding_ratio` | `num_chunks` | `batch_size` | `max_seq_len` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | 0 | 714 | 17,385 | 178 | 4 | 16,666 | 24,573 | **1.474** | 3 | 3 | 8,191 |
| W2 | 0 | 4 | 561 | 19 | 6 | 1,890 | 1,900 | **1.005** | 100 | 100 | 19 |
| W3 | 0 | 39 | 6,563 | 59 | 5 | 6,675 | 15,460 | **2.316** | 10 | 10 | 1,546 |

### Observations

- **`padding_ratio=2.316` on W3** means 57% of all positions processed are padding. This is the single biggest optimisation target for Phase 2.
- **W2 is already optimal** (`padding_ratio=1.005`). Phase 2 must not regress W2 throughput — the SLA bar is "≤ baseline ratio", not "improve".
- **`chunk_plan_ms=714` on W1** captures the `shrink_chunk_to_fit` loop's re-tokenisation. Phase 5b (binary search) targets this.
- **`forward_eval` dominates** in every workload (≥ 94% of batch wall-clock). No CPU-side optimisation will move the needle until `padded_tokens` drops.

## Fixtures

Captured binary fixtures at the same commit for the Phase 2 bit-exact / numerical-equivalence check:

```
tests/fixtures/phase2_baseline/w1.bin    9.2 KB  (2 docs × 3 chunks × 768 dim × 4 B)
tests/fixtures/phase2_baseline/w2.bin  308 KB   (100 docs × 1 chunk × 768 dim × 4 B)
tests/fixtures/phase2_baseline/w3.bin   31 KB   (10 docs × 1 chunk × 768 dim × 4 B)
```

Format: little-endian, self-describing per chunk (see `src/embed/fixtures.rs` module docs). Spec NFR-001 requires `cosine_similarity ≥ 0.99999 AND max_abs_diff ≤ 1e-5` between Phase 2 output and these fixtures.

## How to reproduce

```sh
# Build
cargo build --bin mlx_smoke

# Re-capture fixtures (overwrites committed .bin)
target/debug/mlx_smoke capture-fixture

# Re-measure baseline (stderr, ignore the [DEBUG] lines for summary)
target/debug/mlx_smoke measure-baseline 2> baseline.log
grep "^baseline" baseline.log
```

## Notes

- `run_measure_baseline` warms each workload's shape once before timing (W1 and W3 each trigger distinct MLX compile-cache entries). A cold-start first call adds ~500 ms on this machine.
- `sequential_ms` for W1 is close to `batch_ms` because both run 3 forward passes at `seq_len ≈ 8K`. The batch cost is dominated by its worst chunk, not the sum.
- W3's sequential path amortises the short texts' cost against its own small `seq_len`, while the batch path pads every short text to the longest long one — that's the 1.71× penalty.
