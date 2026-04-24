# Phase 2 Result — embed pipeline

Target document for Phase 2 (issue [#52](https://github.com/thkt/rurico/issues/52)) bucket-batching outcomes. Numbers are placeholders — captured and filled in by PR #6 (sub-phase 2E) once all Phase 2 changes land. Comparison target is [`phase1_baseline.md`](./phase1_baseline.md).

Regenerate via `target/debug/mlx_smoke measure-baseline` on a Phase 2 commit. Fixtures themselves continue to be validated bit-exact through `mlx_smoke verify-fixture`; this file tracks the *performance* outcome.

## Workloads

Same W1/W2/W3 as Phase 1, defined in [`src/embed/workloads.rs`](../../src/embed/workloads.rs). Any workload edit invalidates both fixtures (`mlx_smoke capture-fixture`) and these numbers.

| ID | Shape | Characterisation |
| --- | --- | --- |
| W1 | 2 long texts (~48K + ~22K chars) | Long-document mix. 3 chunks total after splitting |
| W2 | 100 short texts (~55 chars each) | Short-text batch. One chunk per text, `max_seq_len=19` |
| W3 | 10 alternating long/short (5 + 5) | Long × short interleave. Heavy length dispersion |

## Batch vs Sequential

Spec AC-2 target: `batch_ms / sequential_ms ≤ 0.80` (W2 allows negotiated relaxation per PR #0 observation).

| Workload | Phase 1 `batch_ms` | Phase 2 `batch_ms` | Phase 1 ratio | Phase 2 ratio | AC-2 pass? |
| --- | --- | --- | --- | --- | --- |
| W1 | 17,024 | TBD (PR #6) | 1.243 | TBD | TBD |
| W2 | 565 | TBD (PR #6) | 0.094 | TBD | TBD (already passes) |
| W3 | 5,150 | TBD (PR #6) | 1.443 | TBD | TBD |

## Per-phase metrics (from `PhaseMetrics::log()`)

Phase 2 adds `bucket_hist=[N0,N1,N2,N3]` to this log line (see R-S01, PR #4). NFR-003 target: `padding_ratio ≤ 1.10` on every workload.

| Workload | `tokenize_ms` | `chunk_plan_ms` | `forward_eval_ms` | `padding_ratio` | `num_chunks` | `bucket_hist` |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | TBD | TBD | TBD | TBD | 3 | TBD (PR #6) |
| W2 | TBD | TBD | TBD | TBD | 100 | TBD (PR #6) |
| W3 | TBD | TBD | TBD | TBD | 10 | TBD (PR #6) |

### Comparison to Phase 1

PR #6 fills these once all Phase 2 PRs merge:

- **`padding_ratio` delta**: every workload must drop to `≤ 1.10`. W3's Phase 1 `2.316` is the headline target.
- **`forward_eval_ms` vs `real_tokens`**: PR #6 fits `forward_eval_ms` as a linear function of `real_tokens` across W1/W2/W3 and asserts `R² ≥ 0.95` (NFR-005).
- **`bucket_hist` distribution**: W2 expected to concentrate in bucket 0 (`[100,0,0,0]`), W1 and W3 to span 2–3 buckets.

## Fixtures

Phase 2 output must match the committed baseline fixtures at `tests/fixtures/phase2_baseline/w{1,2,3}.bin` (captured on the last Phase 1 commit before bucket batching, despite the directory name) within NFR-001 tolerances: `cosine_similarity ≥ 0.99999 AND max_abs_diff ≤ 1e-5`. Verified by `mlx_smoke verify-fixture` — see [`phase1_baseline.md`](./phase1_baseline.md#fixtures) for format details.

## How to reproduce

```sh
# Build
cargo build --bin mlx_smoke

# Measure Phase 2 numbers (identical harness to phase1_baseline.md)
target/debug/mlx_smoke measure-baseline 2> phase2.log
grep "^baseline" phase2.log

# Optional: re-run bit-exact check against Phase 1 fixtures
target/debug/mlx_smoke verify-fixture
```

## Notes

- Numbers TBD until PR #6 (#52 sub-phase 2E). Until then this file is a *target*, not a record — do not paste estimates.
- `measure-baseline` warms both batched and per-document shapes once per workload before timing (see `src/bin/mlx_smoke.rs::run_measure_baseline`). Without the sequential-side warmup, `ratio = batch / sequential` is biased batch-favourably.
