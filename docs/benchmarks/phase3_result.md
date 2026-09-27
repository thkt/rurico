# Phase 3 Result — embed pipeline

Phase 3 outcomes (issue [#52](https://github.com/thkt/rurico/issues/52)) for the GPU-side pooling effort, measured on Apple Silicon + cached `cl-nagoya/ruri-v3-310m` after PR #62 (Phase 3a probe) and PR #63 (Phase 3b production rewire) had both merged. Comparison target is [`phase2_result.md`](./phase2_result.md). Phase 3 の batch time は、当時の `MEASURE_REPEATS = 3` による warm-up 後の中央値として報告された値。現在の計測実装とは区別する。

この記録は歴史的な観測であり、現在のコードで再実行しても同じ測定にはならない。現行の再実行手順は [CONTRIBUTING](../../CONTRIBUTING.md#options付き推論の計測issue-306) を参照。 Numerical equivalence against Phase 1 fixtures continues to be enforced bit-near-exact through `mlx_smoke verify-fixture` (NFR-001).

## Headline result

GPU側poolingにより、readbackのshapeは `seq × hidden` から `batch × hidden` へ縮小した。記録上の `readback_pool_ms = 0` は整数msへの切捨てを含み、処理時間ゼロやGPU kernel単独時間を示さない。Phase 1 fixtureとの比較結果は以下に保持する。

**2026-09-27訂正（[Issue #306](https://github.com/thkt/rurico/issues/306)）**:
以前「Phase 2」と記した17,024 / 565 / 5,150msは [Phase 1](./phase1_baseline.md#batch-vs-sequential) の値だった。
[Phase 2](./phase2_result.md#batch-vs-sequential) の16,565 / 617 / 2,696msに訂正した。
以下の差は報告済みの数値を引き直した参考差分で、新しい測定でも、同条件で確かめた速度改善率でもない。
旧版の-6.9% / -5.8% / -50.6%という値と、それをGPU poolingの寄与に帰属した説明は撤回する。
照合した原資料の版は `cbb9068b06d957bcf9496e431cc6092416918d13`。

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

`readback_pool_ms = 0` から分かるのは当時の整数ms表示の値だけであり、async-eval handshakeが支配したという原因はこの記録では確認できない。shape縮小と時間の寄与は区別する。

## NFR-004 batch-time reduction

### Aggregate semantics

前版は `forward_eval_ms` と `readback_pool_ms` の包含関係、およびW1の+1.0%を根拠にreadback削減が相殺したと説明していた。しかしPhase 2の比較表にはreadbackの測定値がなく、Phase 3の対応するforward生recordも本資料に残っていない。その寄与分解とforward差分は未確認として撤回する。現在の区間定義は公開APIの `InferenceMetrics` rustdocを参照し、当時の区間へ遡及適用しない。

### 報告済みbatch timeの訂正比較

| Workload | Phase 2 `batch_ms` | Phase 3 `batch_ms`（保存値） | 報告値間の参考差分 |
| --- | --- | --- | --- |
| W1 | 16,565 | 15,852 | -4.3% |
| W2 | 617 | 532 | -13.8% |
| W3 | 2,696 | 2,544 | -5.6% |

差分は `(Phase 3 / Phase 2 - 1) × 100`。Phase 3のfixture比較・batch time・readback表示値は変更していない。

### W3比較の限界

5,150msはPhase 1由来であり、Phase 2の外れ値という旧説明は誤りだった。
Phase 2本文は1回の `measure-baseline` invocation内で3試行の中央値を使うと説明し、別invocation間のW3比率を0.87 / 0.97 / 1.08 / 0.97と記録している。
Phase 3の2,544msも1 invocationの中央値として保存された値だが、両者の生record・固定環境の対応はこの資料から復元できない。-5.6%を再現可能な改善率やpoolingだけの効果として扱わない。

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
