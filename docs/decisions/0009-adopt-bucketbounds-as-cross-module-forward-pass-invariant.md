---
status: "accepted"
date: 2026-05-14
decision-makers: thkt
---

# Adopt BUCKET_BOUNDS as cross-module forward-pass invariant

## Context and Problem Statement

`src/model_io.rs::BUCKET_BOUNDS = [128, 512, 2048, MAX_SEQ_LEN]` は ModernBert forward の token length を 4 段階の bucket に丸める cross-module invariant。embed (`src/embed/mlx.rs::forward_sub_batch` / `embed_query_truncated`) と reranker (`src/reranker/mlx.rs::forward_sub_batch`) の全 caller が `assign_bucket` で round up し、`model.forward(_, _, _, bucket_len)` に渡す。同時に `src/modernbert/model.rs::ModernBert::local_mask_cache` はこの 4 値で keyed されており、cache size は **常に 4 entries** に bounded される。違反すると mask cache が膨らみ、MLX compile cache が呼び出しごとに新 kernel を JIT してメモリ暴発する。invariant は (a) `assign_bucket` panic、(b) `local_mask_cache` doc コメント、(c) `compute_sub_batch_size` test に分散しており、ADR で集約された記述はなかった。

## Decision Drivers

* 違反すると MLX compile cache に kernel が無限に蓄積し、production OOM の遠因になる
* 4 つの magic number (`128 / 512 / 2048 / 8192`) の由来 (= モデル `max_position_embeddings`) を ADR なしで読み解くのは困難
* `assign_bucket` panic は `len > MAX_SEQ_LEN` を防ぐが、「caller が round up を忘れて中間値で `forward()` する」regression は型・lint で守れない
* bucket 値を 5 個に増やす / 値を変える設計変更は cross-module で影響、ADR がないと安全に検討できない

## Considered Options

* Adopt `BUCKET_BOUNDS = [128, 512, 2048, MAX_SEQ_LEN]` as a frozen cross-module invariant (status quo to be pinned)
* Allow dynamic bucket sizing per call site (no fixed table)
* Encode buckets as a type-level enum so non-bucket values are unrepresentable in `model.forward`

## Decision Outcome

Chosen option: **"Adopt `BUCKET_BOUNDS` as a frozen cross-module invariant"**, because 4 値の選定 (geometric-ish coverage with `MAX_SEQ_LEN` cap) は実測トレードオフの結果であり、cross-module で 3 caller (embed × 2 path + reranker × 1) と `local_mask_cache` 整合性に縛られているため。bucket 値の変更は ADR supersede 経由で行う。

### Consequences

* Good, because `local_mask_cache` が常に 4 entries に bounded され MLX compile cache も 4 kernel に収まる
* Good, because 4 値の trade-off (バケット数 × 内 padding ratio) が文書化される
* Bad, because runtime で bucket 数を可変にする最適化 (例: workload に応じて 5 段階) は ADR 改訂が必要
* Bad, because invariant は型システムで強制できず、新規 caller が `assign_bucket` を経由せず生 `seq_len` を渡す regression は code review に委ねる

### Confirmation

- `src/model_io.rs:41-54` の `BUCKET_BOUNDS` const + `assign_bucket` doc が source of truth
- `compute_sub_batch_size_matches_formula_per_bucket` (`src/model_io.rs:450-460`) test が各 bucket の具体的 sub-batch サイズ `(2000, 500, 125, 31)` を pin
- `src/modernbert/model.rs::ModernBert::local_mask_cache` doc が `BUCKET_BOUNDS` への intra-doc link で integrity を表明
- 新規 forward caller を追加する PR では `BUCKET_BOUNDS` への round-up が code review checklist になる

## Pros and Cons of the Options

### Adopt `BUCKET_BOUNDS = [128, 512, 2048, MAX_SEQ_LEN]` as frozen invariant

3 caller (`forward_sub_batch` embed / `embed_query_truncated` / `forward_sub_batch` reranker) が `assign_bucket` で round up、`local_mask_cache` も同じ 4 値で keyed。

* Good, because compile cache + mask cache 両方が 4 entries に bounded
* Good, because 4 値の trade-off が ADR で説明可能
* Bad, because workload-adaptive bucket は不可

### Allow dynamic bucket sizing per call site

caller ごとに自分の最適 bucket 数を決める。

* Good, because workload-specific 最適化が可能
* Bad, because `local_mask_cache` が unbounded growth、MLX compile cache も同様
* Bad, because bucket cache hit rate が caller 間で予測不能

### Type-level bucket enum

`enum Bucket { B128, B512, B2048, B8192 }` を作り `model.forward(_: Bucket)` に渡す。

* Good, because invariant が型で強制される
* Bad, because mlx-rs `Array::from_slice` の shape 引数 (`i32`) との橋渡しが冗長
* Bad, because `BUCKET_BOUNDS` 値を変更する度に enum variant も同期する二重管理

## More Information

### Implementation Guidelines

- 新規 `model.forward` caller は必ず `assign_bucket(seq_len)` → `BUCKET_BOUNDS[bucket_idx]` で round up してから渡す
- `BUCKET_BOUNDS` の値を変更する場合は本 ADR を supersede し、4 値変更による (a) padding ratio 影響、(b) compile cache memory 影響、(c) `local_mask_cache` size を benchmark で測定して新 ADR の Consequences に記録する
- `local_mask_cache` の cap を 4 以外に変える設計変更は本 ADR の Decision を破るため supersede 必須

### Monitoring

`forward_sub_batch` の `BatchMetrics::padding_ratio` が長期的に 1.5 以上に張り付く場合、現在の 4 値が workload と合っていない signal。`docs/benchmarks/phase{2,3}_result.md` の方法論で再計測し、本 ADR を supersede する。
