---
status: "accepted"
date: 2026-05-14
decision-makers: thkt
---

# Adopt MASK_FILL constant to avoid Metal kernel NaN

## Context and Problem Statement

ModernBert backbone (`src/modernbert/model.rs::prepare_4d_attention_mask`) は attention mask を `[batch, 1, 1, seq_len]` の f32 tensor に展開し、masked position に `MASK_FILL = -1e9` を埋め込む。一般的な PyTorch / Transformers 実装は `f32::NEG_INFINITY` を使うが、Apple Silicon の MLX Metal kernel は `seq_len ≲ 9` token 付近で softmax の中間計算が `0.0 * (-inf) = NaN` を引き起こし、forward 出力に NaN が混入する。これは下流の embedding L2 normalize で `0/0` を作り、検索品質を黙って壊す。`-1e9` は softmax で十分に exp underflow して masked rate ≈ 0、かつ `0.0 * (-1e9) = -0.0` で finite を保つ。この選定理由は `src/modernbert/model.rs:382-384` の inline comment にしかなく、新規メンテナが PyTorch 由来の慣習で `f32::NEG_INFINITY` に「直そう」とすると short-sequence NaN regression を再発させる。

## Decision Drivers

* Apple Silicon Metal kernel の `0.0 * (-inf) = NaN` ハザードは hardware-specific で `cargo test` だけでは事前に検知できない (smoke 環境必須)
* math defaults は `-inf` 方向 (PyTorch / FlashAttention) なので "戻そう" 修正が起きやすい
* NaN が混入すると `EmbedError::NonFiniteOutput` で fail-fast するが、その手前で L2 normalize の `0/0` を作って検索ベクトル全体が NaN になる被害が大きい

## Considered Options

* Adopt `MASK_FILL = -1e9` (status quo to be pinned)
* Switch to `f32::NEG_INFINITY` (PyTorch / Transformers convention)
* Adopt mixed-precision masking (`f32::MIN / 2` ≈ `-1.7e38` or `-1e30`)

## Decision Outcome

Chosen option: **"Adopt `MASK_FILL = -1e9`"**, because Apple Silicon Metal kernel が short-sequence で `0.0 * (-inf) = NaN` を生む実測ハザードを `-inf` は trip し、`-1e9` は softmax masking として十分でかつ `0.0 * (-1e9) = -0.0` (finite) を保つため。precision (softmax の数値誤差は約 `exp(-1e9) ≈ 0` で十分に underflow) と numerical stability (NaN 回避) の real trade-off の解。

### Consequences

* Good, because `seq_len ≲ 9` を含む全 forward path で NaN free
* Good, because L2 normalize の `0/0` 経路が閉じ、`EmbedError::NonFiniteOutput` に頼らずに健全な出力を確保
* Bad, because PyTorch 系の慣習と異なるため、新規メンテナは「`-inf` に戻そう」と思いがち
* Bad, because `local attention mask` は別の経路 (`get_local_attention_mask`) で `f32::NEG_INFINITY` を使う (直接書き込みで mask 乗算経由しないため安全)、2 通りの fill 値が共存することによる読みづらさ

### Confirmation

- `src/modernbert/model.rs:384` の `const MASK_FILL` + inline comment が rationale 起点
- 本 ADR が rationale の正本となり、`MASK_FILL` 関連 PR は本 ADR を参照する
- smoke 経由の `mlx_smoke verify-fixture` (ADR 0002) で短 seq の NaN regression を捕捉できる
- `local attention mask` 側で `-inf` を使い続ける理由 (mask 乗算経由しないため `0 * -inf` ハザード対象外) は `get_local_attention_mask` 関数の comment 内 (`src/modernbert/model.rs:466-467`) に明記

## Pros and Cons of the Options

### Adopt `MASK_FILL = -1e9`

mask 乗算経由する global attention mask 専用の有限値。

* Good, because Apple Silicon Metal kernel の NaN ハザードを回避
* Good, because softmax masking として十分な大きさ (`exp(-1e9)` は f32 underflow)
* Bad, because PyTorch 系の慣習 (`-inf`) と異なる

### Switch to `f32::NEG_INFINITY`

PyTorch / Transformers / FlashAttention の慣習。

* Good, because mainstream の参照実装と一致、新規参加者が読みやすい
* Bad, because Apple Silicon Metal kernel が短 sequence で `0.0 * (-inf) = NaN` を生成する実測ハザード
* Bad, because NaN は L2 normalize 経由でベクトル全体を破壊、silent corruption

### Adopt mixed-precision masking (`f32::MIN / 2`)

`-1.7e38` 等の有限・大値を使う。

* Good, because softmax masking としてさらに underflow、`-inf` 互換の数学的振る舞い
* Good, because finite (NaN ハザード回避)
* Bad, because mask + 残差 (`add` op) で f32 overflow を生む可能性、Metal kernel での実測未確認
* Bad, because `-1e9` で実証済の rationale を上書きする実利なし

## More Information

### Implementation Guidelines

- 新規 attention mask 経路を加える際、mask を **乗算** で reduce するなら `MASK_FILL = -1e9` を踏襲する
- mask を **直接書き込み** で reduce する経路 (`get_local_attention_mask` のように `0 * fill` を経由しない) は `f32::NEG_INFINITY` 可
- `MASK_FILL` を変える場合は本 ADR を supersede し、`mlx_smoke verify-fixture` で short / long seq 両方の cosine_min が ADR 0002 の許容範囲内に収まることを再確認

### Monitoring

`EmbedError::NonFiniteOutput` が production で発生したら本 ADR の前提 (mask fill が十分な underflow を生む) が破れている signal。`mlx_smoke verify-fixture` を最新の mlx-rs / macOS / Apple Silicon バージョンで再走する。

### References

- `src/modernbert/model.rs:384` (`MASK_FILL` const definition)
- `src/modernbert/model.rs:414-426` (`prepare_4d_attention_mask` — `MASK_FILL` 適用箇所)
- `src/modernbert/model.rs:466-467` (`get_local_attention_mask` — `f32::NEG_INFINITY` 直接書き込み)
- ADR 0002 (GPU-side pooling — embedding pipeline numerical reproducibility)
