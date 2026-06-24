---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Adopt MLX-only embedding backend and retire the candle fallback

## Context and Problem Statement

rurico の embedding backend は当初 candle と MLX の 2 実装を Cargo feature (`mlx = ["dep:mlx-rs"]`, `default = ["mlx"]`) で切り替える設計だった。2026-03-25 (`a7afcdd`) に `src/embed/candle.rs` と candle 依存を削除して MLX を唯一の backend にし、2026-03-28 (`a03ab95`, BREAKING CHANGE) に `mlx` feature flag 自体を撤廃して `mlx-rs` を非 optional な通常依存に格上げした。結果として `src/embed/embedder.rs` は MLX backend (`EmbedderInner`) のみを Mutex で包む単一実装になり、backend 抽象 (feature gate / 代替 trait impl) は残っていない。この「MLX 専用、candle は復活させない」という決定は code (削除済みファイルの不在) からは読み取れず、commit message にのみ根拠がある。census audit (`docs/audit/2026-06-24-020141-adr-gaps.md` candidate C3) が nominate した。

## Decision Drivers

- candle と MLX の 2 backend 維持は、Metal 最適化 (GPU 側 pooling, compile cache) を片方でしか活かせず二重保守コストを生む
- MLX は Apple Silicon の Metal を直接使い、production target (macOS / Apple Silicon) と一致する
- feature gate の `#[cfg(feature = "mlx")]` (18 箇所) が test / CI matrix を複雑化していた
- downstream バイナリが `rurico/mlx` feature を参照していたため、撤廃は BREAKING で cross-repo 追従が要る

## Considered Options

- MLX 専用にし candle backend を完全撤去する (採用)
- candle を feature gate 背後に残し、非 Apple 環境向けの fallback として維持する
- backend trait 抽象を保ち、将来の backend 追加に備える

## Decision Outcome

Chosen option: **"MLX 専用にし candle を撤去する"**。production target が Apple Silicon に固定され、MLX の Metal 直結最適化 (ADR 0002 の GPU 側 pooling 等) が candle では再現できないため、2 backend の抽象を維持するコストが便益を上回ると判断した。candle 維持は使われない code path の保守と CI matrix 肥大を招き、speculative な backend trait は YAGNI。

### Consequences

- Good, because backend 抽象が消え、`#[cfg(feature = "mlx")]` 18 箇所と feature matrix が CI から除去された
- Good, because MLX 固有最適化を分岐なしに前提にできる
- Bad, because 非 Apple-Silicon 環境では rurico の embedding がビルド・実行不能になった (production target 外として許容)
- Bad, because 別 backend が将来必要になった場合、trait 抽象を新規に導入する設計変更 (本 ADR の supersede) が要る

## Confirmation

- `src/embed/embedder.rs` が MLX backend (`EmbedderInner`) のみを包む単一実装である
- `Cargo.toml` に `mlx` feature と candle 依存が存在しない (`a03ab95` 以降)
- `src/embed/candle.rs` が存在しない
- backend を再び複数化する PR は本 ADR の Decision を破るため supersede が必要

## Pros and Cons of the Options

### MLX-only, candle removed (chosen)

candle source / feature / 依存を全削除し MLX を非 optional 依存にする。

- Good, because 単一 backend で feature gate と CI matrix が消える
- Good, because MLX Metal 最適化を分岐なしに前提化できる
- Bad, because 非 Apple-Silicon でビルド不能

### Keep candle behind a feature gate

candle を fallback として feature 背後に残す。

- Good, because 非 Apple 環境でも embedding が動く
- Bad, because 使われない path の二重保守と `#[cfg]` 分岐の恒常的コスト
- Bad, because MLX 固有最適化が candle path で再現できず実装が乖離する

### Preserve a backend trait abstraction

backend trait を保ち将来追加に備える。

- Good, because 将来の backend 追加が容易
- Bad, because 現時点で 2 実装目が存在せず speculative abstraction (YAGNI 違反)
- Bad, because trait 越しでは MLX 固有 API (GPU pooling 等) を素直に呼べない

## More Information

### Deprecation Target

- 削除済み: `src/embed/candle.rs`、candle 依存、`mlx` Cargo feature (`default = ["mlx"]` / `mlx = ["dep:mlx-rs"]`)
- 撤去コミット: `a7afcdd` (candle source 削除, 2026-03-25) → `a03ab95` (mlx feature 撤廃 BREAKING, 2026-03-28)

### Migration / Rollback

- downstream バイナリは `rurico/mlx` feature 参照を除去する (BREAKING 追従、各リポで個別対応)
- backend を再導入する場合は本 ADR を supersede し、(a) backend 選択機構 (feature か runtime か)、(b) MLX 固有最適化との互換、(c) CI matrix への影響を新 ADR に記録する

### Related

- ADR 0002 (GPU-side pooling) — MLX backend に依存する最適化
- ADR 0005 (prefix-ensemble experiment not adopted) — 不採用を記録する先例パターン
