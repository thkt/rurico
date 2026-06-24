---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Adopt uniform vector-length contract for the Embed trait

## Context and Problem Statement

`src/embed.rs:322` の `Embed` trait は `# Contract` doc で「1 つの implementor が返す全ベクトルは同じ長さ (model の `hidden_size`) でなければならない」と表明する。この長さは `Vec<f32>` / `ChunkedEmbedding` の型には現れず、Rust の型システムで強制できない。違反すると sqlite-vec の `vec0` 仮想テーブルが `CREATE VIRTUAL TABLE` 時に固定した次元数と挿入ベクトルの不一致を起こし、index が silent に壊れる (挿入時 reject か、次元境界を跨いだ近傍計算の誤りとして顕在化する)。この invariant は embed↔storage の module 境界を跨ぐが、根拠は trait の doc コメント 1 箇所にしかなく、storage 側 (`vec0` テーブル作成箇所) には対応する記述がない。census audit (`docs/audit/2026-06-24-020141-adr-gaps.md` candidate C6) が incomplete-contract として nominate した。

## Decision Drivers

- 違反すると sqlite-vec index が silent corruption を起こし、検索結果が無言で劣化する (data integrity)
- 長さの一様性は `Vec<f32>` の型に現れず、コンパイラ・lint では守れない
- 制約は embed (生成側) と storage (消費側) の 2 module に跨り、片方の doc だけでは将来の implementor / caller が辿れない
- 新しい `Embed` implementor (将来のモデル追加、テスト用 fake) が次元混在を返す regression は型で防げない

## Considered Options

- trait doc の uniform-length contract を cross-module invariant として ADR で pin する (status quo を明文化)
- 次元を型パラメータ化する (`Embedding<const N: usize>` / newtype) して非一様を表現不能にする
- storage 挿入時に毎回ランタイムで全ベクトル長を検証する

## Decision Outcome

Chosen option: **"uniform vector-length contract を cross-module invariant として ADR で pin する"**。次元の一様性は MLX backend が単一モデルの `hidden_size` を返すという実装事実に依存し、sqlite-vec の固定次元テーブルと結合しているため、契約として明文化する価値がある。型パラメータ化は mlx-rs の動的次元 (`embedding_dims()` はランタイム値) と橋渡しが冗長で、全ベクトル長のランタイム検証は hot path に無駄なコストを乗せる。

### Consequences

- Good, because storage 側 (`vec0` 固定次元) と embed 側 (実装事実) の結合が文書化され、新規 implementor の checklist になる
- Good, because silent corruption の発生条件 (次元混在) が単一の参照点を持つ
- Bad, because invariant は型で強制できず、新規 `Embed` implementor が非一様長を返す regression は code review に委ねる
- Bad, because 将来モデルを動的に切り替えて次元が変わる設計 (複数モデル併存) は本 ADR の前提を破り supersede が必要

## Confirmation

- `src/embed.rs:322-324` の `Embed` trait `# Contract` doc が source of truth
- production implementor `Embedder` (`src/embed/embedder.rs`) は `embedding_dims` を 1 つ保持し、全出力がその長さに揃う
- storage の `vec0` テーブルはモデルの `hidden_size` で固定次元 `CREATE VIRTUAL TABLE` され、挿入ベクトルとの一致が前提
- 新規 `Embed` implementor を追加する PR では「単一 implementor の全出力が同一長か」が code review checklist になる

## Pros and Cons of the Options

### Pin uniform-length contract as cross-module invariant

trait doc を契約の source of truth とし、embed implementor と storage 挿入の両方がこれに従う。

- Good, because 追加コストゼロで現状の実装事実を明文化できる
- Good, because embed↔storage 境界の暗黙の結合が可視化される
- Bad, because 型による強制ではないため regression は review 依存

### Type-parameterize the dimension

`Embedding<const N: usize>` や次元 newtype で非一様を表現不能にする。

- Good, because invariant がコンパイル時に強制される
- Bad, because `embedding_dims()` がランタイム値 (モデル config 由来) で const generic と相性が悪い
- Bad, because sqlite-vec の `i32` shape 引数との橋渡しが全 call site で冗長になる

### Runtime length validation at storage insert

挿入のたびに全ベクトル長を検証して reject する。

- Good, because corruption を挿入時に確実に止められる
- Bad, because index hot path に毎回の長さチェックコストが乗る
- Bad, because 根本原因 (生成側の契約) ではなく症状を消費側で受け止める patch 的対処

## More Information

### Implementation Guidelines

- 新規 `Embed` implementor は単一インスタンスから常に同一長を返すこと。可変長が必要なら別 implementor に分離する
- storage の `vec0` 固定次元を変える / 複数モデルを動的併存させる設計変更は本 ADR を supersede し、次元混在時の sqlite-vec 挙動を測定して新 ADR の Consequences に記録する

### Related

- ADR 0008 (sqlite-vec auto-extension) — sqlite-vec 連携の前提
