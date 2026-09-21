---
status: "accepted"
date: 2026-06-24
decision-makers: thkt
---

# Adopt MLX-only embedding backend and retire the candle fallback

## Context and Problem Statement

ruricoは2026-03-25 (`a7afcdd`) にcandle sourceと依存を削除し、
2026-03-28 (`a03ab95`, BREAKING CHANGE) に `mlx` feature自体を撤廃して
`mlx-rs` を非optional依存にした。当時は実backendの二重保守をなくすための変更だった。
2026-06-24の本ADRは「実推論backendはMLXのみ、candleを復活させない」という判断を記録した。

その後の [Issue #305](https://github.com/thkt/rurico/issues/305) は、実backendの選択と
CPU処理・公開契約を検証するための依存境界を分ける。初回調査の `32513da` では、rootの
`mlx-rs` とffiの `mlx-sys` が非optionalで、`test-support` だけではMetalビルドを除けなかった。
実装開始版 `c88cd3bf4665405e837b8bdfdce287f04f31e5e0` でもこの依存関係は同じだった。
開始版には #300 の重み検証が追加されており、そのconfig・ヘッダー検証も共有する。

## Decision Drivers

- 実推論はApple Silicon上のMLXで維持し、GPU pooling・compile cache等の既存最適化を保つ
- traits・options・error・mock・storage・retrievalと純粋処理を、Metalとモデル重みなしで検証する
- 既定構成の公開pathを維持し、downstreamの移行とcrate間の型・依存管理を最小限にする
- CPUテスト用にtokenization・長文計画・bucket分配・出力検証をコピーしない

## Considered Options

| 方式 | 保守・移行への影響 | 判断 |
| --- | --- | --- |
| 既存root/ffiにdefault-onの `mlx` featureを設ける | 既存crateと型の所属を維持する。CPU/MLXの2構成と限定したcfgの保守が必要 | #305で採用 |
| core / MLX / 互換facadeへcrate分割する | native依存をcrateで分離できる。一方で内部型・artifact・model I/Oの共有境界、re-export、manifestと依存方向の管理が増える | 今回は不要 |
| 非optionalのMLX依存を維持する | feature管理は少ないが、CPU公開契約の検証にもMetalビルドが必要 | #305の完了条件を満たさない |
| candle等を代替backendとして導入する | 別実装の数値互換・最適化・CIを維持する必要がある | 引き続き不採用・今回の対象外 |

## Decision Outcome

実推論backendをMLXだけにする判断は維持する。#305の依存分離には、default-onの
`mlx` featureを採用する。これはCPU推論backendの追加ではなく、既存の純粋処理と
公開契約をMLXのnative buildから切り離すための構成である。

- root: `default = ["mlx"]`、`mlx = ["dep:mlx-rs", "rurico-ffi/mlx"]`
- rootからffiへは `default-features = false` とし、MLXの有効化を明示的に伝播する
- ffi単独: `default = ["mlx"]`、`mlx = ["dep:mlx-sys"]`。SQLite登録は常に利用可能
- `test-support` はMLXを有効にしない。`smoke` / `test-mlx` は `mlx` を有効にする
- probe binariesとMLX loader/probe integration testsは `required-features = ["mlx"]`

`embed::Embedder`、`reranker::Reranker`、`modernbert::ModernBert`、probe dispatcherとMLX FFIは
`mlx` の背後に置く。既定の公開pathは変えない。`Embed` / `Rerank`、`EmbedOptions`、
error型、`ChunkedEmbedding`、mock、`LazyReranker`、storage、retrieval等はCPU構成でも使える。
純粋処理の実コードを内部のprocessingモジュールへ移し、MLXの呼出元とCPUテストが共有する。

### Consequences

既定利用者のfeature指定変更や別crateへの移行は不要になる。CPU構成の有効依存から
`mlx-rs` / `mlx-sys` を除ける一方、2構成のfeature伝播・lint・公開契約の検証を維持する
必要がある。crate分割は増やさないが、cfgの保守コストがなくなるわけではない。
CPU構成はモデルによる推論を提供しない。recall全体のCPUビルドや別backend導入は対象外。
#306以降のGPU調査を一律に待たせず、#318でruntimeを試作する場合もこの境界と整合させる。

## Confirmation

[CONTRIBUTING](../../CONTRIBUTING.md#cpu構成と公開契約の検証)と `scripts/check-cpu.sh` に
CPU検証を定義する。root・ffi単独とworkspaceの有効依存を確認し、unit・integration・doc・
visibilityテストを実行する。SQLite登録と外部consumerのmock→storage→retrieval→rerankを
含め、全skipを成功とは扱わない。CIはLinuxのCPU検証とmacOSのMLX検証を分ける。

既定MLX構成の既存検証とcoverage閾値・timeoutを維持する。実tokenizer・モデルを要する
ignoredテストとsmokeは引き続き個別実行が必要。CPU成功だけでGPU数値一致を証明しない。
Issueに記録された変更前CIの33分06秒は別版・一回の測定であり、分離後の速度改善を
保証する数値ではない。効果を述べるには同一条件での測定が必要で、検証定義の追加や
ファイル移動を測定結果とは扱わない。

## Migration / History

2026-03の移行ではdownstreamの `rurico/mlx` 参照を除去した。#305では検証用の
`mlx` featureを再導入するが、candle source・依存やbackend切替機構は復活させない。
CPU利用者は `default-features = false` を指定し、mockが必要なら `test-support` を加える。
Cargoのfeatureは加算されるため、同一ビルドの別の依存が既定featureや `mlx` を有効にすると
MLXも有効になる。`--workspace --no-default-features` で両memberの既定featureを無効にする。

別の実backendを導入する場合は本ADRのMLX-only判断をsupersedeし、選択機構、MLX固有
最適化との互換、数値検証とCIへの影響を改めて記録する。

## Related

- [Issue #305](https://github.com/thkt/rurico/issues/305): 今回の合意範囲と初回調査・CI測定の参照
- [ADR 0002](0002-gpu-side-pooling-embed.md): MLXのGPU pooling
- [ADR 0008](0008-adopt-process-level-sqlite-vec-auto-extension.md): SQLite登録の契約
- [2026-06-24 audit](../audit/2026-06-24-020141-adr-gaps.md): 初回ADR候補C3の根拠
