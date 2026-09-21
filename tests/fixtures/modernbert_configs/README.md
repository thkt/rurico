# ModernBERTの正規config

`config_validate_official_model_configs` が、対応する4種類のembeddingモデルと
rerankerのconfigをパースし、`Config::validate` で受理できることを検証する。
JSONは以下の固定revisionの `config.json` のスナップショットで、モデル重みは含まない。
取得済みのHugging Faceキャッシュとバイト単位で一致するファイルを使用している。

| fixture | 取得元のconfig |
| --- | --- |
| `ruri-v3-30m.json` | [ruri-v3-30m / 24899e5](https://huggingface.co/cl-nagoya/ruri-v3-30m/blob/24899e5de370b56d179604a007c0d727bf144504/config.json) |
| `ruri-v3-70m.json` | [ruri-v3-70m / 07a8b0a](https://huggingface.co/cl-nagoya/ruri-v3-70m/blob/07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc/config.json) |
| `ruri-v3-130m.json` | [ruri-v3-130m / e3114c6](https://huggingface.co/cl-nagoya/ruri-v3-130m/blob/e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6/config.json) |
| `ruri-v3-310m.json` | [ruri-v3-310m / 18b60fb](https://huggingface.co/cl-nagoya/ruri-v3-310m/blob/18b60fb8c2b9df296fb4212bb7d23ef94e579cd3/config.json) |
| `ruri-v3-reranker-310m.json` | [ruri-v3-reranker-310m / bb46934](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m/blob/bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3/config.json) |

これらは [Issue #299](https://github.com/thkt/rurico/issues/299) が参照する版で、
`src/embed.rs` と `src/reranker.rs` のrevisionに対応する。製品のrevisionを変更するときは、
対象のconfigを取得してfixtureとこの参照を更新する。合成した値で正規configを置き換えない。

このテストと数値境界テストはCPU上の設定検証で、実行時にネットワークやモデル重みを使わない。
通常の `bash scripts/check.sh` に含まれる。対象だけを実行する場合は次のコマンドを使う。
ビルド環境は [CONTRIBUTING.md](../../../CONTRIBUTING.md#テスト) に従う。

```sh
cargo nextest run --locked --lib modernbert::config::tests
```

設定の受理は、実モデルのload/forward成功や出力の有限性・数値一致を保証しない。
特にf32の正の非正規化数を含む極端なRoPE baseの受理は、変換後の正値・有限性の検査結果に限る。
実モデルの検証は既存の [MLXランタイム・smokeテスト](../../../CONTRIBUTING.md#mlx_smoke-smoke-テスト) と区別する。
