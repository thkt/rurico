# Issue #315 構成識別の設計例

[設計報告](report.md)に比較、公開API案、移行案と検証の限界をまとめる。
これは[Issue #315](https://github.com/thkt/rurico/issues/315)の調査成果であり、採用済みAPIではない。

`spec.py`は保存記録の照合とfingerprintの実行可能な参照例、`document.json`と`fts.json`は公開用の合成構成である。
モデル名・revision・hash・consumer設定は架空であり、実索引へ付与してはいけない。
`test_spec.py`はその設計例を検証する。Python標準ライブラリだけを使い、推論もモデル取得も行わない。
`pair()`と`Snapshot`は入力を入れ子まで複製する。返り値の編集や並行変更を防ぐ仕組みではないため、
利用時の所有権と設定固定の条件は[報告の利用例](report.md#consumerの最小利用例と移行)に従う。
`acquire`のMatchは取得根拠・世代の確認だけを表す。副作用の前に、同じ固定入力を
ID生成（新索引では`expected_kind="embedding"`を指定）または照合で完全検証する。
製品のRust/MLX構成、Cargo.lock、既存fixtureは変更していない。CPU推論backendも追加しない。

checkoutのルートから実行する。

```sh
python3 -B -m unittest discover -s docs/research/issue-315 -v
```

この設計例の検証は標準checkに自動登録されていないため、調査内容を変更した際は上のコマンドも実行する。
Python 3.14.7での結果は[報告の検証節](report.md#検証結果と引き継ぎ)を参照。
標準checkは従来どおり、GPUを利用できるホストで実行する。

```sh
cargo fetch --locked
bash scripts/check.sh
```

標準checkは`test` job相当であり、`coverage`・`security`・`zizmor`は別に同じheadで確認する。
詳細は[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)を使う。
設計例の成功をMetal実測、Rust APIのコンパイル検証、検索品質の証明とは扱わない。
撮影は不要。変更文書も独立評価の対象とする。調査成果の公開は製品APIや保存契約の採用を意味しない。
