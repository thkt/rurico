# Issue #302: 非有限logit・原因chain・互換性の確認

2026-09-21確認。要求は [Issue #302](https://github.com/thkt/rurico/issues/302)、
検証条件は [親Issue #296](https://github.com/thkt/rurico/issues/296) を参照する。
開始commitは `6c0967b1f5f2337a03bb5eac6829f8a8c80c8362` で、着手時のremote mainと一致した。
引き継ぎで指定された追加報告はない。

Issueの初回根拠 `32513da690653a8baf30d2af4ae77d2129838f55` と比べ、
#300で重みの検証・headが変わり、#332でreadbackの純粋処理が
`src/reranker/processing.rs` へ移った。開始版でもsigmoid後の有限性検査と
エラーの文字列化が残っており、本Issueの問題は適用できた。
#300より前のscoreを今回の有限値維持の基準にはしない。

## 飽和の少数例

新しいGPU測定を追加せず、開始commitにある
[#300のホスト比較](issue-300-host-comparison.md#rerankerのlogitとclassifier-bias) と
[保存出力](issue-300-reranker-runtime.txt) の `mode=checked` を再集計した。
モデルは `cl-nagoya/ruri-v3-reranker-310m`、model/tokenizer revisionは
`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`、F32・batch=4・seq=128。
環境と公開4ペアは元の記録を参照する。

logitは `[-4.89451, -5.63552, -3.9008904, -7.0614643]`、scoreは
`[0.007431932, 0.0035561298, 0.019823, 0.0008567868]`。
scoreが0または1となる端点への飽和は0/4件、同点もない。
seed=42/7/42の3試行は同じ4ペアの再ロードで、独立した12入力ではない。
これは過去の実モデル出力の集計であり、変更後のGPU再実行や一般の検索品質・飽和率の推定ではない。

合成入力 `[20, 30, -100, -90]` は現在のf32計算で `[1, 1, 0, 0]` になる。
既存の同点順序テストにこの入力を加え、raw logitの大小ではなくindex `[0, 1, 2, 3]`
となることを検証する。有限値の計算式は変えない。
[同点順序の契約 #40](https://github.com/thkt/rurico/issues/40) を維持し、
raw logitを使う順位変更の採用は #307・#309等の品質評価後の別判断とする。

## consumerと公開型

読み取ったローカルconsumerの対象ソースは以下のcommitと一致した。両checkoutにある
既存のCargo.lock変更は触っておらず、consumer全体をcleanな検証対象とは扱わない。

- [amici `4b2e0de` のeval_harness](https://github.com/thkt/amici/blob/4b2e0de8e0cca04fd454d30bc4c3edf1fe00d5ca/src/bin/eval_harness.rs)
  は `LazyReranker::new(init_reranker)` を使い、`init_reranker` はcache・download・loadの
  エラーを文字列化する。既存の `new` シグネチャは維持するが、この呼出側で失った原因は復元できない。
  chainを使う移行には `with_error` と型付きエラーの伝播が必要。
- [sae `831392f` のsearchテスト](https://github.com/thkt/sae/blob/831392f9f81b38ac50023ae4b74d1d103623245b/src/storage/search.rs)
  は失敗用Rerank実装の3メソッドで `RerankerError::Inference(String)` を構築する。
  依存rev更新時には `Inference { message, source: None }` への変更が必要。

`Inference` / `Tokenizer` / `InitFailed` のtupleからstructへの変更はRustソース互換性を持たない。
variant名・失敗分類・表示接頭辞は維持する。[#190](https://github.com/thkt/rurico/issues/190) の
EmbedErrorと同じmessage/source形式とし、lazyの原因は再呼出しのためArcで共有する。
現在の契約は [README](../../README.md)、構築・matchの移行方法は
[CHANGELOG](../../CHANGELOG.md) と `LazyReranker::with_error` のrustdocに記載した。
consumer自身の変更・ビルドや、全downstreamの互換性検証は今回行っていない。

## 検証範囲

既存readbackテストのInfを受理する期待値は、今回修正する旧動作なので置き換える。
NaN・±Infの拒否、通常の有限score・有限極値・形状不一致・入力順を同じテストで確認する。
各非有限logitから返されたエラーについて、`NonFiniteOutput` の分類に加えて、
従来の表示と `Error::source()` が `None` を返す契約も確認する。
これにより、分類が正しくても表示が変わったり、存在しない原因が付く回帰を検出する。
既存lazyのキャッシュテストは、各メソッドから原因を型で辿り、初期化が一度だけで、
wrapper破棄後も同じ原因が残る検査へ拡張する。文字列closure・並行初期化・空入力の既存検査も維持する。
追加の原因型テストは、MLX Exceptionと実tokenizerのMissingUnkTokenが文字列化や二重box化で
失われる回帰を検出する。表示の検査は別に行い、chainの成否を文字列一致で判定しない。

旧Inf受理以外の検出条件は削除しない。モデル取得やGPU実行を増やさず、既存の
通常checkに含まれるテストを使う。実行時間の短縮や性能改善は主張しない。
`mlx.rs` のeval/readbackを囲むclosureと、その後の `release_inference_output` の順序は維持する。
forward途中のcleanup改善は [#303](https://github.com/thkt/rurico/issues/303) の範囲であり、
今回のテストをGPU解放順・メモリ量の保証には使わない。
標準check・CIはホストの実行に残す。実モデルのignored検証を行う場合は既存手順を使い、
結果と未実行条件を記録する。
