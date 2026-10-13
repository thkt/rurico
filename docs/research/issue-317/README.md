# Issue #317: query batchと共通query再利用の比較

[報告と採否提案](report.md)に、試作の範囲、根拠の版、互換性と修正版の固定モデル比較を示す。
実モデル比較は測定版に固定した証拠として残す。未使用の内側wall計測を除去した後のcoverageは96.12%で通過した。
[最新のホスト確認](report.md#内側wall除去後のホスト確認2026-10-13)から、標準check・独立評価・新headの全CI・公開後確認へ進む。
要求・許可の正本は [Issue #317](https://github.com/thkt/rurico/issues/317) と合意記録。
公開API・製品のtokenizer・consumerは変更しない。

## 通常検証

試作は `cfg(test)` と既存の `smoke` featureに限定した。
モデル不要のspy・prefix/truncate・pair処理、候補のbucket/forward接続、
lock/metricsと計測protocolの検証は、設定済み
`cargo fetch --locked` / `bash scripts/check.sh` のlibtest対象に含まれる。
Pythonの記録検査も既存の `scripts/test_*.py` discoveryで実行する。
check・CI・captureなしの契約を変更しない。

## ホストで追加比較する

Apple Silicon、対応するXcode/Metal、固定310mモデルの既存HF cacheを使う。
[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)に従い、モデル不足時は登録済み
`mlx_smoke` の `prepare-cache` を使う。今回のlibtestからdownloadを起動しない。
他のGPU検証・ビルドと重ねず、同じ変更版をreleaseでビルドする。

```sh
CARGO_TARGET_DIR=/tmp/rurico-317-host-build cargo test --locked --release --lib --features test-support,test-mlx,smoke --no-run --message-format=json > /tmp/rurico-317-build.jsonl
```

`compiler-artifact`の `target.name=rurico`、`profile.test=true`、`executable`を持つ
libtest binaryを一つ選ぶ。古いbinaryやsandbox内の局所確認用binaryは使わない。
担当AIが開始commit `a5471c83fd4f48255a71ee77b223968c2cf1015b`、未commit差分、
Cargo.lock（SHA-256 `d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7`）、
ビルドargvとbinary、Rust/Xcode/Metal、機種・RAM・OS・電源・熱・背景負荷を記録する。
初回MLXビルドのCMake取得はCargoのoffline指定とは別である。

実行前に親 [#296](https://github.com/thkt/rurico/issues/296) の本文・updatedAt・hashと
実行・保存の条項を取得する。報告の旧取得版を現在の原文確認へ読み替えない。
モデルのsnapshot revision・6ファイルのhash・F32 weight headerを確認する。
embeddingは `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`、
rerankerは `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`。
runnerが渡すsnapshotと、実際に製品のcache lookupが返す3ファイルのpathもRust側で照合する。

次のcontextは書式例であり、測定結果ではない。

```json
{
  "machine": "実機の機種・chip",
  "ram_bytes": 25769803776,
  "os": "実際のOS版",
  "rust": "rustc --versionの結果",
  "cargo": "cargo --versionの結果",
  "xcode": "xcodebuild -versionの結果",
  "metal": "xcrun metal --versionの結果",
  "build": "上記ビルドのargv、profile、binaryへの対応",
  "load_conditions": "電源、熱、背景負荷、他GPU測定の有無",
  "embedding_snapshot": "/absolute/HF/snapshots/18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
  "reranker_snapshot": "/absolute/HF/snapshots/bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3",
  "parent_issue_evidence": "親Issueの取得版・hash・実行条件を保存した証拠への参照"
}
```

```sh
python3 docs/research/issue-317/run.py /absolute/path/to/new/rurico-libtest /tmp/rurico-317-context.json /tmp/rurico-317-evidence-NEW --seconds 1800
```

`--seconds` はこの調査の実行予算で、製品のhard limitではない。
例は30分。担当AIは実際のホストの実行枠内で指定する。変更が必要な合意済み上限は緩めない。
runnerはモデルの取得・サーバー・ブラウザーを起動しない。
予算到達・非ゼロ終了・欠損記録・内容変更では部分証拠を保ち、後続比較を停止する。
所有process groupの回収とRSS/pressure/swap観測は既存#318 runnerの関数を再利用する。
RSSに停止閾値を新設しない。OS観測費用と約0.1秒のsampling間隔があり、
瞬間peakやGPU全量を捉える保証はない。SIGKILL・ホスト停止後の回収と最終記録は保証しない。

## 実行範囲と結果の解釈

queryの単発・32件の複数query・長短/空の境界入力と、pairの単発・32件の交互query・
256件の共通query・上限/空の境界入力をそれぞれ比較する。
共通queryの利益を複数queryに一般化しない。256件から5000件への推定もしない。
入力は#307の公開入力と明示した合成入力。pair方式は直前のquery Encodingを1件だけ保持する。
交互queryではcache hitがなく、その不利益も測定へ含める。

baseline/candidateは別processでロードし、ケースごとに先行方式を交互に変える。
各processで最初の呼出しと3回のwarm呼出しを記録する。
ここでcoldはロード・token一致の事前確認後の最初の推論であり、
OSのfile cache、tokenizer、コンパイル環境すべてがcoldという意味ではない。
ロード時間、実tokenize時間、実forward形状/数、wall、Metal peak、出力・順序を保存する。
queryのbaselineも製品の単発経路で成功したtokenize/forwardをtest専用hookから観測し、
件数からforwardを推定しない。既存#306のmetrics形式を候補に再利用する。
pairでは準備区間にEncoding生成・IDs/maskコピー・製品truncateを含む。
queryでは既存単発経路と候補のtokenize区間の定義が共にprefix付加〜truncateである。
候補のunpadded条件検査も候補時間へ含む。wallの範囲は各方式の結果返却までである。

`context.json`、`source.patch`、追加試作source、各processのJSONL/stdout/stderr/status、
`monitor.json`、`summary.json`をcheckout外の新規ディレクトリへ保存する。
source・lock・binary・モデル・contextは前後hashを照合する。
`complete.json`は収集完了であり、受入・改善の証明ではない。
集計のcold値とwarmの最小・中央値・最大、processごとのRSS最大を原資料から照合する。
process RSSはロードと計測後の互換性検証も含み、推論呼出しだけのpeakへ読み替えない。
Metal peakは各呼出し前にresetするが、ロード済みweight/cacheを含む絶対値である。
少数反復とケース別の順序交替だけで安定した速度差を断定しない。

token列、特殊token・境界・overflowは既存pair encodingと一致を要求する。
共通pairのscoreは同じforward形状での完全一致を要求する。single/commonは製品公開rerankの
1回の結果で全index順と全scoreを確認し、multiple/boundaryは公開score_batchで照合する。
queryは変更するbatch shapeに対し、#307の同一token診断
（cosine ≥ 0.99999、最大絶対差 ≤ 1e-5）で比較する。
この既存調査の診断を製品の新しい品質許容差にはしない。非有限・欠損・次元違いは失敗する。

担当AIが実測値と適用条件を[報告](report.md)へ追記し、採否提案を更新する。
生log・個人pathを一括公開せず、公開可能な要約と版・失敗・未確認事項を残す。
変更文書を含む既存の独立評価へ戻す。通常check/CIの成功と、この追加比較の成功を区別する。


旧測定の `results.json` は修正前の版に固定した証拠として保つ。
修正版の測定・CI同条件coverageの結果と版の対応は
[修正版ホスト再検証](report.md#修正版のホスト再検証2026-10-13)で確認できる。
再実行でも既存の閾値・除外・実行上限は変えない。

修正版の実測値は[results-revised.json](results-revised.json)、旧版は[results.json](results.json)に保存する。
両版の適用範囲と独立評価後の修正理由は[報告](report.md#修正版のホスト再検証2026-10-13)を参照。
