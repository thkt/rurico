# Issue #306: 計測APIとベンチ記録の検証

[Issue #306](https://github.com/thkt/rurico/issues/306) に基づき、optionsを渡した一度の推論から
embeddingと利用可能なmetricsを受け取るAPI、および各試行を再集計できるJSONL記録を追加した。
MLX専用構成、bucket・cache cleanup・pauseの位置は維持し、ADR-0006/0007/0009/0014の判断は変更していない。
APIと区間の定義はrustdoc、実行手順とrecordの読み方は
[CONTRIBUTING](../../CONTRIBUTING.md#options付き推論の計測issue-306) を参照。

## 修正後の短時間比較

2026-09-27、同じApple M3 / macOS 27.0 / Rust 1.98.1 / Metal 32023.921で、
`cargo build --locked --features smoke --bin mlx_smoke` のdebug binaryを固定し、
`measure-overhead` を実行した。これは既存W2の先頭3文書と対応する既存fixtureを使う短い比較で、
通常／計測API × batch／sequential × default／非defaultの全8組合せを同じbinary・入力で実行する。
新しいfixtureや速度閾値は追加していない。長文・100文書の性能へは外挿しない。

本作業のビルドと他のGPUテストを終えてから実行し、10秒間、コンパイラ・ビルドツール・既知のモデルテストprocessが
存在しないことを確認した。測定中も0.25秒間隔で監視し、競合を検出したら今回の測定processを終了する条件にした。
7.11秒の実行中25観測に競合はなく、全組合せを最後まで実行できた。
監視対象はcargo/cargo-clippy/cargo-nextest/rustc/cmake/make/ninja/c++/ld/lld/clang系と、
mlx_smoke/probe_embed_smoke/probe_reranker_smoke/rurico系テストprocess。
[開始前観測](issue-306/isolated-final/preflight.json)、[測定中観測](issue-306/isolated-final/load.jsonl)、
[開始・終了と終了コード](issue-306/isolated-final/outcome.json) を保存した。
OSや通常アプリのGPU利用をゼロにする測定ではなく、監視間隔未満の活動や未知のprocessまでは保証しない。

33試行（準備8・warm 24・空入力1）が成功した。
default batchは `[3,128]` の1 forward、非defaultは `[2,128]` と `[1,128]` の2 forward・2 pauseを確認。
通常／計測の全呼出しでfixtureの文書順・chunk構成・数値一致を検査し、計測時はshape・token数・pauseも検査した。
[raw](issue-306/isolated-final/raw.jsonl) から [summary](issue-306/isolated-final/summary.jsonl) を再生成し、8集計の完全一致を確認。
[source manifest](issue-306/isolated-final/source-manifest.json) は測定時のsource・fixture・lockfileと測定前後で一致し、
[モデル内容](issue-306/isolated-final/model-proof.json) も固定revisionのhashへ再照合した。
raw内のcommitは開始版、dirty=trueで、実装差分とbinaryのhashを別に保持する。
報告と証拠ファイルは測定終了後に追加したため、掲載後の作業差分hashとは異なる。

短時間モードでは、workloadの選択後に生成関数を呼ぶ。使用しないW1/W3の生成を避け、
採用するW2はprofileごとに生成する。先頭3件・入力hash・fixture照合・warm-up・交互順は維持する。
この準備処理は推論wallの外側で、生成方式の変更による速度・割当量の改善幅は測っていない。
変更前の [短時間record](issue-306/isolated/raw.jsonl) と
[そのmanifest](issue-306/isolated/source-manifest.json) も履歴として保持する。
現行版との差は `src/bin/mlx_smoke/records.rs` の1ファイルであり、以下の表には混ぜない。

単位はms。各セルは **中央値 [最小, 最大]**、各n=3。モデルload・初回推論・warm-upは除外。

| options | 呼出し | 通常API | 計測API | 中央値の差（計測−通常） |
| --- | --- | ---: | ---: | ---: |
| default | batch | 106.029 [103.713, 106.432] | 105.747 [105.723, 105.819] | -0.282 |
| default | sequential | 131.789 [131.676, 135.555] | 134.107 [133.756, 137.231] | +2.318 |
| 256 + 1ms | batch | 121.475 [121.232, 123.013] | 124.511 [120.790, 124.829] | +3.036 |
| 256 + 1ms | sequential | 135.141 [134.197, 137.901] | 137.153 [135.419, 139.434] | +2.011 |

この入力で観測した追加差は約-0.28〜+3.04msだった。負の値を高速化と解釈せず、
3試行と実行順・OS schedulingによる揺れを含む比較として扱う。厳密な追加コストの上限や、
長文・多数forwardでのコスト、release性能は未確認。RSS/Metalメモリは未計測のままnullとした。

forward一覧は内部metricsから公開snapshotへ所有権を移し、不要な割当て・コピーを除去した。
上表は修正後の通常／計測APIの比較であり、修正前の負荷が重なった値との差を、この変更だけの効果とはしない。

## 修正前の全workload記録

以下は負荷が重なった履歴であり、現在の性能比較の完了証拠には使わない。
短時間比較とは入力サイズと実装版が異なるため集計を混ぜず、長文を含む機能検証の履歴として保持する。

2026-09-27、Apple M3 / Mac15,3、macOS 27.0 (26A428)、Rust/Cargo 1.98.1、
Xcode 27.0、Metal 32023.921で実行した。次のdebug buildを直後にコピーし、固定した実行ファイルを使った。
release buildの性能評価ではない。依存の一括更新やfixtureの再生成は行っていない。

```sh
cargo build --locked --features smoke --bin mlx_smoke
RUST_LOG=warn <直前にビルドしたmlx_smoke> measure-records > raw.jsonl
<同じmlx_smoke> summarize-records raw.jsonl > summary.jsonl
```

記録はcheckout外へ出力し、終了後にこのディレクトリへコピーした。
開始commitは `cbb9068b06d957bcf9496e431cc6092416918d13`、未commitの実装差分を含むため `dirty=true`。
[raw.jsonl](issue-306/raw.jsonl) の差分・untracked・lockfile・実行ファイルhashと、
[source-manifest.json](issue-306/source-manifest.json) のRust source・fixtureのhashで対象を識別する。
修正開始時にはsourceとfixtureのhashがすべて一致した。所有権移譲と短時間比較モードを追加した後は、
metrics・embedder・metricsのテスト・smoke binary・recordsの5ファイルが異なるため、
このmanifestと実行ファイルhashを現行実装の測定証拠として扱わない。
記録・説明文の追加後も作業差分hashが変わる。履歴のmanifestと生記録は更新しない。
[model-proof.json](issue-306/model-proof.json) は固定revisionの公式ETagとローカル内容の照合結果。
model/config/tokenizerはrevision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` に一致した。

生記録はmodel load 1件、推論・空入力検証97件、集計24件の計122行。
[summary.jsonl](issue-306/summary.jsonl) はモデルをロードせずrawから再生成し、埋め込まれた24集計と
JSON値の完全一致を確認した。全warm groupはn=3で、初回推論・warm-up・空入力は集計対象外。
model loadは0.195秒だったが、cache lookupとtokenizer検証は区間外で、OS cacheも消していない。

### 全workloadで確認した契約

97試行すべて成功。defaultと `token_budget=256, forward_pause=1ms`、通常／計測、
batch／singletonの連続呼出しで、既存Phase2 fixtureに対して文書順・chunk構成・数値を検査した。
許容値は既存のcosine ≥ 0.99999、最大絶対差 ≤ 1e-5。
これは既存fixtureとの退行検査であり、公式実装との比較や検索品質の評価は#307の範囲。

計測値のchunk数、実forward shapeとpadded token数、pause回数、要求時間以上のsleepを検査した。
batchのforward数はdefaultでW1/W2/W3が1/1/2回、非defaultで3/50/8回だった。
非defaultのpause回数も3/50/8回。W2はdefaultの `[100,128]` 1回から `[2,128]` 50回へ分割された。
空入力は出力0件・forward 0回・pause 0回。未分離のtokenizationは `null`、通常APIのmetricsも `null`。

### 全workloadの通常APIと計測APIの観測値

単位は秒。各セルは **中央値 [最小, 最大]**、各n=3。外側wallは一連のAPI呼出しを測り、
JSON化とfixture比較を含まない。sequentialも同じoptionsをsingleton batch呼出しへ渡している。

| options | workload | 呼出し | 通常API | 計測API |
| --- | --- | --- | ---: | ---: |
| default | W1 | batch | 15.892 [15.732, 19.072] | 15.173 [14.806, 16.926] |
| default | W1 | sequential | 17.990 [15.138, 18.159] | 15.085 [15.079, 17.483] |
| default | W2 | batch | 3.104 [3.074, 3.130] | 3.109 [3.060, 3.226] |
| default | W2 | sequential | 4.626 [4.557, 7.824] | 4.667 [4.525, 8.548] |
| default | W3 | batch | 3.625 [3.567, 4.098] | 3.663 [3.598, 3.818] |
| default | W3 | sequential | 4.101 [3.771, 4.521] | 4.085 [3.966, 4.526] |
| 256 + 1ms | W1 | batch | 15.744 [15.693, 15.829] | 15.176 [15.069, 15.781] |
| 256 + 1ms | W1 | sequential | 15.787 [15.762, 15.963] | 15.803 [15.337, 16.171] |
| 256 + 1ms | W2 | batch | 6.804 [4.436, 7.101] | 4.109 [4.004, 4.782] |
| 256 + 1ms | W2 | sequential | 6.011 [5.805, 7.428] | 4.880 [4.783, 9.863] |
| 256 + 1ms | W3 | batch | 4.401 [3.816, 4.404] | 3.941 [3.578, 4.188] |
| 256 + 1ms | W3 | sequential | 3.723 [3.671, 3.841] | 3.947 [3.893, 4.220] |

通常経路にも従来のphase計測があるため、この差は詳細収集と公開snapshotの追加コストの観測であり、
計測なしのGPU kernel時間との比較ではない。

開始時にビルドprocessがないことを確認し、この作業のビルド・GPUテストは直列にした。
ただし途中で別作業のコンパイルが始まり、全体の比較は隔離条件を満たしていない。
[build-load.jsonl](issue-306/build-load.jsonl) は5秒間隔のprocess名・CPU使用率・直前の完了sequence。
160観測中31観測でビルドprocessを検出し、default W2の後半、default W3、非default W1のwarm-upと重なった。
対象processはcargo/rustc/clang/clang++/cmake/c++で、5秒未満の活動や他の負荷の不在を保証しない。

全試行を削除・選別せず掲載した。ビルドprocessが観測されなかったdefault W1でも、batchの中央値は
通常15.892秒・計測15.173秒（差 −0.718秒、約 −4.5%）だが、範囲が重なり、各3試行しかない。
追加計測が高速化したとは判断できず、追加コストの安定した推定値や上限もこの測定からは確定できない。
非default W2/W3の期間にもビルドprocessは観測されなかったが、ばらつきが大きく同じ制約がある。
この履歴からビルドが観測されなかった部分だけを選別して採用せず、修正後の比較は上記の別実行に基づく。

各callのRSS・Metalメモリ・GPU kernel時間は未計測。OS cache未消去、毎forwardのbuffer/compile cache削除、
同一process内での実行順、3試行という小標本、debug buildも比較上の制約として残る。
本Issueは性能向上率を完了条件とせず、将来の比較で使う計測API・記録・検証を整える。

## テストと既存報告の訂正

所有権移譲の修正前後のホストの `bash scripts/check.sh` は成功し、nextest 379件成功・29件skip、doc test 2件成功、
check/clippy/fmt成功を確認した。最終版にも標準checkと独立評価を適用する。
smoke binaryの単体テスト14件はworkload生成の修正後も成功し、モデル不要の再集計CLI統合テスト1件も短時間比較追加後に成功した。
単体テストはfallbackの一度だけの委譲・optionsとエラーの維持、sub-msの保持、未計測の区別、
原recordと集計の対応、条件やwarm-upの混入、実行順を対象とする。
既存のplanner・cleanup失敗注入テストも標準checkで引き続き実行する。

既存のignored `smoke_measure_baseline` は失敗した。変更前mainでも `measure-baseline` を実行し、
同じ3種類の閾値違反を再現した（[診断記録](issue-306/legacy-baseline.json)）。
W2/W3のpadding率は両版とも6.7724867/1.6299626で、従来閾値1.2を超える。
R²はmainが0.9218、実装版が0.9208で、従来閾値0.95未満だった。閾値やfixtureは変更していない。
mainのsequentialは `embed_document`、実装版はoptions付きsingleton batchで、warm-upと順序も違うため、
両版の時間を同条件の性能比較には使わない。今回、旧ベンチの性能ゲート通過は確認できていない。
失敗時の各workload・readback shape・linearity・診断の出力項目とshapeの算術関係は別途確認した。

独立評価と同じheadのCIはPRで確認する。通常checkの成功を実モデル検証の代わりにせず、上記の別実行を根拠にする。

[Phase3](phase3_result.md) の旧「Phase2」列は同じ開始版の
[Phase1](phase1_baseline.md#batch-vs-sequential) の値だった。
[Phase2](phase2_result.md#batch-vs-sequential) の16,565/617/2,696msへ訂正し、
誤った改善率とpooling単独への寄与説明を取り除いた。歴史的観測を今回の同条件比較へ流用していない。
