# Issue #313 重複候補の契約案と集約allocationの比較

既定は現在の加算契約を維持し、単一query・単一source内で同じdoc/chunkを繰り返す場合のbest rank統合を、明示的な追加入口の候補として提案する。統合・拒否・wire変更は未採用で、製品には実装していない。別source、別chunk、parentの`None`とchildの`Some(_)`を同一視しない。

修正前の内部比較では、固定slotはallocation回数を減らさず、要求byte総数と追加live heap peakが小さくなった。固定slotの重み取得を呼出し内で一度にまとめた現在版でも、36 process・252 callとRSSを再測定した。修正前の数値は履歴として分ける。所有権を渡すidentity候補は、準備済みVecを移譲する区間では追加allocationがなかった。今回のTopK選択候補は小入力・大量入力ともallocationが増えたため、現候補の採用は推奨しない。ホストで36 process・252 callとprocess peak RSSを取得した。外部作業の停止と負荷隔離は未確認で、時間差を候補による速度改善とは判断しない。製品方式は未採用で、最新統合版の全体checkと独立評価はこの報告の更新後に行う。

現在の修正は公開head `64c80cc3` に#379マージ後のmain `e1ba0ed7` を通常mergeした未commit版である。現在の確認・残る検証は[今回の状態](#379マージ後のmain統合)を参照する。以下のR1測定と01dd1c6統合の記録は履歴であり、旧check・accepted・CIを今回の成功へ読み替えない。

## 根拠の版と適用条件

要求・権限は [#313](https://github.com/thkt/rurico/issues/313)（ユーザーから渡された更新時刻2026-10-07T17:05:22Z）と [親#296](https://github.com/thkt/rurico/issues/296)を正本とする。調査結果の提示までが今回の範囲で、方式採用とconsumer移行は次の判断である。

初回開始版と当時のローカル`origin/main`は `24725a72be44300afc24186b82b14bcb3f5f9d3d`。初回のホスト準備でremote mainも同じcommitと確認した。root Cargo.lockのSHA-256は `743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。今回取り込むmainでも変更はない。2026-10-08見直し版`c8f250d60a5afb9944b9008d22ad6f4dda2d7103`と開始版の`src/retrieval.rs`・`src/retrieval/tests.rs`に差はなかった。#298の修正はそのまま利用し、除算・合算・recency・平均を再実装していない。

初回根拠版 `32513da690653a8baf30d2af4ae77d2129838f55` から開始版へのretrieval差分は、quotient/totalsの非有限除外、recencyの有限additionとchunk tie、overflowを避ける平均・source平均の変更だった。ADR-0004にこの版間の差はない。初回の倍増観測は今回の小例で再確認し、旧版の数値境界を現行の契約として利用しない。

| 出典 | 状態・適用範囲 | 今回の扱い |
| --- | --- | --- |
| [ADR-0004](../../decisions/0004-retrieval-and-rerank-pipeline-contract-for-rurico.md)、開始版 | accepted、Stage 2/3の型と位置、source contribution保持 | 公開trait/API/JSONを維持。整列済みStage 2入力での保証と直接呼出しを区別する |
| [ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)、開始版 | accepted、検索評価の所有者はamici | 検索評価をruricoへ移植しない。ADRのbaseline一致の期待を今回の実測へ読み替えない |
| [#307報告](../issue-307/report.md)、blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 共有済み引き継ぎ版。数値はrurico `d0639cc8…`、検索はamici `547f9ee2…`の限定観測 | 開始版・現在の報告blobとも一致し、根拠の改訂はない。その有限入力・reference compositionの観測を今回の重複案の採用許可や品質保証に使わない |
| [README](../../../README.md#aggregator-の使い分け)、[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)、開始版 | 現行操作・MLX全体check・ignored実モデルの区別 | 今回もroot checkはホスト。CPU部分検証の成功でMLX/実モデルの検証を代替しない |

親Issueの旧check例は履歴であり、現在の契約は`bash scripts/check.sh`に従う。現行checkは`test-support,test-mlx,smoke`を含む。旧記録、#307の数値、検索baselineや許容差を変更していない。初回開始入力にはfindings/assessments/handoffがなかった。今回のホスト復帰では評価R1の3指摘と追加証拠を読み、下記のassessmentsとhandoffへ反映した。独立評価acceptedを自己確認で代替しない。

## 現行の順序・同点・source map

[製品source](../../../src/retrieval.rs)と[固定期待値の検証](../../../src/retrieval/tests.rs)を照合した。Stage 2は有限なscore/source contributionを返し、scoreの`total_cmp`降順、doc_id昇順、chunk_id昇順に整列する。Stage 3のpipeline保証にはこの入力条件が必要である。

| Strategy | 直接渡した未整列入力 | 同点とsource map |
| --- | --- | --- |
| Identity | 入力順・chunk・score・mapをそのままclone | 再整列・統合をしない |
| Dedupe | parentの最初の出現を入力順で残す。後続の高スコアを選ばない | 最初のmapだけを残しchunkを`None`にする |
| MaxChunk | parentごとに数値比較`>`で最大を選び、score降順・doc_id昇順に出力 | 同じ数値なら最初のchunkのmap。負けたchunkのsourceを合算しない |
| TopKAverage | parent内をscoreの`total_cmp`降順で安定整列し上位kを平均。出力は平均降順・doc_id昇順 | 同点のk境界は入力順。選択chunkのsourceだけを同じ件数で平均。欠落はゼロ、全選択chunkで欠落したsourceはmapにも存在しない |

MaxChunkでは`-0.0`と`+0.0`が数値的に等しく先頭を残すが、TopKでは`total_cmp`により`+0.0`を先に選ぶ。両者のk=1相当の処理を無条件に置換できない。recencyはscoreだけを変えるため、source合計がscoreと一致する前提も置けない。空mapはsource情報不明であり、sourceの寄与がゼロだったという証拠ではない。

現在の製品テストで、Dedupeが後続の高スコアで先頭を置き換えないこと、MaxChunkの複数parent同点がdoc_id昇順になり先頭chunkのsourceを保持すること、TopKが未整列入力を並べ替え、異なるchunk ID/sourceの同点k境界で入力順のsourceを選ぶことを固定期待値で確認する。最新mainのPR #374のテストは今回のcheckoutへ取り込み、入力と期待値を照合して重複例を統合した。signed zeroの検証は数値比較とtotal_cmpの違いを守るため保持する。調査用TopK選択候補にも独立したdyadic固定値があり、候補と製品baselineの一致だけに依存しない。期待値を製品のsort/mergeから生成していない。TopKの極値・非有限値・欠落source、RRFのoverflow、recencyのoverflow/tieは既存回帰検証を保持した。

## consumer入力の照合と限界

2026-10-08にローカルGitの以下のcommitのsourceを読んだ。関係ファイルの作業差分はなく、checkoutや依存を変更していない。これらのcommitの公開状態は今回再確認していない。

- [amici `4b2e0de8…`のpipeline](https://github.com/thkt/amici/blob/4b2e0de8e0cca04fd454d30bc4c3edf1fe00d5ca/src/eval/pipeline.rs)、blob `9ea21ca667f5337434b78e14de3ed5f0c9f14699`: FTSはparentごとに一行でchunk=`None`、Vectorは各chunkの行を`Some(chunk_id)`として列挙する。sourceごとに列挙位置をrankにし、両リストを連結する。複数Vector chunkやFTS parentとVector childの共存は正当な入力で、docだけの重複排除は誤り。merge後にAggregatorを呼び、kにtruncateするため、整列前提は実際の切り捨てに関わる。
- [recall `5a9fab0c…`のadapter](https://github.com/thkt/recall/blob/5a9fab0c9e69aee084ba4e273a9f0c80cb65ff9a/src/hybrid.rs)、blob `67a1385bf20c5a76e50cf17cc533c428a03ccfe2`: 両sourceの列挙rank、chunk=`None`、raw score=`0`でWeightedRrfへ渡す。[検索source](https://github.com/thkt/recall/blob/5a9fab0c9e69aee084ba4e273a9f0c80cb65ff9a/src/search.rs)、blob `737076d433f18f5650453fdc9ce18aa57089009f`はFTSとVectorをそれぞれsessionでGROUP BYする。確認した通常経路では同source内のsession重複は検索段階で除かれる。
- [sae `831392f9…`の検索source](https://github.com/thkt/sae/blob/831392f9f81b38ac50023ae4b74d1d103623245b/src/storage/search.rs)、blob `9b997bad4d3f81808e96363796d9ff8d501de04f`: sourceごとの列挙rankを使う独自RRFであり、WeightedRrfを呼ばない。ruricoに重複入口を追加しても自動移行されない。recencyもrerank後の別契約であり、今回の内部候補へ移さない。

#307固定版amici `547f9ee2ed734a2eab316fdbd62f194849875ee4`をホストで取得した。archive SHA-256は `5d994e7a799b97ae27c63184709d7afde2f244a707ab074b77c73a52d4397d7a`、pipeline SHA-256は保存済みmanifestと同じ `afac0ca54715f18ac2b223cfca454c19e8f733a3a0917fcb1aebfd915a06f1f7`。その[固定pipeline](https://github.com/thkt/amici/blob/547f9ee2ed734a2eab316fdbd62f194849875ee4/src/eval/pipeline.rs)でもFTS parentは`None`、Vector childは`Some(chunk_id)`で、source別の列挙rankを付け、merge後にaggregate・truncateする。consumerの製品コードと元のlockは変更していない。

この静的照合は重複候補の意味を確認する根拠で、#307の品質実測を新しい重複方式の保証へ広げるものではない。yomu、全custom consumer、実データやmulti-query/prefix合成は未確認で、方式採用・移行前に対象を確定する必要がある。

## 重複契約の未採用案

同source/doc/chunk/rankを2回渡すと、現行RRFは各候補の寄与を加算する。raw scoreはRRFに使われない。rankが異なる同source/doc/chunkの再出現も加算される。候補の入力順はrankを再計算せず、同じ加算項でも一般の浮動小数点総和の最終bitは入力順で変わり得る。固定期待値の再現ではdyadicな`rrf_k=4, rank=0`を使い、同sourceの2寄与`0.25+0.25`と別sourceの`0.25`を、score=`0.75`、source map=`{fts:0.5,vector:0.25}`として確認した。別chunkは別hitで`0.25`になる。逆順でもこの例の期待値は同じである。

| 案 | 重複のキーと処理 | source contribution・API・JSONへの影響 |
| --- | --- | --- |
| 現行加算 | すべての候補を加算。rankもそのまま使用 | 重複分がsource subtotalとscoreの双方に入る。既存API/JSONの変更なし。独立した検索signalの合成にも使われ得るため、無断で変更しない |
| best rank統合 | 単一query/sourceの`(source, doc_id, chunk_id)`ごとに最小rankだけを残す。同rankでは先頭を残す | sourceごとに1寄与になる。同docでも別source/chunkは残る。既存shapeを保持しても数値・順位は変わる。追加入口での明示選択とconsumer側合意が必要 |
| 拒否 | 同じ上記キーが再出現したら、異rankでも最初の重複位置を返して処理を止める | mergeの既存`Vec`返却では拒否理由を表現できず、Result付き追加入口等が必要。成功時JSONは維持できるが、失敗経路と移行は未合意 |

現在の通常consumerで繰り返しが必要な根拠は確認できなかったが、custom経路まで不正入力とは断定できない。推奨は加算を維持し、単一検索リストの再送に限ったbest rank入口を次の採用候補にすること。複数queryや複数prefixのsignalを統合するなら現在のsource enumだけでは区別できず、今回の候補を適用しない。拒否は診断用候補で、全consumerの既定にしない。

## 修正前の内部比較の条件と部分結果（履歴）

調査候補は[probe](probe/src/main.rs)だけに置き、製品の実行処理は変更していない。製品source・既存テストを直接参照するCPU用crateであり、製品のCPU依存分離の採用ではない。モデル・tokenization・品質評価を測定していない。

同じ合成入力・binary・standalone release profileで比較した。smallは10 parent × 3 chunk（30 hit、60 candidate）、largeは1000 × 100（100,000 hit、200,000 candidate）、FTS/Vectorの2source、k=3。複数parent、未整列・同点score、異なるsource contributionを含む。入力作成・所有権移譲前のVec準備・出力dropは時間区間外で、両identity経路には同じ準備済みVecを渡す。所有権候補の値はAPI呼出し区間の差であり、入力作成を含むend-to-end費用ではない。

[部分context](results/partial-context.json)は開始commit、実行したsource/lock/binaryのhash、Rust/Cargo 1.99.0、macOS 27.0.1 arm64と未確認条件を記録する。source前後一致と大小入力でbaseline/candidateの完全な出力一致を確認した。各variant/workloadを一度warm-upし、同processで7回測定し、呼出し順を交互に反転した。counter付きSystem allocatorを全候補に適用するため、counter費用を除いた製品性能への外挿はできない。機種/RAM、負荷隔離、process RSSはこの部分測定では未確認である。

以下は時間の最小／中央値／最大（μs）とallocation指標の中央値。原数値は[raw](results/partial-cpu.jsonl)と[再集計](results/partial-summary.json)。allocation回数はalloc/alloc_zeroed/reallocの成功を数え、要求byteはreallocの新しいsize全体を含む。peakは測定直前のlive requested heapを基準とする追加分であり、allocator内部領域やRSSではない。

| 入力 | 候補 | 時間 μs（min / median / max） | allocation回数 | 要求byte総数 | 追加heap peak byte |
| --- | --- | --- | ---: | ---: | ---: |
| small | rrf_map | 5.750 / 6.541 / 11.583 | 100 | 24948 | 11672 |
| small | rrf_slots | 6.167 / 7.458 / 10.250 | 100 | 22964 | 10960 |
| small | identity_borrowed | 1.333 / 1.416 / 2.250 | 91 | 5760 | 5760 |
| small | identity_owned | 0.000 / 0.041 / 0.042 | 0 | 0 | 0 |
| small | topk_sort | 2.917 / 3.083 / 5.042 | 36 | 5234 | 3190 |
| small | topk_select | 6.416 / 7.292 / 9.667 | 153 | 16086 | 9158 |
| large | rrf_map | 33387.667 / 34914.541 / 38073.500 | 300033 | 67393140 | 34096904 |
| large | rrf_slots | 29580.167 / 30017.542 / 32508.125 | 300033 | 63198900 | 31999752 |
| large | identity_borrowed | 4315.375 / 4425.917 / 4788.459 | 300001 | 19200000 | 19200000 |
| large | identity_owned | 0.000 / 0.000 / 0.083 | 0 | 0 | 0 |
| large | topk_sort | 3034.208 / 3107.917 / 3831.750 | 8020 | 2583428 | 1108059 |
| large | topk_select | 3226.375 / 3394.875 / 3696.209 | 19041 | 5498832 | 1109576 |

移譲区間の0nsはclock分解能による観測で、ゼロ費用や速度倍率の証明ではない。process RSSは`null`。`/usr/bin/time -l`はsandboxで`sysctl kern.clockrate: Operation not permitted`、終了1になった。完全なrunner測定として扱わず、失敗記録をcheckout外に保持し、その後の直接CPU実行を別の部分観測として保存した。

## 修正前のホストでの完全なCPU観測（履歴）

Mac15,3・Apple M3・24 GiB、macOS 27.0.1で、12組を各3 process × 7 call、合計36 process・252 call測定した。実行全体は13.15328秒（build・test・clippyを含む）。各組21 callの時間とallocation、3 processのRSSを保存した。[context](results/host-context.json)、[原数値](results/host-raw.json)、[再集計](results/host-summary.json)、[完了記録](results/host-complete.json)は初回の部分結果とは別に保持する。外部負荷の停止・隔離は未確認で、時間を因果的な速度改善と判断しない。

| 入力 | 候補 | 時間 μs（min / median / max） | allocation回数 | 要求byte | 追加heap peak byte | process RSS中央値 byte |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| small | rrf_map | 5.625 / 6.958 / 9.667 | 100 | 24948 | 11672 | 2195456 |
| small | rrf_slots | 5.917 / 7.084 / 8.875 | 100 | 22964 | 10960 | 2162688 |
| small | identity_borrowed | 1.291 / 1.542 / 2.250 | 91 | 5760 | 5760 | 2113536 |
| small | identity_owned | 0.000 / 0.000 / 0.042 | 0 | 0 | 0 | 2097152 |
| small | topk_sort | 3.125 / 4.000 / 5.042 | 36 | 5234 | 3190 | 2097152 |
| small | topk_select | 6.459 / 9.125 / 10.667 | 153 | 16086 | 9158 | 2211840 |
| large | rrf_map | 33545.542 / 34697.709 / 36051.250 | 300033 | 67393140 | 34096904 | 169590784 |
| large | rrf_slots | 29953.958 / 30325.458 / 31689.708 | 300033 | 63198900 | 31999752 | 157515776 |
| large | identity_borrowed | 4319.833 / 4425.333 / 5110.958 | 300001 | 19200000 | 19200000 | 88342528 |
| large | identity_owned | 0.000 / 0.000 / 0.083 | 0 | 0 | 0 | 62242816 |
| large | topk_sort | 2877.959 / 3135.792 / 3807.458 | 8020 | 2583428 | 1108059 | 51298304 |
| large | topk_select | 2962.875 / 3175.334 / 3524.750 | 19041 | 5498832 | 1109576 | 51757056 |

RSSは入力準備・ランタイムも含むprocess peakで、測定区間の追加heap peakと同一ではない。所有権移譲の0nsはclock分解能の下限で、ゼロ費用や速度倍率ではない。旧測定前後のsource・lock・binaryは一致した。その後のコメント整理とcfg(test)の担当範囲整理・候補検証追加を経てreleaseを再buildし、測定binary SHA-256 `801f71f4b26e24bbf0250d7f62581e4bffa84d355acf04f2fb16e8d17da3fa8b`との一致を確認した。[最終source確認](results/final-source-check.json)は初回コメント版とは別に保持する。

| 内部候補 | 観測と保守費用 | 現時点の推奨（未採用） |
| --- | --- | --- |
| 固定2slot | Accumulator内のmapをOption付きslotにし、同じ順で加算して有限性を検査する。返却時に既存HashMapを作るためallocation回数は同じ。presenceを失うと欠落sourceとゼロを区別できない。新source追加でindex/match/output列挙の対応が必要になる | 要求byte・追加heap peak減少の候補として保持。修正前の大小入力のRSS・時間分布は取得済み。重み取得をまとめた現在版の比較も下記の別原本へ保存した。負荷を隔離した速度検証と呼出元への影響確認を経て採否を判断。将来source追加の保守費は定性的で、修正時間は未測定 |
| 所有権付きidentity | 準備済みVecを移すだけならcloneの追加allocationがない。現行borrowed traitに置換できず、呼出元が元Vecを不要とする場合だけ使える | 追加入口の候補として保持。chunk再利用やcontext再構成の必要性をconsumerごとに確認してから採用判断。全pipelineの削減量は未測定 |
| TopK選択 | score＋入力indexで上位kを選び、選択列を同じ順でsort。#298の製品平均処理を再利用するため、選択hitをcloneして再groupする費用も含めた。大小入力ともallocationが増え、時間も有利な観測ではなかった | 現prototypeは見送り。選択処理だけを速いと仮定してclone/再group費用を除外しない。将来の直接平均案は別の測定で判断し、今回の数字を流用しない |

今回の内部比較と重複契約の意味変更は混ぜていない。固定slot/TopK候補は同じ入力の現行出力と比較し、best rank・拒否は小さな契約例で別に確認する。JSONは既存の`HashMap<CandidateSource,f64>`、lowercase enum、欠落fieldのserde defaultを維持する。新しいwireや重複排除は採用後の別実装とする。

## 検証の価値と残る作業

既存の3つのchunk集約テストの入力と固定期待値を強め、Dedupeの最大値選択への変更、MaxChunkのparent同点順の変更、TopKへのchunk ID優先の追加を検出する。新しいテスト件数やモデル・待機は増やさず、同じ公開入口と全出力比較を使う。旧入力の0.9等の個別値はdyadic値へ置き換えたため、その値固有の観測は失うが、parentへの折りたたみ、最大値・平均、入力順の保証は残る。独立unique-parentケースの削除で失った正常契約は別PR待ちにせず現在の入力強化で補った。Identityの未整列保持は既存テストで足りるため追加しない。候補の固定値・baseline比較は、製品呼出し前の整列では隠れ得る製品の同点回帰を直接テストする代わりにはならない。

極値・非有限値・欠落source・recency・legacy serdeの既存検証は別の失敗条件を守るため維持する。調査候補の少数の差分検証は、k=0/1/len超過、非有限値・overflow時の製品baselineとの一致、best rank/rejectの区別に使う。欠落sourceの平均は既存の製品検証で確認する。製品baselineは既存の固定期待値で検証しており、候補の一致検査だけで数値契約を証明しない。

初回53件の後、R1修正前の担当範囲整理と候補固定値追加のsourceで54件が成功し、調査crateのclippyとfmtも成功した。報告されたテスト実行時間は0.00秒（丸め）で、実行時間削減量の測定ではない。共有可変状態・ネットワーク・sleepを使わない。反復flake率は未測定で、安定性改善は主張しない。共有fixtureと固定値の保守は増えるものの、選択sourceを誤る現実的な失敗の追加検出に使える。件数やcoverageだけを維持理由にはしない。

初回の広い固定値検証では、MaxChunkの`>`を`>=`にする改変とTopKのchunk ID tie-break追加を検出した。この履歴は最終の担当範囲整理前の結果で、削除したケースの最終成功証拠として流用しない。極値・非有限値・欠落source・recency・legacy serdeの既存検証は保持した。最初のコピー間比較では共有build先のcacheが別コピーのbinaryを再利用したため、その結果を棄却し、build先を分離した。

修正前のCPUホスト測定と固定amici source照合は完了した。固定slotを変更した現在版のCPUホスト測定も完了した。通常checkには調査crate・runner・RSS測定が含まれないため、個別の実行記録も独立評価へ渡す。rootの全体checkと最終の独立評価は残る。

日本語はnatural-japaneseの固定版 `9a78a42964096da509b8f3e011f0085a5f080151` のクイック手順と手動チェックリストで確認した。初回はofflineのuv cacheにsudachipyがなく実行不能だった。R1修正前のホスト更新後の固定版lintは終了0だった。技術比較の語彙反復と読解負荷の情報指摘を確認し、版・数値・未確認条件を維持した。条件・数値・参照は原資料へ照合した。

担当AIは追加測定後の報告をコード・要求・原数値へ照合し、変更文書を既存の独立評価に含める。rootの全体checkは設定通りホストで実行する。UI媒体は不要で、captureなしの契約を変更しない。commit・push・公開・方式採用は行っていない。

## ホスト復帰のassessmentsとhandoff（2026-10-08、R1修正時の履歴）

評価対象 `a6e23a563d78851085f6c3fc8c17df75702e753f0f52c2d4e724f14266134278` の独立評価R1を現在のsourceと照合した。初回停止記録・実装の生応答・成果物diffと、追加ホスト証拠の93ファイルを読み、記載hashの一致を確認した。評価履歴と過去の修正記録は空であり、今回R1-1〜R1-3を初めて修正した。証拠のpassedは実行結果で、受入は引き継がない。旧runの記録は変更していない。

assessments:

- code / R1-1: 固定slotのsource_weights読取りをcandidateループの前にまとめ、同じsource→slot対応から参照する。設定は呼出し中の共有借用で変化しない。欠落重み=0、非有限寄与の除外、候補順の加算、score/subtotalの有限性判定は維持した。製品runtime/API/JSON、#298の処理は変更しない。旧版の252 call・36 processは原数値から再集計して全12組のsummaryと一致したが、現在のprobe sourceとは異なるため現在版の測定証拠はneeds_changes（更新が必要）。旧binaryとの一致を現在版へ引き継がない。現在版の別原本とbinary hashは以下を参照する。
- tests / R1-2: 上記3つの既存テストを独立固定期待値へ強めた。小さな入力で実際の順序・source選択回帰を区別でき、製品経路と候補経路を混同しない。極値・非有限値・欠落source・recency・legacy serdeは異なる不具合を防ぐため維持する。件数を増やさず共有可変状態・sleep・モデル・ネットワークを加えないため、追加の保守費は入力と期待値の照合に限られる。実行時間・flake率の改善は未測定で主張しない。
- documentation / R1-3: Candidate rustdocを、Stage 2のparentごと一件・有限score/source・score降順とdoc_id同点順・TopKのk>0へ限定し、k=0は空結果と明記した。READMEと実装、既存zero-k検証へ照合した。accepted ADRと#307履歴は変更しない。#307の現在blobと開始版blobはともに `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` で、根拠の改訂はない。固定amiciの静的照合と限定品質観測は維持し、採用許可へ広げない。

handoff:

- 現在版のsource hashは[修正source記録](results/repair-r1-source.json)を参照する。担当AIは権限のあるApple Siliconホストで、[既存runnerの測定条件](README.md#ホストで時間allocationrssを測る)を維持し、新規保存先とbuild先で修正版の大小入力・baseline/candidate比較を実行する。旧ログと新ログを混ぜない。期待する証拠はsource/lockの前後一致、binary hash、36 process・252 callのraw/再集計、RSS、機種/RAM・負荷条件である。測定区間・所有権準備費・allocator counter費とRSSの区別を維持する。負荷を確保できない観測から速度改善を主張しない。
- 設定済み `bash scripts/check.sh` は製品check・テスト・doctest・clippy・fmtで、調査crate/runner/RSSを実行しない。captureは不要。追加測定の要約と現在sourceの対応をこの報告へ反映し、R1の全指摘と変更文書を含む独立評価へ戻す。通常checkの予定だけを追加ホスト要求の理由にしない。
- 全consumer・実データ・multi-query/prefixの区別、Vec再利用は引き続き未確認。方式採用・移行には対象を調べてユーザーへ互換性と採用範囲を提示する。今回の静的照合はその判断を代替しない。

今回のCPU部分検証は新規build先で54件成功、clippy成功、fmt成功、差分の空白検査成功だった。実行はモデル不要の調査crateであり、製品MLX全体check・doctestはホストに残る。診断用の一時コピーでは、DedupeをMaxChunkへ置換、MaxChunkのdoc_id同点順を逆転、TopKにchunk ID tie-breakを追加した3改変が、それぞれ強化した固定期待値のassertで失敗した（終了101）。コピーとbuild先は各改変で分離し、現在の製品sourceは改変していない。診断ログはcheckout外の別原本に保存した。この検出力確認は今回の3失敗条件に限り、受入・実モデル品質・測定性能の証拠ではない。既存テストの入力強化により件数とモデル条件は維持したが、変更前後の実行時間を同一条件で比較しておらず、削減効果やflake改善を主張しない。

R1修正後の日本語確認も固定版 `9a78a42964096da509b8f3e011f0085a5f080151` を使い、HEAD一致・作業差分なしを確認した。今回のlintは専用のoffline cacheにsudachipyがなく終了1となり、実行未完了である。同じ版の手動チェックリストで結論、数量、対象版、条件、権限、未確認事項、参照を照合した。旧版のlint成功を今回の更新へ引き継がず、変更文書を次の独立評価へ渡す。

## R1修正版のホストCPU観測

既存runnerを新規保存先・build先で実行し、36 process・252 call、全12組のRSSを取得した。全体16.685秒にはbuild・54テスト・clippyを含む。source/lockの前後一致を確認し、binary SHA-256は `af3e40373e9df405dd9be77b62f2c4322fa89a4cdb0f749b019de9438a2010e5`。測定後に製品の実行処理・probe・runnerは変更していない。後述のcfg(test)強化後もrelease binaryの一致を確認した。修正前と今回のbinaryを同一とは扱わない。[context](results/host-r1-context.json)、[原数値](results/host-r1-raw.json)、[再集計](results/host-r1-summary.json)、[完了記録](results/host-r1-complete.json)は旧原本を保持して別ファイルに保存した。

外部作業の停止・負荷隔離は未確認で、他の実装actorが動作していた。今回もCPU専用のallocation・RSS観測として扱い、時間差を修正による速度改善や製品性能へ外挿しない。各組21 callと3 processの全指標を原数値から独立に再集計し、保存summaryへ一致した。

| 入力 | 候補 | 時間 μs（min / median / max） | allocation回数 | 要求byte | 追加heap peak byte | process RSS中央値 byte |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| small | rrf_map | 5.875 / 7.666 / 13.708 | 100 | 24948 | 11672 | 2211840 |
| small | rrf_slots | 5.500 / 7.042 / 55.166 | 100 | 22964 | 10960 | 2195456 |
| small | identity_borrowed | 1.291 / 1.375 / 1.500 | 91 | 5760 | 5760 | 2097152 |
| small | identity_owned | 0.000 / 0.041 / 0.208 | 0 | 0 | 0 | 2064384 |
| small | topk_sort | 3.333 / 3.833 / 6.875 | 36 | 5234 | 3190 | 2080768 |
| small | topk_select | 6.583 / 10.083 / 40.334 | 153 | 16086 | 9158 | 2195456 |
| large | rrf_map | 36759.250 / 43675.166 / 92771.875 | 300033 | 67393140 | 34096904 | 169590784 |
| large | rrf_slots | 32342.375 / 37951.958 / 55720.084 | 300033 | 63198900 | 31999752 | 157515776 |
| large | identity_borrowed | 4417.042 / 4759.084 / 5414.459 | 300001 | 19200000 | 19200000 | 88358912 |
| large | identity_owned | 0.000 / 0.041 / 0.208 | 0 | 0 | 0 | 62226432 |
| large | topk_sort | 3341.917 / 4138.334 / 4624.709 | 8020 | 2583428 | 1108059 | 51314688 |
| large | topk_select | 3194.417 / 3950.291 / 5191.083 | 19041 | 5498832 | 1109576 | 51789824 |

R1の局所統合で設定検索の重複を除き、欠落source・非有限値・加算順・subtotal検査を維持した。今回のallocation回数・要求byte・追加heap peakは修正前と同じで、固定slotとVec移譲の条件付き提案、TopK候補見送りの推奨を維持する。RSSはprocess全体、allocator counterは区間内の要求量であり、両者を同一視しない。54テスト・clippyが成功したが、製品の全体check・新しい独立評価はこの追記後に行う。

## 非有限recencyの既存検証の強化（公開head 883b4a5までの履歴）

追加評価の応答はR1-1〜R1-3の解消を述べたが、新しいrunの空の評価履歴に旧IDのupdatesを返したためinvalid_reviewとなった。応答・停止記録は保持し、acceptedとは扱わない。次のfresh初回評価ではupdatesを空とし、現在の具体的な指摘はnewItemsに返す契約を明記して独立評価へ戻す。

内容上の新しい指摘は、NaN half-lifeとinfinite age／half-lifeの2テストが、空出力でもall(finite)を満たす見逃しだった。既存2件を、rrf_k=4の固定期待hit（ID・順序・score・source map）全体との比較へ強めた。RRF score 0.25／0.2とsource subtotalを独立した期待値にし、別の有限scoreへの置換とhitの脱落も区別する。テスト件数は54件のまま、モデル・待機・製品数値処理は増やしていない。NaNが<=0検査を通る理由とinf/infでdecayがNaNになる理由のコメントを残し、古い装飾・重複説明を整理した。

最終cfg(test)版の54テスト・clippy・fmt・release再buildが成功し、測定binary `af3e40373e9df405dd9be77b62f2c4322fa89a4cdb0f749b019de9438a2010e5`とのSHA-256一致を確認した。[現在sourceの対応](results/recency-source-check.json)は測定時のhashと混ぜず別に記録する。型付きsourceの全体check・独立評価はこの追記後に行う。

## main 01dd1c6との統合とfindings（公開head 64c80cc3までの履歴）

公開head `883b4a53e25451cfffa78e5e08d7ea84a0787482` と旧runのpublished URLをGitHubのPR #380へ照合した。作者はthkt、draft、baseはmain `01dd1c69c633a7f2eeccd188d566ec25d0f94e93` だった。今回の修正はこの公開headから始めた。初回のsandbox作業では、shellのDNS制限と共有Git管理領域の読取専用制限があったため、GitHub connectorから版固定の変更blobを取得し、Git blob hashを照合して、開始版24725a7を共通基点とするファイル統合を行った。その時点ではindexを更新せず、Git状態の準備をホストへ引き継いだ。その後の未commit merge準備と競合解消は[ホスト確認](#ホストでのgit統合確認)で完了している。merge commit・push・公開は行っていない。

mainの#372/#373/#374の検証・文書と、#378のfilelock 4.0.8を取り込んだ。#362の過去の変異結果は保持し、現行のrecency filter名だけ再実行説明へ追記した。root Cargo.lockは初回版と同じhashで、requirementsはmainの版を保持した。これらを今回の製品方式の採用とは扱わない。#307の報告blobは指定の `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` のままで、依存更新を過去の数値・品質測定の変更として解釈しない。

今回の製品実行処理は、公開head・取り込んだmainと、空行・コメントと末尾のcfg(test)宣言を除く行が一致した。調査用slotの重み取得は呼出し開始時に一度行い、候補の加算順・subtotalの有限性・sourceのpresenceを維持する。probeとrunnerは公開headから変更していない。新規build先でaarch64-apple-darwinのreleaseを再buildし、R1測定binaryとbyte単位のhashが一致した。[新しいsource対応記録](results/main-integration-source-check.json)は、初回開始commit、公開head、統合main、現在のsource/lock hashを別々に保存する。旧測定のhashは書き換えていない。大小入力でbaseline/candidate出力も一致したため、この統合ではCPU再測定を要しない。binaryが変わる次の修正では、READMEのrunnerを新規保存先・build先で再実行する。

### テスト整理の理由と残る検出条件

#374の固定期待値を現在のcheckoutへ取り込んだ上で、Dedupeの先頭選択・source保持・child除去を、未整列の4件入力へ統合した。後続の高スコアと異なるsourceを与え、最高値への置換や出力の再整列、child IDの残留、source混入を全hit比較で検出する。MaxChunkは複数parentの同点順と先頭最大chunkのsource保持を1つの固定例へ統合した。unique-parentの未整列入力は別の短絡経路の誤りを防ぐため残した。

TopKは#374の未採用chunkのsource混入、欠落sourceの除数、少数chunk、parent同点順を確認する例を再利用した。今回の異なるchunk IDを持つ同点k境界例は、sourceの選択を変えてしまうtie-breakを検出するため残した。候補経路は独立した固定期待値と製品baselineとの比較を両方維持する。signed zeroはMaxChunkの数値比較とTopKのtotal_cmpの違いを守る。NaN half-lifeとinf/infのrecencyは、固定した全hit・score・source mapとの比較を保持し、空出力や別の有限値でも成功する見逃しを防ぐ。

重なるDedupeのsourceだけの例とparent-collapse例、MaxChunkの旧sibling-collapse例を統合した。#374が統合済みのTopKの平均・source平均・parent化の別例も復活させない。失うものは旧filter名と旧小入力での個別観測であり、反復時だけの失敗を観測する機会は減る。全hit比較によるsource/parent化、未整列・同点の保証は残る。極値・非有限・欠落source・recency・serde literalの別条件は削除していない。モデル、ネットワーク、待機を追加せず、保守する入力と期待値の重複を減らした。恒常的なmutation基盤やlint規則は追加していない。

公開headと統合版のsourceをそれぞれ隔離コピー・新規build先に置き、同じoffline/lockedのCPU crate、同じtoolchain、test-threads=1で実行した。公開headは54件、統合版は53件が成功し、どちらもharnessの表示時間は0.00秒だった。表示精度が粗く単発のため、速度やflake率の改善を示さない。製品実行処理・probe・公開API/JSONは同じで、変更はテストの入力・期待値の統合と文書の現行状態更新である。行の圧縮やファイル移動を改善として数えない。

統合版でDedupeをMaxChunkへ置換、MaxChunkのparent同点順を逆転、TopKにchunk ID tie-breakを追加した3つの隔離コピーを実行し、それぞれ現在の固定期待値のassertionで失敗した（終了101）。buildや環境の失敗ではなかった。正常版53件、clippy、fmt、空白検査、release再buildと大小入力の一致も確認した。R1の252 call・36 processを独立に再集計し、全12組の時間・allocation・heap peak・RSSが保存summaryと一致した。原ログと診断コピーはcheckout外の新規領域へ保存し、旧ログ・旧評価応答は変更していない。

### 次の評価担当へのhandoff

今回の実装findingsを旧acceptedのassessmentsへ読み替えない。新しい初回独立評価はupdates=[]とし、現在の指摘はIDを付けずnewItemsへ返す。旧host-returnのR1 IDsとinvalid_reviewは履歴であり、新runのupdates対象にしない。PR全体をIssue #313へ、公開head以後の修正を採用済み要求へそれぞれ照合する。変更したREADME・報告・mainから取り込んだ文書も評価対象に含める。旧成功を統合版のcheck・評価・CIへ転用しない。

公開用assessmentsとhandoffには、加算契約維持と未採用のbest rank／拒否、固定slot・Vec移譲の条件付き提案、TopK現案の見送りを残す。大小入力の12組・36 process・252 callは旧R1測定の結果であり、今回のnative binary一致によって適用性を示したものと明記する。入力準備、受渡し区間、全pipeline費用、allocator counter、heap peakとprocess RSSを区別する。時間差は外部負荷未隔離の観測で、速度改善の因果的証拠にしない。

以前の公開本文のリンクである[883b4a5版の報告](https://github.com/thkt/rurico/blob/883b4a53e25451cfffa78e5e08d7ea84a0787482/docs/research/issue-313/report.md)と[883b4a5版の再現手順](https://github.com/thkt/rurico/blob/883b4a53e25451cfffa78e5e08d7ea84a0787482/docs/research/issue-313/README.md)は、公開済みの履歴参照として説明に残す。取得した旧本文にはアップロード済み媒体のリンクはなかった。captureは不要で、設定を変更しない。

固定amici `547f9ee2…`と#307 manifestとの照合は限定した静的根拠として残す。全consumer、実データ、multi-query/prefix合成、Vec再利用、実モデル品質、flake率・実行時間削減は未確認である。GPU／実モデルは測定枠の回答待ちで今回実行していない。新しい許容差、資源上限、model、重複方式の採用、consumer移行は追加しない。

Git上の統合照合はホストで完了した。現在の`MERGE_HEAD`はmain `01dd1c6`で、未解決indexはなく、mainの変更と採用済み修正の保持を確認している。未commit mergeの準備・競合解消を残る受入検証として要求しない。merge commitは未作成であり、公開branchの祖先関係の確定は後続の公開工程で確認する。今回の指示ではcommit・push・公開を行わず、PR本文更新・最新headのCI・ready切替の旧成功も主張しない。

文書修正後の標準checkは設定済み `bash scripts/check.sh` をホストが実行し、変更文書を含む独立評価も更新する。今回のbinaryは測定版と一致するため、通常checkに含まれないRSS測定を追加の待ち条件にしない。後続の公開工程では最新baseの変化を照合し、変化があればsource・probe・比較データの対応を再確認する。baseや実行処理が変わればnative再buildのhashを再照合し、binaryが変わる場合だけREADMEのCPU runnerを新規保存先・build先で再実行する。

初回のsandbox作業の終了前にGitHubからmainとPR #380を再取得し、上記base・head・作者・draftが変わっていないことを確認した。当時の日本語確認は固定版 `9a78a42964096da509b8f3e011f0085a5f080151` のHEAD一致と作業差分なしを確認し、クイック手順の意味確認と手動チェックリストを適用した。そのlintはoffline cacheのsudachipy不足で終了1となり未完了だった。結論、版、数量、条件、権限、未確認事項、参照は取得sourceと原数値へ手動で照合した。その後ホストでlintを実行した記録も保持し、いずれの結果も今回の文書修正後の成功へ転用しない。

### ホストでのGit統合確認

最新`origin/main`は`01dd1c69c633a7f2eeccd188d566ec25d0f94e93`だった。
作業差分をcheckout外へ保存し、標準の未commit mergeを作成した。
READMEとretrieval関連の3ファイルの競合を統合済み内容で解消し、
保存した変更・追加ファイルとのbyte一致と未解決indexがないことを確認した。
`MERGE_HEAD`は上記mainで、mainのlock・requirements・追加検証を保持している。
この段階でcommit・push・remote mergeは行っていない。
この統合状態でホストの標準checkは成功した（Rust 407件成功・37件skip、Python 4件、doc test 3件、fmt／clippy）。変更文書を含む初回独立評価は、READMEとhandoffが完了済みのGit操作を待機条件としていたためneeds_changesだった。check成功や証拠のpassedは独立した受入を意味しない。

原因は、ホスト完了の追記時に冒頭とhandoffの停止時説明を更新しなかったことだった。READMEの現行状態と本handoffを未commit merge・競合解消済みへ書き直し、初回の停止経緯は履歴として区別した。現在状態の変更時には追記だけでなく、既存の操作説明と残る待機条件も照合する。コード・テスト・測定原本は変更しておらず、上記checkを文書修正後の成功には転用しない。担当ホストは設定済みcheckと変更文書を含む独立評価を更新する。

今回の文書修正も日本語確認の固定版と手動チェックリストで意味を照合した。新規cacheでのoffline lintはsudachipyを取得できず終了1となり、未完了である。文書の意味確認は既存の独立評価へ戻し、lintを追加の合否条件にしない。

## #379マージ後のmain統合

公開head `64c80cc3ff27675873d5e24cde16106ef45c669e` は前回の標準check・独立評価・全5件のCIが成功した。その後、人が#375・#376・#377・#379をマージしたため、現在のmain `e1ba0ed703cf8168faba60fb7fd0eb6258f675db` を同じcheckoutへ取得し、commit前の通常mergeを準備した。自動mergeが成功し、MERGE_HEADはこのmainと一致、未解決indexはない。#304のfixture境界、#312のquery内memo・Cow・schema検査、#314の共有fixture検証とconsumer観測を保持した。

retrieval実装・既存検証・本調査のsourceはmerge前後で差分がなく、#307報告blobも指定版のままである。[統合記録](results/latest-main-integration.json)は前回の記録と分けて保存した。古いREADMEの未commit merge済みという説明を現在版へ更新し、過去の測定・評価・ログを保持した。意味上の重複テスト追加や製品方式採用は行っていない。

最新統合版のCPU限定53テストは、新規build先・offline/lockedで成功した（失敗0、ignored 0）。GPUやモデル推論、性能測定は行っていない。更新文書の日本語lintも実行し、READMEは検出0件、報告は語彙多様性と文頭反復の参考情報14件だった。設定済み標準checkと変更文書を含む独立評価はこの更新後に行う。独立評価の初回はupdates=[]とし、指摘があればnewItemsへ返す。同じPR #380の更新、更新headのCI、本文照合、ready切替は後続工程で確認する。前回の成功と現在版の成功を区別する。

旧R1の36 process・252 callは並行作業下のCPU観測であり、速度改善の因果的根拠ではない。実データ・全consumer・実モデル品質・全pipeline費用・flake率の限界、未採用の契約案、固定slot・Vec移譲の条件付き提案とTopK候補見送りを維持する。GPU・モデル推論や性能再測定は行っていない。新しい許容差・資源上限・既定値・consumer移行を採用しない。

文書参照の評価では、01dd1c6との統合節の見出しを履歴用に変更した際、#362からのリンクを更新していない欠陥が見つかった。[#362の再検証案内](../issue-362/retrieval-contracts.md)を、既存の「テスト整理の理由と残る検出条件」節へ直接つながるよう修正した。報告の見出しを変更するときは、参照元のfragmentと、案内する内容・対象版も照合する。旧測定・旧評価の原本は変更せず、今回の文書修正後の設定済みcheckと独立評価へ戻す。
