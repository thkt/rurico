# Issue #309: 長さ別rerankバッチの比較

長短混在の用途では、既存bucketのまま長さ別に分ける方式を次の製品実装の候補として推奨する。
固定50ペアでwarm中央値は6.418秒から2.273秒へ約65%短縮し、Metal peak中央値は約28%減った。
3種類の境界比較でも時間の利益を観測した。同長と上限の時間差はばらつきの範囲内だった。
logit最大絶対差2.98e-6、score最大絶対差6.56e-7で、両方の順位と同点順も全条件で一致した。

公開APIへの採用は今回見送る。試作はtest + smoke内だけで選択でき、製品APIは従来方式のままである。
今回の完了対象は比較と推奨案であり、一般の入力、全モデル、consumerの検索品質や5000ペアの保証ではない。
9件を混ぜた境界比較はpressure停止したため、各境界3件へ分けて測定した。
この未完了履歴を分割版の成功で置き換えず、対象用途の入力分布・許容費用を確認してから採用を判断する。

## 合意・開始版・適用する根拠

要求の正本は[Issue #309](https://github.com/thkt/rurico/issues/309)、updatedAt `2026-10-07T17:05:12Z`。
2026-10-12、ユーザーは測定前に、logitとscoreの最大絶対差を各2.5e-5以下、
順位は現行版と一致、完全同点は元index順とする比較条件を承認した。
製品の許容差、bucket表、cache policy、default modelやconsumer移行の変更は合意していない。

開始commitは`5288f4fb38b436084d63ec9cd56754f6e118459b`で、ホストからGitHubのmainとの一致を確認した。
Cargo.lock SHA-256は`d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7`。
#318のマージ後のsourceであり、旧根拠版`32513da6`や見直し版`c8f250d6`の数値を比較値へ流用していない。
rerankerは全pairをencodeし、全件最大長のbucketと既存budgetで分割する方式を維持している。
型付きエラー、重み・有限性検査とcleanupの既存修正も再実装しない。

| 出典（開始commitのblob） | 適用する内容 |
| --- | --- |
| [#307](../issue-307/report.md) `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 固定モデルと数値・近接順位の診断を再利用。今回の2.5e-5と完全順位一致で判定し、旧診断値を製品基準にしない |
| [#318](../issue-318/report.md) `dc9acecdcde4a324e9494ac5e5cc0d0be828a279` | 全入力token保持と1 forwardの規模を分ける。小さい窓・sink・hard limitの案は今回採用しない |
| [ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md) `e96818842c029af9466cb4b0891d10d20d68f489` | thktによるaccepted。128/512/2048/8192を維持。4 bucketだけで全shapeのcacheやprocess peakが4種類に収まるとは扱わない |
| [ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md) `5c2840daf74b8364a88b37f864ed5717d6bbf403` | thktによるaccepted。既存MLX専用構成を維持し、CPU/candle backendを追加しない |

4文書のblobは開始版と一致した。sandboxで親とremoteの取得が失敗した履歴は初回runに残す。
ホストでは親[#296](https://github.com/thkt/rurico/issues/296)と[#306](https://github.com/thkt/rurico/issues/306)を取得した。
親のupdatedAt `2026-10-07T17:08:32Z`と本文SHA-256
`5918980b98047e009322bf25711d01b7aab98802c826719ef3cc0bcab7d00ee6`は前回と一致した。
setup/check/captureは設定どおりで、UI媒体は不要。

## 同じencoding・readback・cleanupで比較する

[候補dispatch](../../../src/reranker/bucket_spike.rs)は元indexを固定bucket別に保持し、
既存`compute_sub_batch_size(bucket, None)`で分割して、forward結果を元indexへ書き戻す。
返却過不足・forward失敗では途中成功を返さず後続を止める。
同じdispatchをモデル不要検証と[実モデルprobe](../../../src/reranker/mlx.rs)が呼ぶ。

現行と候補はpair encoder、EOS付きtruncate、mask/padding、sub-forwardと
[logit検査](../../../src/reranker/processing.rs)を共有する。
一時Array解放後の既存`run_inference` cleanupとエラー伝播も維持する。
候補のgather copy、全tokenと復元slotの保持はwall・RSSに含め、streamingやhard limitとはしない。

一度のforwardのCPU readbackからlogitとsigmoid scoreを作る。
測定のための再推論をcallに混ぜない。probe終了前の製品経路との照合は、時間記録の外で別に行う。
その照合とmodel loadはprocess RSSの範囲には含まれる。
wallはpair encoding、分配/gather、全forwardとscore作成を含むhostの包含区間である。
`host_forward_ns`はpadding・forward/eval・readback・cleanupを含み、wallへ足し合わせない。
入力文字列の生成とmodel loadはwallの外。mutex待ちとGPU kernel単独時間は未測定である。

## 7条件の結果

Apple M3（Mac15,3、24 GiB）、macOS 27.0.1、Rust/Cargo 1.99.0、Xcode 27.0、Metal 32023.921で実行した。
固定`cl-nagoya/ruri-v3-reranker-310m`、revision `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`、F32。
モデル・config・tokenizerの内容hashは#307保存manifestに一致し、前後hashも一致した。
release、locked、`test-mlx,smoke`、ビルド並列数2で作成した同じbinaryを両方式で使った。
追加flagsはcommandに指定していないが、継承した設定すべてを独立に監査したとは扱わない。

各条件を独立processの3組で交互順に比較し、各processの初回をcold、続く3 callをwarmとした。
coldはmodel load後の初回であり、OSのfile cacheを消したcold diskではない。
以下は現行→候補。warmは各processの3 callの中央値を、さらに3 processで中央値にした値。
時間はms、Metal peakはMiBで、3 processの中央値である。

| 入力 | cold | warm | Metal peak | padded tokens | forward数 | 時間の判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 同長50ペア | 1479.9 → 1531.5 | 1519.7 → 1520.3 | 2899.4 → 2899.4 | 6400 → 6400 | 1 → 1 | ばらつき内 |
| 長短混在50ペア | 6356.7 → 2383.1 | 6418.4 → 2273.5 | 3953.5 → 2831.5 | 25600 → 9472 | 1 → 2 | 利益を観測 |
| 127/128/129 | 403.5 → 222.1 | 405.7 → 198.9 | 2389.9 → 2082.0 | 1536 → 768 | 1 → 2 | 利益を観測 |
| 511/512/513 | 1878.4 → 1006.0 | 1970.2 → 892.0 | 2718.4 → 2487.4 | 6144 → 3072 | 1 → 2 | 利益を観測 |
| 2047/2048/2049 | 14198.8 → 6058.0 | 14188.2 → 6191.0 | 4866.5 → 3306.4 | 24576 → 12288 | 1 → 2 | 利益を観測 |
| 8191/8192/8193（切り詰め前） | 14111.8 → 14131.7 | 14167.4 → 14095.3 | 4866.5 → 4866.5 | 24576 → 24576 | 1 → 1 | ばらつき内 |
| 空入力 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0 → 0 | 0 → 0 | 判定しない |

[小さい生記録と集計](results/split/summary.json)に各3組の最小・中央値・最大を保存した。
観測範囲が重ならず、中央値差が両方式の観測範囲幅の大きい方を超えた場合だけ、今回の利益・退行signalとする。
小標本の記述量であり、有意差や一般の速度上限の証明ではない。
混在入力のwarm範囲は現行6355.3–6421.4 ms、候補2271.2–2361.9 msだった。
同長・上限はばらつき内、空入力は速度を判定しない。観測した条件では退行signalはなかった。

混在50ペアではpadded tokenが25600→9472（63%減）、forwardが1→2となった。
追加forwardとgather費用を含めても、今回の分布では時間とMetal peakが減った。
同長と上限はbucket・forward数が同じで、peakも同じだった。
全長を一つへ揃える分布ではpadding削減の利益がないため、方式採用を入力分布から切り離さない。

7条件・42 process・168 callのlogit/scoreと順位は合意値を満たした。
完全同点は元index順。mixedは図書館と猫の2 queryを含むため、全体のscore orderは機械的な数値診断である。
異なるqueryを横断するscoreを検索上の関連度順位と解釈せず、一般の検索品質改善は主張しない。
空入力はmodelロード後のinner経路の観測であり、公開APIの空入力コストを測った値ではない。

約20.9分、8959標本でpressureは常に1、swap増分0、process RSS標本最大は約1.98 GiBだった。
RSSはmodel load・入力準備・非計測照合を含むprocess範囲の標本最大で、Metal peakと範囲が違う。
短い瞬間最大、CPU/Metal共有全量、他appのGPU利用や熱の全期間は保証しない。
process名とCPU使用率の時系列を私的な生ログに保存し、測定とこちらの別build/GPU作業は並行させていない。
42 processの正常終了とgroup不在、source/model/binaryの前後一致を確認した。

## 停止した二つの測定は未完了のまま残す

[初回](results/attempt-1.json)は最初の同長referenceでpressure 1→2となり、約1.09秒で停止した。
RSS標本最大約1266.8 MiB、swap増分0、完了callは0件だった。
その後ユーザーがメモリを確保し、再測定を承認した。

[再測定](results/attempt-2.json)は同長・混在の12 processを完了したが、境界9件のreferenceで停止した。
旧版の混在warm 6.436→2.360秒などは同版の部分結果として保存し、現在版の値へ流用していない。
予定shapeは8192×9。原因は特定していないが、小規模比較を継続するため、各境界3件へ分けた。
9件混在の完了を分割版で代用せず、guard・製品budget・数値条件は緩めていない。
両停止で自身のgroup不在を確認し、他のアプリやチャットは終了していない。
OOM・リーク・candidateの退行を、pressure停止だけから断定しない。

[#364](../../benchmarks/issue-364/README.md)の5000ペア未完了も維持する。
短文だけなら既存budgetは2000/2000/1000行のforwardを許し、長さ別分配でも変わらない。
全入力token保持と1 forward規模、単一巨大入力、全CPU/GPU上限、OOM/リーク不在は別の課題である。

## 検証と再現

[README](README.md)にビルド、guard付き実測、単独の既存実モデル検証の手順を記す。
公開記録は個人pathをmodel内容hashへ置き換えたもので、モデル・binary・他process一覧は公開していない。
source・binary・lock・入力・modelの識別は[environment.json](results/split/environment.json)を参照する。
公開記録からの再集計は次のコマンドでできる。新しい推論やbinaryの再検証ではない。

```sh
python3 docs/research/issue-309/summarize.py docs/research/issue-309/results/split
```

モデル不要の3試験は、候補dispatchへ異なる識別値を通して、2033行の複数sub-batch・端数・同点を確認する。
同長500+1、全境界、空入力、過不足と失敗後の停止も同じ候補経路で確認した。
製品sortの同点順を逆にする変異と、scatterを一つのslotへ壊す変異を対応する試験が検出した。
sandboxの一時wrapperは製品定数とsortを機械的に抽出してMLXをリンクしない。
それを正式なcrate全体の成功と混同せず、ホストの標準checkでも試験を実行した。

Pythonの最初の追加2件は欠損記録・別model/input・logit差の取り落としを防ぐ。
既存の数値比較器、budget、truncate、非有限値、cleanupの異なる失敗条件を維持し、同等のGPU総当たりを増やさない。
初回checkはclippyのimport違反3件で失敗したが、修正後はRust 431件、Python 16件、
doctest 3件、clippy、formatが成功した。ignoredの29件は成功件数へ含めない。
その後の境界入力分割と文書更新は、復帰runの標準checkと独立評価で改めて確認する。

同じrelease binaryで既存の`real_model_cleanup_memory`と
`cached_model_scores_singleton_batch_and_reranks_in_input_order`を、同じguardで一件ずつ実行して成功した。
cleanupは固定embed/reranker 310mの通常・故障注入の生標本を保持する。全入力のリーク不在へ一般化しない。
対象名・実行1件・終了0・group不在を確認し、空振り成功とは扱っていない。

手を入れたファイルでは、自明な処理コメントとartifact aliasの移動履歴を削除した。
score_batchを「単一forward」とする誤った説明も修正し、入力順・数値・budget・cleanupの契約説明は残した。
日本語は固定版natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151`のクイック手順で推敲する。
sandboxのlint依存取得失敗は初回runへ残し、ホストで実行した検査と区別する。
変更後の文書を含む独立評価と、公開した同じheadのCIは復帰runで確認する。

独立評価後、公開再集計にも件数・有限性・順位・mode/state・bucket/budgetの共通検査を適用した。
重なる時間分布は利益・退行の判定へ使わず、保存済み7条件の結論は変わらない。
初回停止のRSS要約も原標本へ訂正した。referenceの最後の計測結果を製品経路との照合へ再利用し、
同じ観測の余分な5回目を削除した。4回の計測と製品呼出し、candidateの別shape照合は維持する。
削除した1回だけで起こる一過性失敗を観測する機会は減る。保存済み実測はこの局所修正前の版であり、
測定中の推論とcleanupは変更していないが、process RSSは旧版の余分な照合を含む。

修正後の通し検証はreferenceのモデル起動中にpressure 1→2で停止した。
[停止記録](results/correction-attempt.json)は完了call 0件、RSS標本最大約1477.5 MiB、swap増分0、
所有group不在を示す。candidateは開始していない。旧7条件の比較完了と区別し、
修正版の製品経路との通し照合が成功したとは扱わない。guardと数値条件は維持する。

修正後の標準checkは成功し、公開入口の不正記録と重なる時間分布を守る追加2件も成功した。
メモリ確保後、修正版のreference・candidate各1件の通し検証に成功した。
[修正版の記録](results/correction-parity.json)は同長50ペア、各4回の記録と製品経路との照合、
合意値・順位一致、pressure 1・swap増分0・終了0・group不在を示す。
性能比較の全7条件を修正版で再実行したものではない。独立評価の再判定は復帰runで確認する。
今回の修正依頼ではcommit・push・公開を行わない。

## 2026-10-12: call件数検査の修正と再評価

### assessments

評価対象`1dbafd28b2b20344c43f5c637fef7243dc2373974456ec09769b07af370b899a`の
必須指摘R1-1は、raw入口と公開再集計入口が、共通検査へ渡す同じcall列の件数を再検査していたことである。
両入口はheaderが1件であることを確認し、4 callの契約を`validate_calls`へ集約した。
model/path、出力件数・有限性・順位、mode/state、bucket/budget、実測前後hashの照合は維持する。
開始commit・Cargo.lock・Rust実装・既存テスト・check/capture契約はこの修正で変更していない。

関連Python検証4件が成功した（今回の実行は0.202秒）。保存済み42原観測をraw入口で検査し、
pathを内容hashへ置き換えた公開記録との一致と、7条件の再集計値の完全一致を確認した。
両入口へ0・3・5 callを与えると共通検査が拒否した。正常4 callは受理する。
欠損記録検証と保存値の再集計で今回の保証を確認できるため、新しい恒久テストは追加しない。
既存テストは欠損出力の誤受入、別入力比較、数値差の取り落とし、重なる時間分布の誤判定という
異なる不具合を防ぐため維持する。検出条件の喪失はない。GPU試験は追加していない。
変更前後の同条件での実行時間差・不安定さ・全体費用は未測定で、速度改善を主張しない。

別runの旧R1-1〜R1-4も現在の内容へ照合した。欠損出力は公開入口の既存否定例で拒否され、
重なるwarm分布はunresolvedとなる。最後のreference結果の再利用は現在のRust実装と
測定後差分で確認し、candidate側の別shape照合・製品呼出しは残る。
削除した余分な推論だけで起こる一過性失敗の観測機会は減るが、4 callと製品照合は維持する。
初回停止は原8標本の最大1328316416 bytes（1266.8 MiB）、修正版の停止は原9標本の最大
1549221888 bytes（1477.5 MiB）で、[初回](results/attempt-1.json)と
[修正版停止](results/correction-attempt.json)および本文と一致した。原記録は変更していない。
capacityエラーで未完了だった旧repairや形式不正の評価応答を、修正成功・受入の根拠には使わない。

指定4文書は上記の開始版blobと一致した。追加証拠46件のhash、測定条件と原観測を照合し、
証拠のpassedを独立した受入とは扱わない。旧7条件の測定はその測定版の観測として利用できる。
計測後のRust変更はreference照合の再利用に限られ、計測区間は不変だが旧RSSは余分な照合を含む。
[修正版の通し照合](results/correction-parity.json)は同長50ペアの両モードだけに適用する。
今回の修正は記録の件数検査のみで、新しい推論や全7条件の再実測ではない。

### handoff

変更後の全体checkは設定済み`bash scripts/check.sh`をホストで実行し、変更文書と本再評価を
独立評価へ渡す。captureはnull、UI媒体は不要で、追加のGPU検証を今回の局所修正の条件にはしない。
旧7条件と修正版同長50ペアの適用範囲を区別し、標本間の瞬間最大・継承build設定全体・
他アプリのGPU利用や熱状態の全期間、全モデル・一般検索品質は未確認として引き継ぐ。
9件混在と5000ペアの未完了、全入力保持量と1 forward規模は上記の#318/#364側に残る。
製品採用時は対象用途の入力分布と許容費用を調査し、API採用・資源条件をユーザーの判断へ戻す。
