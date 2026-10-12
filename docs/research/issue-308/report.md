# Issue #308: 固定bucket表3構成の実測比較

現時点では製品への区分追加を見送る。固定310m/F32の比較では、中間の長さを新しい区分へ収める
境界入力で速度が改善した。一方、全区分を使うembeddingのMetal peakは64/68 MiB増え、
1024境界では1024追加構成のpeakが現行より140 MiB高かった。
実入力を使ったrerankerの混在50件は3構成とも512区分で、追加区分の利益を確認できなかった。
代表的な実入力、主対象options、製品の許容メモリとconsumer移行が決まるまで、4区分を維持する。

3構成の比較入口は[README](README.md)、全ケースのcold/warmの範囲・peak・入力accountingと
継続系列は[実測集計](results/observed.json)にある。embeddingの合意済み基準を全件で満たし、
rerankerは全比較でlogit/score差0・順位一致を観測した。
区分採用・検索品質・別モデルでの効果まで確認した結果ではない。

rerankerで照合したのは製品の内部score_batch経路で、公開wrapperは通していない。
実測集計の`public API parity`という注記は当時の誤記として保持し、この報告で範囲を訂正する。
比較条件の承認原文は[合意記録](comparison-agreement.md)に示した。
独立評価時の参照記録には原文がなかったため、担当AIがチャットの人の回答を照合して補足した。

## 出典と適用性

要求・合意の正本は[Issue #308](https://github.com/thkt/rurico/issues/308)、
引き継いだ更新時刻は`2026-10-07T17:05:09Z`。本作業の確認日は2026-10-12。
開始HEADは`f561ab8aed009f4a0a38470f94a8802a1d0ef897`で、指定のstartCommitに一致した。
ホストでorigin/mainをfetchし、開始時点の最新mainとの一致を確認した。
Cargo.lockは変更せず、測定時は同じ未commit差分を含むsource hashも固定する。

以下の現行報告は開始版の指定blobとすべて一致し、今回改訂していない。
報告の観測・提案を新しい許可として扱わない。

| 出典 | 開始版のblob | 今回の適用条件・合意状態 |
| --- | --- | --- |
| [#307報告](../issue-307/report.md) | `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 固定310m/F32の数値比較の出発点。新しい品質許容差・区分採用の合意ではない |
| [#309報告](../issue-309/report.md) | `594c3d03944ba605c4505922d572556b6f9fc109` | 長さ別分配は調査候補。#308への適用は[人の回答原文](comparison-agreement.md)で確認した。各2.5e-5・順位一致・完全同点の元index順は比較条件で、製品採用の合意ではない |
| [ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md) | `e96818842c029af9466cb4b0891d10d20d68f489` | acceptedの4区分invariant。製品採用時は後続ADRによるsupersedeが必要 |
| [ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md) | `5c2840daf74b8364a88b37f864ed5717d6bbf403` | acceptedのMLX-only条件。別backendによる代替実測はしない |

Issueが参照する`c8f250d…`から開始HEADまでに、planner実行境界・readback計測・smokeの版間比較と
rerankerの製品関数共用が更新されている。現行の
[文書処理](../../../src/embed/processing.rs)、[metrics](../../../src/embed/metrics.rs)、
[smoke記録](../../../src/bin/mlx_smoke/records.rs)、[reranker処理](../../../src/reranker/processing.rs)へ照合した。
旧計測APIを複製せず、製品APIと現行metrics、#309のreference経路を使う。
4区分の値自体は同じである。

Issue本文の保存済み256件のCPU計算は過去の観測として維持する。
私的本文・個別token列を再取得せず、今回の公開合成入力で独立して3構成を測定する。
605,696/478,720/441,856位置、標準budgetの6/6/7 forwardという過去の値を、
今回の入力やGPU実測の値として再利用しない。mask容量の273.1/337.1/341.1 MiBも計算値である。

## Consumer影響と後続判断

現行の`InferenceMetrics`と`BatchMetrics`は公開`[usize; 4]`のhistogramを持つ。
製品で5/6要素へ変更すれば、arrayを受けるsource、pattern、serializeされた配列長のconsumerへ影響する。
内部の文書分配・`PhaseMetrics`、#309試作の分配も固定4要素である。
比較コピーでは必要な配列だけを追従させ、元checkoutのsource互換性は維持した。

[smoke](../../../src/bin/mlx_smoke.rs)には`[usize; 4]`の引数・fixture、
histogramのindex 2以降を長文とするSLA分類がある。
1024をindex 2へ追加した場合、そのままでは分類条件も変わる。
phaseログは`bucket_hist_0`〜`bucket_hist_3`だけを出すため、製品採用時には追加区分の表現も決める必要がある。
今回の比較はこれらのsmoke baseline判定を使わず、実際のsequence lengthと全要素を保存する。
標準smoke binaryが候補配列でビルドできることは未検証であり、比較コピーのlibtest成功から推定しない。

`ModernBert`のlocal maskはsequence lengthをkeyにしたHashMapで、cache自体が4件で拒否する実装ではない。
callerが固定表を使うことでshape集合を制限するというADR-0009の条件を適用する。
kernel数はbatch shape等にも依存し得るため、ADRの「4 kernel」という記述を実測peakの上限保証にしない。
候補ではlocal maskのkey集合が5/6値になる可能性があり、全区分を使う計算上の容量と実観測を比較する。

rerankerの各2.5e-5・順位一致・完全同点の元index順は#308の合意済み比較条件で、製品許容差にはしない。
差0と順位・同点順一致は保存済み出力の観測事実で、許容差の承認の証明ではない。
今回は製品採用を見送るため、新しい許容差の採用判断は行わない。
主対象options、機種の製品許容メモリとconsumer移行も未合意である。
比較の出力を理由に製品hard limitや新baselineを導入しない。
採用を推奨する場合は、実測結果を踏まえてこれらの判断とADR-0009のsupersede案へ戻す。
#309の長さ別分配、#310のcache、#321のprecisionは同じ比較へ混ぜない。

## 測定条件と完全性

2026-10-12、Mac15,3（M3、24 GiB）、macOS 27.0.1、Xcode 27.0、Metal 32023.921、
Rust/Cargo 1.99.0で測定した。release、`--locked --features test-mlx,smoke`、ビルド並列2、
追加RUSTFLAGSなしを3構成で固定した。embedding/reranker310mのrevisionと3ファイルのSHA-256は
#307の固定manifestと一致し、測定後も変化していない。

開始sourceのSHA-256は`2e685e6b592e03911dba34d36591553d8ff1332351f4708bb1fbdf9eaef1f848`、
planは`50f60dc8a36c5592a34e9e6334f1fde24cd4aa193a182beba6d0275016e928c0`。
各候補のsourceとbinaryを固定し、各測定区間の前後に照合した。
製品のbucket表は変えず、隔離コピーだけを4/5/6区分にした。
候補コピーの既存テストfixtureはlibtestをコンパイルするために配列へ追従させた。
候補版の固定4区分テストやsmoke binaryが成功したという証拠にはしない。

embeddingは文書budget 8192、forward後pause 1 ms。rerankerはbudget 256000、
現行の全pair最大長方式を固定した。モデルloadと入力準備はcallの時間から除外した。
coldはload後の初回で、OS file cacheを消した起動ではない。queryは文書計測後に実行した。
3つの独立process組で実行順を循環し、各processの初回と残り3回のwarm中央値を分けた。
以下の表はwarm中央値3組の中央値（最小〜最大）、単位は秒である。
全cold値とMetal peak/RSSの範囲は実測集計に保存した。

### Embeddingの主要比較

| 入力 | 現行 | +4096 | +1024/+4096 |
| --- | ---: | ---: | ---: |
| boundary128 | 0.201 (0.199〜0.214) | 0.201 (0.201〜0.202) | 0.202 (0.198〜0.202) |
| boundary512 | 0.901 (0.895〜0.995) | 0.896 (0.895〜0.918) | 0.543 (0.538〜0.549) |
| boundary1024 | 1.875 (1.871〜1.908) | 1.994 (1.989〜2.137) | 1.199 (1.189〜1.200) |
| boundary2048 | 6.241 (6.226〜6.250) | 3.038 (3.035〜3.041) | 3.039 (3.037〜3.043) |
| boundary4096 | 14.661 (14.657〜14.676) | 8.274 (8.247〜8.280) | 8.268 (8.235〜8.273) |
| limit | 20.156 (19.560〜20.472) | 16.712 (16.384〜17.647) | 16.862 (16.387〜17.116) |
| mixed | 11.677 (11.642〜11.714) | 8.378 (8.366〜8.440) | 7.983 (7.980〜8.013) |
| multichunk | 15.914 (15.860〜15.916) | 15.856 (15.852〜15.896) | 15.500 (15.463〜15.512) |

### Rerankerの主要比較

| 入力 | 現行 | +4096 | +1024/+4096 |
| --- | ---: | ---: | ---: |
| boundary128 | 0.425 (0.421〜0.427) | 0.425 (0.425〜0.436) | 0.421 (0.418〜0.425) |
| boundary512 | 2.042 (2.039〜2.046) | 2.039 (2.036〜2.041) | 0.893 (0.889〜0.897) |
| boundary1024 | 2.040 (2.038〜2.047) | 2.058 (2.049〜2.060) | 2.049 (2.038〜2.067) |
| boundary2048 | 15.209 (15.193〜15.223) | 5.162 (5.149〜5.218) | 5.148 (5.135〜5.149) |
| boundary4096 | 15.231 (15.197〜15.251) | 15.173 (15.167〜15.230) | 15.198 (15.195〜15.275) |
| limit | 15.203 (15.161〜15.289) | 15.227 (15.218〜15.229) | 15.237 (15.230〜15.272) |
| mixed | 6.798 (6.793〜6.817) | 6.813 (6.798〜6.818) | 6.821 (6.819〜6.829) |
| same | 1.613 (1.610〜1.628) | 1.617 (1.608〜1.619) | 1.610 (1.606〜1.614) |
| empty | 0.000 (0.000〜0.000) | 0.000 (0.000〜0.000) | 0.000 (0.000〜0.000) |

embeddingのmixedは実token長13/129/513/1025/2049/4097/127で、
各構成の全区分を同じprocessで使った。warm中央値は現行11.677秒から8.378/7.983秒へ、
約28.3%/31.6%短縮した。一方Metal peakは3305.3→3369.3/3373.3 MiB、
RSSの標本最大は2026.6→2154.6/2162.6 MiBだった。
全表のmask容量推定273.0625/337.0625/341.0625 MiBと、process/callの実peakは別の値である。

2048境界では8192へpaddingされていた入力が4096へ移り、embeddingのwarm中央値は約51.3%、
rerankerは約66.1%短縮した。rerankerのMetal peakは4866.5→3018.5 MiBだった。
1024境界のembeddingでは新構成のforwardが分かれ、時間は短縮してもMetal peakが
2716.2→2856.2 MiBへ増えた。paddingの削減だけからpeak低下を推定しない。

embeddingのlimitは8191/8192/8193tokenの3文書で、plannerは4chunk
（8191/8192/8192/2055）へ分けた。末尾2055tokenが8192から4096区分へ移るため、
全構成が8192だけを使った比較ではない。multichunkは13/16400/513tokenの3文書で、
chunk長13/8192/8192/4124/513、文書ごとのchunk数1/3/1を保った。
4124tokenは候補でも8192区分なので、4096追加の利益は限定的だった。

rerankerのmixedは#307の固定入力を50pairへ展開したもので、実長13〜136、全構成512区分だった。
sameは同じ短いpair50件、emptyは0件でforwardとtoken数0を確認した。
新旧の4096境界とlimitでは全構成が8192を使い、warmの差は小さい。
これらを安定した利益として扱わない。同じshapeでもcoldのMetal peakには観測幅があり、
4096境界とlimitの現行最大4938.5 MiB対候補4866.5 MiBを区分追加の効果とは断定しない。
rerankerはcaseごとに独立processを使い、全mask keyの同時保持は観測していない。

## 数値・継続観測・停止記録

主要比較はembedding8ケース×3組×3構成、reranker9ケース×3組×3構成、
各4callで収集した。文書・queryの912件のvector比較はcosine ≥ 0.99999かつ最大絶対差 ≤ 1e-5を通過し、
最大絶対差は0。rerankerの2832件のpair比較はlogit/score各2.5e-5の診断値以下で、
両方の最大絶対差は0、score/logitの順位と元indexによる完全同点順も一致した。
文書ID・chunk順・次元768・有限性・prefix/overlapを含むtoken列、histogram・forward・
padding・pause accountingも比較した。rerankerは計測後に製品の内部score_batch経路とのparityを確認した。
呼出先はRerankerInner::score_batchで、公開Reranker wrapperの空入力処理・mutex取得は通していない。
既存のwrapper検証と今回の実モデル観測は範囲が異なる。

mixedの継続観測は各構成を別processで31callずつ実行した。
embeddingのprocess観測時間は375.8/270.1/258.1秒、rerankerは217.0/216.5/217.0秒だった。
追加のvector448件とpair3100件もembedding基準・reranker診断値以内で、順位も一致し、最大絶対差はすべて0。
embeddingのMetal系列は3305.3/3369.3/3373.3 MiB、rerankerは全構成3953.5 MiBで各31callとも一定だった。
process RSSの標本最大はembedding2026.5/2154.5/2161.5 MiB、
reranker1479.3/1480.5/1480.5 MiBだった。数分の固定入力の観測であり、長期のleakやOOM不存在を証明しない。

初回の測定は128/512/1024境界の27processが成功した後、2048境界の現行初回で
memory pressure 1→2となり停止した。このprocessでは文書callの完了記録がなかった。
モデルload中に止まったとは断定しない。28の所有groupはすべて回収した。
ユーザーのメモリ確保後、未完了の2048/4096/limitから再測定し、成功済みの3境界は繰り返さなかった。
初回の停止記録と生ログを保持し、caseごとに9processの成功・回収と対象版を照合して集計した。
停止run全体を成功へ書き換えていない。

再測定はRSS 8 GiB超、pressure 1以外、swap増分256 MiB超、30分の期限という
同じガードを維持し、全区間を完了した。電源はAC、開始時に熱・性能警告の記録なし。
担当AIの別ビルド・GPU作業は測定中停止した。他の所有者のprocessは終了していない。
資源は約0.1秒、process名/CPUは約1秒で観測した。温度の連続観測や他アプリのGPU無負荷は保証しない。
RSSはload・入力準備・query・内部score_batch照合を含むprocessの標本で、瞬間peakとは区別する。
callのMetal peakとwallも、GPU kernel時間やGPUのみのメモリとは読み替えない。

## 検証の保守費用と残る範囲

追加したモデル不要の検証は、旧shapeの誤記録、文書IDの欠落・重複、prefix token差、
不完全な反復、数値不一致の消失、coldをwarmへ混ぜる集計と、対象0件の測定成功を検出する。
0件の組合せと重複選択を拒否する入口修正は測定後に追加した。
今回の有効な選択は修正前後で同じcase順になり、Rust/source・binary・演算条件は変更していない。
測定後のその他の変更は結果JSON・報告の更新である。再ビルドによる同じ演算の再測定は重ねなかった。

モデル不要のPython22件は5.697秒で成功し、format検査も成功した。
日本語はnatural-japaneseの固定commit `9a78a42964096da509b8f3e011f0085a5f080151`で確認した。
機械lintの語彙反復の情報と環境・維持対象の列挙2件は、技術用語と条件を保つため変更しなかった。
原資料との意味と数値を照合した。

既存テストの削除・統合はなく、既存のplanner・順序・prefix/overlap・有限性の検出条件は維持した。
比較用のモデル検証はignoredで、通常checkのPython discoveryには軽量な記録検証だけを追加した。
手を入れた既存コードのコメントを通読し、readback・cleanup・時間範囲・内部経路と公開wrapperの区別を説明するものは残した。
説明のためだけの同等な演算テストや、設定・coverage閾値の緩和は追加していない。
標準check・独立評価・同じheadのCI結果は、実装CLIの記録と公開PRで別途確認する。
CI成功を実モデル測定の代替にはしない。

embeddingの入力は合成文字列で、#307のコード・英語・topic prefix全体、amiciの60文書/168クエリ、
私的256件、30m/70m/130mと一般consumerの検索品質は測定していない。
rerankerのmixed/sameは#307の固定fixtureを使ったが、新区分を使う代表的な実入力の50件比較ではない。
queryは数値比較だけで、単独のcold/warm速度・peak比較ではない。
embeddingの標準budget 256000にこの時間短縮率を適用しない。

## ホスト評価後の修正（2026-10-12）

準備完了時のcheckout再ハッシュを1回へ集約した。ビルド前の観測は維持し、
ビルド後はHEAD・plan・新しく列挙したファイル集合とhash・候補source・binaryを確認してから
prepared記録を書き、その記録の整合だけを続けて確認する。測定前後は別途再観測する。
工程をまたぐhashキャッシュは使わず、同じ境界での重複読取りだけを除いた。
準備時間・性能の効果は未測定である。

既存の記録検証では、変更したcheckoutでも準備完了を残す不具合や重複読取りを検出できなかった。
実ファイルとhashを使い、ビルドとGit応答を模擬した検証を1件追加した。
新規・削除・内容変更、HEADや候補sourceの変更で完了記録が残らず、
正常時のcheckout読取りがビルド前後各1回であることを確認する。
準備後のソース・binary・plan変更の拒否も同じ検証で確認する。
旧工程を適用した対照では失敗し、修正版の関連5件は0.037秒で成功した。
小さなファイルだけを扱い、GPU・ビルドを伴わず既存検証にない準備境界を守るため維持する。
削除・統合したテストはなく、失う検出条件もない。実ビルドの並行書込みや不安定さ、
テスト追加分の維持費用・実行時間差は測定していない。

保存された生記録159processを再照合し、vector1360件・pair5932件の差0、順位一致、
cold/warm集計とMetal系列の一致を確認した。測定版との差は準備検査・手順・説明・関連テストで、
Rustの差はrerankerのコメント訂正だけ、Cargo・演算・入力・計測範囲は同じである。
実測集計と失敗記録は書き換えず、新しい推論は行っていない。
既存の独立評価の指摘3件に対し、重複hashの統合、内部経路への範囲訂正、
回答原文の補足を修正として提示する。標準checkと独立評価は新しいrunで行い、過去の成功を引き継がない。

修正担当のoffline lintは依存cache不足で未完了だった。担当AIは同じ固定版と既存cacheで
変更文書を再検査し、原資料との意味・数値・合意範囲を確認した。
軽量検証23件は5.654秒で成功した。lintの技術用語・条件の列挙と否定の指摘、
原文引用を含む合意記録の文長の指摘は、意味を保つため変更しなかった。
上の22件・5.697秒とlint成功はホスト検証時の過去の記録で、修正後の成功へ引き継がない。

今回の実測から区分追加の利益と費用を切り分け、比較と推奨案をまとめた。
製品への追加は見送り、代表的な実入力と主対象optionsでの利益、許容メモリ、公開metrics/consumerの
移行を合意してから採用を判断する。その際はADR-0009の後続判断とsmokeの分類・配列表現を変更する。
本PRではADR-0009のaccepted状態、製品表、consumer、検索品質baselineを変更しない。
