# Issue #318: 入力保持とforward規模を分ける逐次推論案

推奨する最小案は、既存Vec APIを残し、consumer側の小さい処理窓と同期sinkから検討することである。
2026-10-12、Apple M3で8ケースを実測し、修正後にcontrol・deadline・mixedと、embedding専用のfairness・ownershipを再測定した。固定短文50ペアでは25 row窓のallocator peakが
50 row一括より低く、wall中央値は約2%長かった。全入力tokenの保持と1 forward規模を別々に観測できた。
prepared queueによるCPU準備の重なりは0.4 ms前後で、wall短縮の因果説明には足りない。
この入力で明示runtimeや常駐daemon、全consumer移行を採用する根拠は不足している。
窓サイズやhard limit、deadlineは製品へ採用していない。5000ペアのGPU完了、OOM、リーク、
依存更新の回帰、巨大単一documentを含む全CPUメモリ上限は未確認である。
不要なreranker準備を除いた現在のfairness/ownershipは再測定済みである。
[修正後の結果](#embedding専用ケースの準備修正後)を参照する。準備削除そのものの費用削減幅は未測定である。
以下の旧測定値は各保存版の観測であり、修正後の保持条件のRSSや改善幅には流用しない。

## 根拠と開始版

要求と合意の正本は [Issue #318](https://github.com/thkt/rurico/issues/318)。
本文の2026-10-08見直しに従い、#305のCPU依存分離を採用しない。
今回の開始HEADとローカルのorigin/mainは`4d8e9cc640a101b3935ea2ef3edcb43f1ad2d068`で一致した。
Cargo.lockのSHA-256は`d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7`。
担当AIがGitHubのマージ状態と新規cloneを照合し、#392のマージcommitから開始した。

| 出典・版 | 適用する根拠と合意状態 |
| --- | --- |
| [#307報告](../issue-307/report.md)、blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 現在のファイルと完全一致。固定revision、FP32・shape別数値比較、amiciの観測を再利用する。診断基準を新しい製品許容差・全入力の同値性へ昇格させない |
| [ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md)、blob `e96818842c029af9466cb4b0891d10d20d68f489` | 現在と完全一致、thktによるaccepted。128/512/2048/8192のround-upを維持する。4 bucketという規則だけで全batch shapeのcompile cacheやprocess peakが4種類に収まるとは確認できない |
| [ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md)、blob `5c2840daf74b8364a88b37f864ed5717d6bbf403` | 現在と完全一致、thktによるaccepted。MLX専用、candle/CPU backend追加をしない。調査crateもMLXをそのまま使う |
| [現行metrics](../../../src/embed/metrics.rs)・[#306報告](../../benchmarks/issue-306-metrics.md) | host区間、mutex待ち、forward shape、readback観測の定義を使う。wallは包含区間。GPU kernel時間、reranker mutex待ち、CPU/Metal全量をゼロとして補完しない |
| [現行processing](../../../src/embed/processing.rs)・[reranker](../../../src/reranker/mlx.rs) | #332の処理境界を使用。入力の保持とGPU forwardを別に観測する |
| [#364記録](../../benchmarks/issue-364/README.md) | 2026-10-10、M3/24 GiB、開始版`16e5ca97`＋同記録の差分で5000ペアはpressure監視で中断。50ペアの結果を今回の版へ流用しない。保護条件は製品契約の追加許可ではない |
| [ADR-0002](../../decisions/0002-gpu-side-pooling-embed.md)・[cache境界](../../../src/mlx_cache.rs) | GPU poolingとArray解放後の既存cleanupを維持。Rustの解放順序はGPU実測の代用にならない |

Issueにリンクされた`c8f250d`から開始版までを照合した。
Embedderのlock範囲は変わらず、前処理からGPU・cleanup・pauseを覆う。
embedding processingには組立境界の回帰検証と前処理copy削減が入り、metricsにはreadback要素の実観測が追加された。
rerankerのbucket/budget計算はprocessingへ移ったが、全ペアのids/maskを保持する方式は同じである。
Cargo.lockは更新されており、旧版との因果比較を同じ依存条件で行ったことにはならない。

sandboxの初回・修正時には親#296と#306/#332のremote本文をDNS失敗などで取得できなかった。
#306/#332の動作は現行sourceと共有済み報告へ照合した。親#296はホストで原資料を取得し、
更新版・本文hashと実行/保存への適用を[修正後の検証](#修正後のホスト検証と親の条件)へ示す。
取得失敗と最初の証拠不足は履歴として保持し、過去の照合済みという主張だけを受入根拠にしない。

## CPU保持とGPU規模は別の量である

現行rerankerは全ペアをtokenizeし、ids/masksの外側Vecと各token列、全scoreを保持する。
最大長から一つのbucketを選び、固定budget 256000でforwardを分割する。
短文bucket128では2000・2000・1000となる。成功した50ペアと1 forwardの規模が40倍違う。
embeddingも全documentのchunk token、bucket、復元slot、結果を保持する。
一つの巨大documentのtokenizer Encoding/offsetsは、処理窓だけでは制限できない。

既存token budgetはbest-effort hintで、1件がbudgetを超えても1件forwardする。
pauseは待機の要求で、処理時間やメモリの上限ではない。
CPU保持を小さくしても、1 forwardのGPU規模やmodel常駐量が自動的に小さくなるわけではない。

F32、hidden=768、intermediate=3072、sequence=128の単一tensor論理量は次のとおり。
式は`batch × sequence × width × dtype bytes`。shape/dtypeの観測は`shapes`ケースで別に取得する。

| batch | hidden (MiB) | QKV投影 (MiB) | GLU分割前 (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 0.375 | 1.125 | 3 |
| 10 | 3.75 | 11.25 | 30 |
| 25 | 9.375 | 28.125 | 75 |
| 50 | 18.75 | 56.25 | 150 |
| 2000 | 750 | 2250 | 6000 |

これは実際の割当量・同時生存量・必要メモリの合算ではない。
QKVはattention内でreshape/transposition/index/ropeを経て使われ、GLUはMLP内でsplit/GELU/multiplyを経て使われる。
Rustのscope終了と、遅延graphが依存を保持する期間、GPU実allocationの生存期間は区別する。
試作はArrayを追加保持せず、shape/dtype metadataだけを観測する。allocatorのactive/cache/peakはforward後に別に読む。
融合・再利用・OS共有メモリの全生存期間やOOM原因の特定は、これだけでは完了しない。

## 試作の契約と比較対象

[planner](planner.rs)は純粋なCPUの`PreparedBatch`で、doc/chunk IDとids/maskを所有する。
固定bucketへround-upし、row数・padded token数・prepared byte数を超える追加を拒否する。
byte数は予約したRow slotとids/mask payloadを含む。一件だけでも超える場合は拒否し、hintのように1件実行しない。
この上限の適用対象はprepared payloadだけで、入力文字列、tokenizer一時領域、allocator管理量、結果、weights、GPUを含めない。
試作値は比較条件であり、製品のhard limitの合意値ではない。

[host](host.rs)は固定の`("query", "doc")`を遅延生成し、既存のtokenizer/truncate/forward/cleanupを使う。
既存forwardがVec rowを受けるため、consumerにids/mask adapter copyが残る。測定費用に含める。
既存のreranker内部境界だけをcrate内で参照可能にした。公開Vec APIの型と動作は変更していない。
embedder Mutexやglobal cache lockの範囲も変更しない。

順序は入力順で、rerankerは`(doc=pair index, chunk=0)`。複数chunkの一般化ではdoc内のchunk ordinalと最終chunk通知を別に必要とする。
embeddingのchunk生成・prefix/overlapの正しさは既存planner/組立テストを維持する。
この試作は長文のchunk列を遅延生成する製品実装ではない。

queueは`sync_channel(1)`で、一つのproducerが準備し、一つのconsumerがGPUを使う。
consumerはforward結果を同期sinkへ一括で渡し、返却後に次のforwardへ進む。
内部に全結果を貯めない。sinkが遅いとproducerのsendが待ち、backpressureとなる。
prepared payloadの同時保持は最大3 batch（producer/queued/consumer）。
一般plannerのFull返却には最大1 rowのlookaheadが加わる。byte capを超えた巨大rowを準備前から拒否する機能は未実装。
現在の短文実験は固定row数で組むため、このlookaheadをqueueへ持ち越さない。
consumerのadapter copy・CPU padding（ids/maskで8 × rows × bucket bytes）と一時scoreも別に必要となる。

sinkはbatchのatomicな受理を前提とする。部分書込み後の失敗や再送をexactly-onceとしない。
失敗時は受理済みbatch watermarkと失敗ID・原因を保持する設計案で、恒久sinkのtransaction/resumeは未実装。
cancel/deadlineは準備・forward間で確認し、in-flight GPUを即時停止しない。
cancel後にはreceiverをdropして、send待ちproducerを解放してからjoinする。
forwardのErr/abortは実験を失敗として止め、部分実行logを保持する。
[Vec adapter](planner.rs)は完全な件数・順序・有限値だけ受理し、途中Err/不足を成功へ変換しない。
既存Vec互換を選ぶconsumerはO(N)結果を所有するため、streamingの全メモリ上限を約束できない。

| 比較 | 揃える条件と観測 |
| --- | --- |
| retention | 実tokenizer、50/500/5000入力。全ids/masksのcapacity byteと10 row窓の最大prepared byte。GPUはロードしない |
| shapes | 1/10/25/50 row、bucket128。QKV/GLU/hiddenの実Array metadataを別processで観測。timing比較へ混ぜない |
| forward | 同じ50入力を1/10/25/50 rowに分割。3反復、direct scoreとの差とallocator peak。2000 rowを実行しない |
| overlap | 同じ50入力・10 row窓、逐次準備対1-slot queue。3反復・順序反転、実prepare/forward区間、send待ち、wall、peak |
| control | 3反復。期限切れはforward前に0件、cancelは最初の10件後。完全Vecへの変換を拒否し、producer回収を確認 |
| deadline | 修正後の追加ケース、実測済み。逐次/queue各3反復。最初のforward直前に測定用10 msの期限を一度だけ設定し、実forward開始前は期限内、終了時は期限超過、次のforwardなし、cancelなしで10件返却、完全Vec拒否を確認 |
| fairness | 同時到着12 query/4 bulk。bulk先行対query最大2連続。3反復・順序反転、各待ち時間、bulk残数。最大2は未採用の試作規則 |
| ownership | 同じembedderを共有対別instance。2 worker ×4 turn、3反復・順序反転。既存#306 metricsのlock待ち・wallとprocess-global peak |
| mixed | embed/rerankerを同じ呼出元で逐次実行する案対並列実行。3反復・順序反転、wallとpeak。reranker lock待ちは未計測のnull |

ownershipは両embedderを両条件でロードして保持するため、常駐weights量の差を測る条件ではない。
forward/overlapには同じ50ペアのdirect呼出しも各反復の先頭に記録する。
初回directはwarm-upとして別に使う。directが常に先行し、capの順序も固定なので、順序・熱条件の偏りは残る。
同時推論の追加peakと同一mutex待ちを切り分ける。モデル一つ対二つの常駐量は別測定が必要となる。
mixedも実際の専用runtimeを採用した測定ではなく、単一所有者の逐次dispatchの費用を調べる対照である。
queryが持続到着する一般の公平性、consumer別SLA、すべてのモデル組合せは未確認。
固定短文のscoreは同値入力なので、実モデルの出力だけで順序取り違えを検出できない。
順序は異なるtoken識別値を使うpure試験と既存rerankerの順位・入力順試験で補う。

## 最小導入案と再開条件

第一段階ではdirect Vec APIを維持し、consumer側の小さい窓と同期sinkで資源量を測る。
公開APIにはまだ新しいoptionsを追加しない。既存呼出しのtokenizerやcache方針を変えず、効果を比較する。
第二段階は実測で保持量の問題を確認した場合に、crate内のprepared境界とbyte/token capを導入する案である。
prefix/overlapとdoc/chunk復元を既存境界で保ち、外側入力と巨大単一documentへの事前拒否条件を用途別に合意する。
第三段階の明示runtimeは、model/GPU/cleanupを一所有者へ寄せる案として比較する。
全processのcacheまで排他的に所有するには既存direct APIとの共存ルールが必要で、単なるmutex移動では所有権を保証しない。
daemon・全API移行・別backend/crate分離は自動採用しない。

5000ペアのGPU再実行は、まず50ペア以下の反復でshape・出力・peak・回収を確認してから判断する。
反復で未完了・圧迫・継続増加があれば、入力保持とforward scaleのどちらが増えたかを分けて調べる。
小さいforwardの5000逐次実行と、旧2000-row forwardの再現は別の検証条件である。
上限を緩める、timeoutを延ばす、ignored試験を削除するだけで再開・成功とはしない。
同じ保護条件で拡大できなければ、原因と代替案を示し、実行上限の変更が必要な時だけ人へ戻す。

次の値は未合意で、測定値から都合よく確定しない。

| 対象用途 | 採用前に人が判断する事項と影響 |
| --- | --- |
| 対話検索 | 許容query待ち時間・全応答deadline、bulkとの公平性。bulkを後回しにする範囲で索引更新の遅延が変わる |
| 大量index/rerank | 許容process/GPU常駐量、入力byte/token/単一doc cap。上限超過の拒否か分割かで完了件数が変わる |
| 永続sink/再開 | chunkだけの部分完了を許すか、doc単位atomicとするか。途中失敗後の公開・再送・重複処理が変わる |
| 複数model/consumer | model切替、同時常駐、admissionの主体、direct APIとの共存。常駐量と待ち時間の配分が変わる |

今回はこれらを採用せず、質問を具体化する調査範囲を維持する。
方式採用が必要となる段階では、人へ推奨案と実測・代替案を示す。

## 履歴: sandboxでの初回準備時点の検証

この節はホスト検証より前の状態であり、現在の追加作業要求ではない。
その後のビルド・標準check・lint・小規模実測は次の二節、復帰後の修正と残る検証は末尾のassessments/handoffに示す。

2026-10-12のsandboxでは、Rust 1.99.0でpure plannerの3テストが成功した。
single-row超過拒否、bucket増大前のflush、byte cap、入力増加時のpayload上限、ID/順序、途中失敗・不足・非有限、
cancel/deadline判定とquery最大2連続の選択を確認した。標準checkにも同じpure試験を含める。
超過をFullへ読み替える、途中Errを捨てる、queryを無制限に優先する3つの改変を
checkout外の比較用コピーへ個別に入れ、対応する試験がassertionで失敗することも確認した。
実際のchunk生成やGPU failure cleanupは既存検証を残し、同じ保証を複製しない。
scopeの違うhard limit・sink transaction・GPU即時cancelの保証は増やしていない。

新しいscope付きmetadata観測の試験は、他threadへ有効化を漏らさず、終了後に無効へ戻すことを確認する。
これはホストの標準checkで実行するmodel-free Array試験で、sandboxでは実行していない。
数値・単位・参照・合意状態と、codeとの差を照合して文書を推敲した。
調査crateの`cargo metadata --offline --locked`は成功した。test targetの`cargo check`は
MLXのCMake取得先をDNS解決できず失敗し、Rust側の全体コンパイルは未確認である。
製品と調査crateのformat確認、Python runnerのsyntax確認、Git差分の空白検査は成功した。
固定版natural-japaneseのクイック手順を参照したが、lintはuv cache書込み拒否の後、
書込み可能なcacheでoffline実行してもSudachiPy未配置のため失敗した。
同版の手動checklistで意味と語順を照合した。lint未実行を指摘0件や手順の完了とは扱わない。
変更文書を含む独立評価、全体check、同じheadのCI、上表の実モデル受入検証は未完了である。

実行手順と証拠の期待値は[README](README.md)。通常checkは実モデルケースを実行せず、capture媒体も不要なので、
担当AIによる追加ホスト検証が必要となる。測定後は既存報告へ版・数値・条件・限界を追記し、同じ差分基準で独立評価へ戻す。
過去の#307/#306/#364証拠はその版の結果として残し、新しい成功結果へ読み替えない。

## 履歴: 2026-10-12の初回ホスト準備

担当AIは親#296の最新本文を取得して実行・保存条件を照合したと報告した。
最初の提供証拠には本文・updatedAt・適用条項がなく、独立評価では未確認とされた（R1-5）。
下記の再検証では原資料も保存してこの不足を補った。
Apple M3・24 GiB上で、固定revisionのembedding/reranker各3ファイルをHF cacheのETagと内容hashへ照合した。
モデルを再取得していない。この確認は推論・性能測定ではない。

研究用crateの最初のClippyは、plannerの重複module登録とabsolute_paths違反の計13件で失敗した。
plannerの登録を一つに統一し、importへ整理した修正版のClippyは終了0だった。
研究用release libtestのビルドも成功した。コメント整理後の最終binaryは別のビルド記録で確定する。
既存の標準checkは428件成功、doctest・Clippy・formatまで終了0を確認した。
既存のvisibility検証はcold buildでSLOW表示になったが、条件を変えず完了した。
planner 3件とmetadata観測のthread分離1件を含み、実モデルのignored検証は実行していない。

変更した既存ファイルのコメントを点検し、処理を言い換えるだけの2行を削除した。
rerankerの説明は、全pairのtoken・score保持とforward分割の範囲を明示する内容へ短縮した。
数値・Metal固有の制約、公開契約、cacheの理由を説明するコメントは保持した。

固定版natural-japaneseのlintをホストで実行し、文章のリズム・語彙反復と並列列挙の指摘を確認した。
専門用語と検証範囲の対応を優先し、指摘だけを理由に条件を省略しない。
上記は測定前の準備結果である。その後の実モデル8ケースの結果を以下へ示す。
この時点では独立評価・PR公開・同じheadのCIは未完了だった。現在の状態は修正後の節とPRの公開記録を参照する。

## 履歴: 修正前の実モデル8ケースの結果

以下は[environment.json](results/environment.json)に固定した修正前sourceの観測である。
R1-1/R1-2修正後のmixedとdeadline、embedding準備修正後のRSSの証拠へ流用しない。
fairness/ownershipはembeddingを推論したが、旧contextは未使用のrerankerを識別し、余分なtokenizerを保持した。
その誤識別・準備条件を隠すために元JSONを書き換えず、旧版の証拠として保持する。
fairness/ownershipも不要なreranker tokenizerを保持した旧条件であり、末尾の追加修正後の受入証拠ではない。

ユーザーから時間の必要な検証を任意のタイミングで行う許可を受け、担当AIの別ビルド・GPU作業を止めて実行した。
既存desktopアプリは稼働したまま、電源・熱状態は未計測である。測定前のpressureは正常だった。
固定モデル・source・binaryの前後hashが一致し、8ケースすべて終了0、所有process groupの不在を確認した。
監視488サンプルのRSS最大は3001.88 MiB、pressureはすべて1、swap使用量の観測増加は0だった。
RSSは0.1秒を目安とするsampleの最大で、瞬間peakやGPU/共有メモリ全量ではない。

[環境・版・監視要約](results/environment.json)、[保存観測](results/retention.json)と各ケースのJSON、
[集計](results/summary.json)を共有した。個人pathと巨大な重み、生のprocess一覧は公開しない。
[summarize.py](summarize.py)は件数・ケース条件を検査し、読んだ観測JSONのSHA-256を計算・記録する。
保存された期待hashとの照合は行わない。元証拠との一致確認には、再生成した集計の`inputs_sha256`を
保存済み[summary.json](results/summary.json)と別途比較する。[READMEの再集計手順](README.md#保存観測を再集計する)に比較方法を示す。
case別JSONは元の保存観測とbyte単位で一致する。各ケースは別processであり、peakを足さない。

### 全入力保持とforward規模

実tokenizerの短文pairでは、全ids/masks保持量が50/500/5000件で4800/48000/480000 bytesとなった。
10 row窓の最大prepared payloadは全件数で960 bytesだった。外側入力・tokenizer scratch・結果・weightsは含まない。
queueの同時保持は最大3 batchなので、payloadだけの実装上の上限と総RSSを混同しない。

rerankerの実Array metadataはF32、bucket128で、hidden/QKV/GLU分割前のwidthが768/2304/6144だった。
1/10/25/50 rowの論理shapeを確認した。単一tensorの表はこの観測に合うが、融合・allocatorの同時生存量の証明ではない。

| 同じ50 pairの実行 | wall ms 最小 / 中央値 / 最大 | allocator peak MiB（3反復とも同値） | directとの最大score差 |
| --- | ---: | ---: | ---: |
| direct API相当 | 1709.136 / 1718.831 / 1752.880 | 2899.375 | 比較基準 |
| 1 row窓 | 2038.906 / 2066.578 / 2100.048 | 1446.470 | 9.8720193e-8 |
| 10 row窓 | 1761.240 / 1781.873 / 1809.416 | 2498.724 | 0 |
| 25 row窓 | 1719.481 / 1755.294 / 1768.381 | 2474.356 | 0 |
| 50 row窓 | 1706.777 / 1722.807 / 1738.760 | 2899.375 | 0 |

25 row窓はdirect中央値に対しwall約2.12%増、allocator peak約14.66%減だった。
1 row窓はpeakが小さい一方、wall約20.23%増。10と25のpeakは単調でなく、単純なrow数比例を仮定できない。
directは各反復で先行し、窓の順序も固定した。初回warm-upを除いた3反復の範囲で、因果的な改善率・一般の最適値は確定しない。
score差は測定前の#307診断基準内だった。同値短文だけなので、これを全入力の順位・検索品質の保証にしない。

### CPU/GPU overlapと制御

10 row窓の逐次準備はwall中央値1802.742 ms、queueは1760.501 msで、観測差は約2.34%。
同じallocator peakは2498.724 MiBだった。実際のprepare/forward区間の重なりは
0.363/0.434/0.480 ms、準備の総時間は0.447/0.513/0.571 msだった。
42 ms前後のwall差をこの重なりだけでは説明できない。短文ではCPU準備の比率が小さく、queue導入の利益は未確定である。

期限切れは3反復とも0件を返し、GPU forward前に止まった。最初のforward後のcancelでは各10件まで完了し、
50件の完全Vecへの変換を拒否した。cancel実行のwallは350.369/364.153/525.078 ms。
GPU実行中の即時中断、実行中に到来する期限までの厳密終了、永続sinkの部分書込み回収は保証しない。

### 公平性・複数instance・複数model

同時到着12 query/4 bulkの試作dispatchでは、queryのqueue待ち中央値はbulk先行で555.998 ms、
query最大2連続で429.340 msとなった。bulkは143.043 msから326.030 msへ増えた。
値は3試行のquery36件/bulk12件をまとめた記述統計で、独立標本や有意差としない。
3試行すべてで全taskが完了した。持続到着・一般の公平性・consumer SLAは未確認である。

同じEmbedderを共有した24呼出しのlock待ち中央値は58.774 ms、別instanceでは0.000292 ms。
最大はそれぞれ92.275/0.001959 msだった。allocator peak中央値は共有2644.427 MiB、別instance2760.319 MiB。
両条件で2モデルをロード・保持するため、1モデル常駐と2モデル常駐の資源差ではない。
並列の速さやcache race不在をこのlock値だけで断定しない。

embed/reranker混在のwall中央値は逐次83.560 ms、並列78.532 ms、allocator peak中央値は2646.682/2717.808 MiB。
逐次の範囲72.635–118.580 msと並列70.938–79.167 msは重なる。
同時実行時にpeakが増える観測はあるが、専用runtimeの全費用・常駐切替・raceの不存在を証明しない。
rerankerのlock待ちは未計測のnullを保持し、ゼロに補完していない。

### 結論と残る判断

この条件ではprepared payloadは入力件数と独立に保て、forward分割の資源・時間のtrade-offを観測できた。
第一候補はconsumerの小さい窓と同期sinkで、queue・明示runtimeの一括導入は見送る。
25 rowはこの固定短文の比較点であり、製品default・単一入力のhard limitへ採用する値ではない。
先に示した用途別の許容メモリ・deadline・部分完了の判断と、実入力での比較が採用前に必要である。
5000 pairのGPU完了と長文の全CPU上限は未確認として残し、過去の中断を成功へ読み替えていない。

## 修正後のホスト検証と親の条件

この節はresults-revision1に保存した過去の検証である。追加修正の対象版と未完了事項は末尾へ示す。

修正後の測定対象は開始commitとその時点の未commit差分、固定lock、host.rs SHA-256
`72e547f7adde5a0673fcc928a8221672a150cecc5882f06c9e0de08435eb72d9`、run.py SHA-256
`5b3939d56fd6eea5e7f53e68420844f95465de6920c13f0ff330d06c096080be`である。
元の8ケースは上書きせず、[修正後の環境](results-revision1/environment.json)と
[control](results-revision1/control.json)・[deadline](results-revision1/deadline.json)・
[mixed](results-revision1/mixed.json)を別保存した。JSONはcheckout外の生観測とbyte一致する。
正常測定ではsource/model/binaryの前後hashが一致し、3ケースとも終了0、所有groupの不在を確認した。
監視67サンプルのRSS最大は2894.469 MiB、pressureはすべて1、swap使用量2481.56 MiBの増加は0だった。
RSSは標本最大であり、瞬間peak・GPU共有メモリ全量・OOMやリークの不存在を保証しない。
ビルド・変異検証と実測は順番に行い、担当AIの別ビルド/GPU処理を測定と重ねていない。

親[#296](https://github.com/thkt/rurico/issues/296)はホストのGitHub CLIで再取得した。
`updatedAt=2026-10-07T17:08:32Z`、本文SHA-256
`5918980b98047e009322bf25711d01b7aab98802c826719ef3cc0bcab7d00ee6`で、初回ホスト取得本文と一致した。
[版情報](results-revision1/parent-296-version.json)を共有し、本文全文・原文条項・取得JSONをホスト証拠へ保持した。
「セットアップ・検証・成果物」の原文に従い、次を今回の実行へ適用した。

| 親の条件 | 今回の対応 |
| --- | --- |
| 実装時main/lock・Rust・機種/OS・model/tokenizer revisionの固定 | 開始版4d8e9cc、lockの上記hash、M3/24 GiB、macOS/Xcode/Metal/Rustと6固定ファイルのrevision/hashを環境JSONに保存 |
| Apple Siliconと対応Metal Toolchain、既存MLX構成 | 既存mlx-rs 0.32.0/mlx-sys 0.6.0をそのまま使い、依存分離・追加backendなし |
| cargo fetch --locked、標準checkと変更に必要な追加検証 | trusted CLIのsetupで依存を取得。標準checkと研究crateのClippy/release build、明示したignored実モデル試験を区別 |
| 同じ入力・重み・options、比較軸・warm/cold・順序・shape/forward・wall・RSS/Metal・ばらつきの記録 | 50固定短文pair、310m固定重み、bucket 128、順序と各3反復の生記録。MLX allocator peakとRSSを別表示。電源・熱は未計測、因果的改善を断定しない |
| 報告/再現例はPRへ、巨大生出力は承認保存先へ、私的入力・認証情報は公開しない | docs/research/issue-318に合成入力の観測・版・報告を保存。重み・個人path・process一覧は公開せず、ホスト生logはcheckout/旧run外の承認された保存先に保持 |
| 未確定の許容メモリ・品質・deadlineを都合よく採用しない | 既存#307診断基準と調査打切り条件を使い、製品上限/SLAや方式採用へ転用しない |

### 期限境界と検出力

開始前期限切れは3反復とも0件、cancelは3反復とも10件で止まった。
追加deadlineは逐次/queue各3反復で、forward開始時の残り時間9.994–9.999 ms、
終了時の超過313.214–362.715 msを観測した。各10件だけを返し、cancelなし、次forwardなし、
完全50件Vecへの変換拒否を確認した。10 msは測定専用で、実行中GPUの即時中断や製品deadlineではない。

[変異記録](results-revision1/mutations.json)は独立コピーだけに適用した2条件である。
queue consumerまたは逐次loopの期限検査を初回に限定すると、いずれも次のforward直前の
`measurement deadline expired before forward`でlibtestが終了101、runnerが終了1になった。
ビルドは成功、guard停止なし、所有groupは回収済みで、これらを成功した測定へ読み替えない。
正常版の復元後は[deadlineの6条件](results-revision1/restored-deadline.json)が終了0で再度成功した。
変異ごとのsource/binary hash、argv・stderr・終了コードとgroup不在はホスト生logへ保存した。
非変異source・モデルのhashは検証系列の終了後にも一致した。失敗runnerはcomplete.jsonを生成していない。
本checkoutへ変異を入れていない。初回応答の別3変異は生log不足のため検出実績へ加算しない。

旧controlは開始前期限切れとcancel/backpressure回収を守るので、追加deadlineへ置換しない。
pure planner 3件はpayload上限・順序・途中失敗・bulkの選択機会を、小さいmodel-free試験で守る。
metadataのscope/thread分離、既存chunk/prefix/overlap・reranker順位・cleanupの保証も保持した。
同じ判定をなぞる常設試験や別の検証基盤を追加していない。

### 重複準備修正後の混在比較

同一のreranker VerifiedArtifactsをcontextとモデル構築で共有し、二度目のcache検証とtokenizer保持を削除した。
embeddingのcache検証、モデル構築時の全重み検証と前後hashは維持する。
wallとallocator peakは各条件のmixed_peak 3記録から最小/中央値/最大を求めた。

| 条件 | wall ms（最小 / 中央値 / 最大） | allocator peak MiB（最小 / 中央値 / 最大） |
| --- | --- | --- |
| 逐次 | 79.285 / 87.372 / 121.891 | 2646.682 / 2646.682 / 2646.682 |
| 並列 | 70.885 / 71.602 / 76.949 | 2702.308 / 2706.880 / 2717.011 |

モデル準備はこのwall区間より前である。冗長な準備の削除をwall短縮の根拠にしない。
小さい3反復と背景負荷の条件から、一般の並列速度改善・race不在・常駐切替費用は確定しない。
旧mixedは別tokenizerを余分に保持した準備条件なので、新しい版のRSS証拠へ流用しない。

### 単一rowのbyte超過を独立に検出する

追加の静的評価で、既存のOversize否定試験がtoken/byteの両上限を同時に超え、
byte判定を失っても別の理由で通る見逃しが確認された。既存の3件の軽量試験内へ、
tokenとrow上限には収まり、単一rowのpayload 40 bytesだけが許容量32 bytesを超える条件を追加した。
空batchと既存rowがあるbatchの双方でOversizeを確認し、拒否後のbyte量・bucket・ID/token/mask保持も確認する。
複数rowの合計超過がFullとなる既存対照は維持した。

正常版のpure 3件は終了0。独立した一時コピーで単一row byte判定だけを除くと、
追加したbyte-only Oversize assertionが失敗して終了101となった。ビルドは終了0で、検出条件を確認できた。
変異は本checkoutへ適用していない。この追加はcfg(test)内の試験だけで、非testのplanner sourceはbyte一致する。
旧測定のplannerファイル全体hashと現在版は異なるが、推論・保持・dispatchの処理と実モデル入力は変更していない。
測定証拠の版は上書きせず、この適用条件で再利用し、実モデルを再実行していない。

## embedding専用ケースの準備修正後

再評価対象`0592af05414487a43dd1c0f5c758e5619bb75a7dcb8738646d2835522c2049b7`で指摘された
R1-1は、以前のmixedのR1-1と異なり、embedding専用ケースの不要なreranker準備である。
fairness/ownershipをreranker準備より前へ分岐し、各関数が取得したembedding VerifiedArtifactsを
contextとEmbedder構築で共有した。embeddingの存在/config/tokenizer/kind検査とロード時の全重み検証を残し、
rerankerを使うケースの検証、runnerの6モデルファイルとsource/binaryの前後hashも保つ。

[results-revision2の環境・版・監視要約](results-revision2/environment.json)と
[fairness](results-revision2/fairness.json)・[ownership](results-revision2/ownership.json)を別保存した。
host.rsの測定SHA-256は`efe1e644ebabc2f4c5510ed1dcea790f28a74ad064232b13e397faf598c5c0f6`。
両contextは実際の推論対象`cl-nagoya/ruri-v3-310m`、revision
`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`を示す。新binaryの2ケースは終了0、
source/model/binaryの前後hashが一致し、所有groupの不在を確認した。
監視68サンプルのRSS最大は2787.188 MiB、pressureはすべて1、swap 2481.56 MiBの増加は0だった。
各JSONは元の観測とbyte一致し、モデルの重量や個人pathを一括公開していない。

fairnessは3試行のquery 36件/bulk 12件を条件別にまとめたqueue_waitの記述統計である。
最大2連続は未採用の試作規則で、独立標本数・有意差・consumer SLAを意味しない。
96 taskすべてが完了した。持続到着時の一般の公平性は未確認である。

| 条件・task | queue待ち ms（最小 / 中央値 / 最大） |
| --- | --- |
| bulk先行・query | 304.748 / 545.504 / 784.604 |
| bulk先行・bulk | 0.000 / 125.190 / 274.279 |
| query最大2連続・query | 0.000 / 430.998 / 790.367 |
| query最大2連続・bulk | 82.051 / 331.170 / 584.283 |

ownershipは条件別24 callのlock待ちと3試行のallocator peakを集計した。
両条件で2モデルをロード・保持するため、1モデル対2モデルの常駐費用の比較ではない。

| 条件 | lock待ち ms（最小 / 中央値 / 最大） | allocator peak MiB（最小 / 中央値 / 最大） |
| --- | --- | --- |
| 同instance共有 | 0.000250 / 58.411771 / 97.867208 | 2644.364 / 2644.427 / 2644.427 |
| 別instance | 0.000083 / 0.000271 / 0.000833 | 2755.928 / 2757.000 / 2767.241 |

query優先でquery待ちは小さく、bulk待ちは大きくなった。別instanceのmodel lock待ちは小さかったが、
この値だけで並列速度やcache race不在を保証しない。各表はJSONの条件別値から最小・中央値・最大を求めた。
不要な準備の削除はコードで確認できるが、その削減費用を個別に測っておらず、
旧観測とのRSS差やwall差をその効果へ帰属しない。背景desktopアプリ、電源・熱未計測と小さい3反復の条件を維持する。

## assessmentsと残る確認

2026-10-12の要約修正では、開始版`4d8e9cc`を差分基準に、停止記録・修正生応答・評価履歴と追加証拠を
現在の成果物へ再照合した。測定版のsource hashは現在と一致し、fairness/ownershipの公開JSONは
生stderrの観測と対応する。`passed`は実行の主張として扱い、以前の受入を引き継いでいない。
前の修正では結果節を追記して冒頭の未実施状態を残したため、状態の混在が再発した。
両文書の冒頭を測定済みの状態と結果節への参照へ更新し、再実行手順と未実施の要求を区別した。

同じR1-1でも評価対象が異なるため、参照元の対象版と対応する根拠を分けて評価した。

| 参照元target・指摘 | 現在の成果物と保存証拠による再評価 |
| --- | --- |
| `d271a063` R1-1: mixedの重複準備 | 解消。`mixed_ownership(&artifacts)`はcontextで使ったreranker成果物を借用する。[混在比較](#重複準備修正後の混在比較)はその保存版の結果で、最新binaryでの再実行ではない |
| `d271a063` R1-2: 実行中の期限到来 | 解消。[期限境界](#期限境界と検出力)の正常・逐次/queue変異・復元記録で、期限内開始、期限超過終了、後続forward停止、10件の部分返却と完全Vec拒否を確認した。GPU即時中断の証拠ではない |
| `d271a063` R1-3: hash保証の説明 | 解消。[再集計手順](README.md#保存観測を再集計する)はhashの計算・記録と保存集計との別途byte比較を区別する。保存期待値自体の真正性までは保証しない |
| `d271a063` R1-4: 過去と現在の状態混在 | 元のsandbox節は履歴化済み。ただし同種の欠陥が新しい再測定の冒頭要約へ再発したため、下記`2a74ede1` R1-1として今回修正した |
| `d271a063` R1-5: 親#296の適用条件不足 | 解消。[親の条件](#修正後のホスト検証と親の条件)の取得原資料、更新版・本文hashと実行/保存条項を照合した。標準checkと追加実モデル比較、非公開生logと公開要約を区別する |
| `0592af05` R1-1: embedding専用ケースの準備 | 解消。reranker準備前の分岐とembedding成果物の共有を現行sourceで確認した。[results-revision2](results-revision2/environment.json)は同じsourceで96 task、48 callと6 peakを記録し、embeddingのmodel/revision、生観測と公開JSONが対応する |
| 形式不正だった評価のbyte-only試験不足 | 解消。[単一rowのbyte超過](#単一rowのbyte超過を独立に検出する)はtoken/row上限内の独立入力を使い、byte判定だけを除く変異が該当assertionで失敗する。形式不正応答を有効な受入へ読み替えない |
| `2a74ede1` R1-1: 冒頭の再測定待ち | 今回修正。上記のsource・context・生観測との照合に基づき、READMEと本報告の冒頭を測定済みへ更新した。文書修正後のcheck・独立評価はこれからである |

直前のembedding準備修正は上記の追加測定で確認した。以前のmixedでのVerifiedArtifacts共有、
control/deadlineの停止・部分返却・完全Vec拒否、2期限変異と正常版復元は保存証拠を再照合した。
その実装修正はembedding専用の分岐・contextに限られ、stream/forward・mixedの処理は不変である。
その保存版の境界証拠を再利用し、results-revision2のbinaryで再実行したとは報告していない。
byte超過の独立入力と変異検出、親#296の原資料・更新版・適用条項も保持した。
旧8ケースの再集計byte一致とhashの計算/期待値照合の区別、過去と現在の状態の区別を維持する。

embedding準備修正後のホストでは研究crateのClippy・release buildは終了0。標準checkも終了0、
Rust 428件成功/28件skip、Python 14件、doctest 3件、Clippy・format成功を確認した。
当時の改訂報告はホストで同じ固定版の日本語lintを実行し終了0。文長のリズム・語彙反復と並列列挙の指摘を
数値・条件・原資料へ照合した。指摘数を意味の一致の保証へ読み替えない。
修正時のsandbox lintはoffline依存解決でsudachipyがなく終了1となった履歴を残し、成功へ読み替えない。
今回の要約修正版の日本語lintもoffline依存解決でsudachipyがなく終了1となり、未完了である。
固定版の手動チェックリストで変更文と段落先頭を確認し、測定版・状態・数値・条件を原資料へ照合した。
今回の変更後の全体checkは未実行で、文書を含む独立評価もこれからである。commit・公開は行わない。
既存pure 3件、control/deadline、metadata、chunk/prefix/overlap・順位・cleanupの異なる検出条件は維持し、
分岐の配置をなぞる常設試験は追加していない。旧run・失敗生log・元JSONを変更せず、受入を引き継がない。

### handoff

今回の変更は文書の要約と評価・引き継ぎの更新だけで、実装・入力・測定条件を変更していない。
修正後のfairness/ownership測定をこの文書修正だけで再実行する必要はない。
設定済み`bash scripts/check.sh`と変更文書を含む独立評価へ戻す。captureは不要で、契約は変更しない。
5000ペアの調査を再開する場合、小さいforwardでの逐次完了と旧2000-row forwardの再現を別条件で扱う。

5000ペアのGPU完了、OOM/リーク、巨大単一文書の全CPU上限、永続sink回復、持続到着時の公平性、
GPU即時中断は未確認として残す。採用段階の許容メモリ・deadline・部分完了公開単位・consumer配分・
direct API共存条件は人が判断する。今回の窓・10 ms・保護条件を合意値へ転用しない。
