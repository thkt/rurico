# Issue #310: mask生成とcache policyの比較結果

全体のpolicy変更は見送る。合成maskのforward内共有は長文embeddingでメモリと時間を減らしたが、
512境界ではMetal peakが320 MiB増え、rerankerのmixedでは速度改善を再現できなかった。
長文に限定した共有が次の検討候補であり、今回の変更は比較入口と証拠の保存までである。
GPU生成・buffer保持・compile clear周期変更にも全体採用を支持する効果は得られなかった。
独自Metal/window kernelはattention単独の支配性を確認できていないため保留する。

[実測値と分布](results/comparison.json)には17ケース・5候補・各3process、計255processの
3,285 callを保存した。製品bucket、F32、公開API、cleanup policyは変更しない。

## 出典・版・適用条件

要求と合意は[Issue #310](https://github.com/thkt/rurico/issues/310)、
受領した更新時刻は`2026-10-07T17:05:14Z`。確認日は2026-10-13。
開始HEADとローカルorigin/mainはともに`4cd349961e5d656dac0e272ec8d9b6573a854da3`。
Cargo.lockのSHA-256は`d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7`で変更しない。
実装担当のsandboxではオンライン取得に失敗した。担当AIがホストで最新mainと親Issue #296を
読み戻し、開始HEADとの一致と検証・保存条件を2026-10-13に確認した。#322は過去の互換性修正で、
今回の環境では5候補のビルド、全mask照合、実モデルのcleanupと出力比較が成功した。

指定された次の資料は、startCommitのblobと現在のファイルがすべて一致した。
観測や他Issue限定の合意を今回の採用規則に昇格させない。

| 出典 | 照合したblob | 今回の適用条件と合意状態 |
| --- | --- | --- |
| [#307報告](../issue-307/report.md) | `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 固定310m/F32の公式数値・amici品質の過去実測。今回のmask/cache効果や別入力の品質は未確認 |
| [#308報告](../issue-308/report.md) | `5fb0fbf4013aeeefb4b85c117626bbd739a49560` | bucket追加の独立比較と見送り。runner・入力・出力照合を利用する。数値やreranker比較許容差の承認は#308限定 |
| [ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md) | `e96818842c029af9466cb4b0891d10d20d68f489` | acceptedの内部4 bucket invariant。公開ModernBertの任意seq_lenを4件に制限する実装ではない |
| [ADR-0010](../../decisions/0010-adopt-maskfill-constant-to-avoid-metal-kernel-nan.md) | `d45f91d4701e40fe96b89845cc7ab4f5e63200df` | acceptedのglobal mask乗算用-1e9。localの直接選択には-infを維持し、0×-infを導入しない |
| [ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md) | `5c2840daf74b8364a88b37f864ed5717d6bbf403` | acceptedのMLX-only条件。別backendの結果で置き換えない |

Issue本文の`c8f250d…`から開始HEADまでに[#311](../../benchmarks/issue-311/README.md)のtoken準備、
推論境界の検証、#359の実測記録、#318/#308の観測入口が更新されている。
[現行forward](../../../src/modernbert/model.rs)、[cleanup](../../../src/mlx_cache.rs)、
[embed計測](../../../src/embed/mlx.rs)へ照合した。CPU local mask・層ごとの合成・両cache clearは現行にも残る。
旧APIをコピーせず、#306由来の現行metricsと実readback量を利用する。
[#303の実測](../../benchmarks/issue-303-cleanup-memory.md)は当時の93試行の証拠として保持し、
今回の候補の成功へ読み替えず、同じ現行テストを全候補で再実行する。

## lock固定APIの調査結果

Cargo.lockのmlx-rs 0.32.0 / mlx-sys 0.6.0を使用する。
mlx-sys配布ソースの`src/mlx-c/CMakeLists.txt`はMLX v0.32.2を固定している。
インストール済み同版のRust sourceと取得済みMLX sourceを読んだ。
再現時には[mlx-rs 0.32.0](https://docs.rs/crate/mlx-rs/0.32.0/source/src/memory.rs)、
[MLX v0.32.2 compile](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)と
[Metal allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)から追える。

buffer poolはprocess全体のallocator cacheで、live weights・modelのlocal maskを解放する操作ではない。
`memory::set_cache_limit`はallocator poolの設定、`active_memory/cache_memory/peak_memory`はallocator観測である。
Metal allocatorは返却時にpoolへ入れ、割当時に上限を超えたcacheを回収する。
そのためlimit変更だけでは瞬時のcache削減やGPU全資源のhard limitを保証しない。
候補はcleanup境界の追加観測も使う。

compile cacheは`static thread_local`のcompiled function cacheで、FFIは現在threadのhandleを取得してclear/freeする。
Metalのpipeline/kernel cacheとは別である。現在の`src`にはcompile transformの利用が見つからず、
この変更の効果は主にclear操作費用に限られる可能性があるが、後述の実測では明確な全体改善を確認できなかった。
entry数・bytesのgetterやbyte上限を提供するAPIは今回確認したRust bindingには見つからなかった。
8 forward周期は保持期間を制限する調査案で、byte上限の証明ではない。

GPU local maskは`Array::arange::<_, i32>`、reshape/subtract/abs/le、`ops::select`で構築できる。
localの0/-infとglobalの-1e9を維持する。生成をGPU側へ移してもdense F32の長さ²の容量は残る。
8192²×4 bytes = 256 MiBは計算値であり、host/Metal peakの実測ではない。

[提示されたPython SDPA文書](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
は参照時MLX 0.32.3であり、Rustで使えるAPIの証明にしない。
scoutはDNS失敗し、代替のweb取得で本文を確認した。
lock固定Rust `fast.rs`のmask指定はArrayまたはCausalで、双方向window引数はない。
これだけで全backendにwindowがないと断定しない。独自kernelやcausal置換は提案していない。

## 実測条件と数値の照合

| 条件 | 版・設定 |
| --- | --- |
| ホスト | Mac15,3 / Apple M3 / 24 GiB |
| OS | macOS 27.0.1 (26A434) |
| Xcode・Metal | Xcode 27.0 (27A266a) / Metal 32023.921 |
| Rust・Cargo | rustc 1.99.0 / Cargo 1.99.0 |
| ビルド | release・locked・test-mlx,smoke / jobs 2 |

固定310m/F32のモデルrevisionと3ファイルのhashを#307 manifestへ照合し、各測定の前後にも確認した。
embedding revisionは`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`、
rerankerは`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`。
内部bucketは128/512/2048/8192、入力・batch・budget・pauseは[再現条件](README.md)を維持した。

各候補は独立processを3回使い、通常9 call、mixed/multichunkは31 call。
初回はモデルload・入力準備後のcold、残りはwarmで、OS file cacheを空にしたcoldではない。
以下は各processのwarm中央値から取った3値の中央値（ms）。min/maxとcold、
Metal peak、RSS、cleanup後メモリ、観測回数はJSONにある。反復callを独立標本として水増ししない。
順序は候補を0/2/4ずらして循環させたが、全順列を均等に試す設計ではない。

| ケース | current | shared | gpu | buffer | compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| embedding/boundary1024 | 1943.20 | 1916.71 | 1923.05 | 1955.41 | 1912.09 |
| embedding/boundary128 | 214.09 | 202.79 | 206.00 | 197.70 | 203.97 |
| embedding/boundary2048 | 6211.96 | 6109.46 | 6226.38 | 6365.86 | 6204.52 |
| embedding/boundary4096 | 14461.58 | 14141.35 | 14450.19 | 14733.23 | 14462.16 |
| embedding/boundary512 | 900.35 | 897.32 | 907.91 | 901.70 | 902.10 |
| embedding/limit | 19006.77 | 18852.79 | 19734.30 | 19831.33 | 20414.18 |
| embedding/mixed | 10888.95 | 10676.97 | 10886.69 | 10992.27 | 10881.79 |
| embedding/multichunk | 14816.61 | 14477.79 | 14803.53 | 14967.35 | 14847.34 |
| reranker/boundary1024 | 1902.92 | 1888.34 | 1906.20 | 1954.84 | 1902.49 |
| reranker/boundary128 | 389.72 | 387.53 | 390.29 | 388.65 | 389.07 |
| reranker/boundary2048 | 14157.49 | 13894.90 | 14172.13 | 14461.34 | 14163.66 |
| reranker/boundary4096 | 14191.43 | 13921.07 | 14195.51 | 14471.58 | 14169.05 |
| reranker/boundary512 | 1900.51 | 1885.73 | 1901.92 | 1944.53 | 1897.27 |
| reranker/empty | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| reranker/limit | 14206.43 | 13916.87 | 14176.73 | 14491.46 | 14212.02 |
| reranker/mixed | 6867.01 | 6911.48 | 6949.34 | 7240.70 | 6888.16 |
| reranker/same | 1500.53 | 1498.61 | 1502.76 | 1527.60 | 1502.28 |

emptyの値は表示桁では0.00 msとなり、モデルforwardの高速化の根拠には使わない。
全ケースでembeddingの文書全要素と各process末尾のquery、rerankerのlogit/score差は0だった。
文書/chunk ID・入力token・shape・有限性・実readback量を照合し、rerankerの順位と同点順も一致した。
#308/#309の2.5e-5という承認を今回へ移していない。比較器の`within_criteria`は過去の診断値で、
採用判定には実差と完全な順位一致を使った。未使用入力・他モデルの一致やamici品質の再測定は主張しない。

## ボトルネックと候補の判断

現行embeddingの128境界ではforward/evalのcall中央値203.64 ms、cache clear 6.19 ms、
readback 0.012 msだった。512境界では889.99 / 9.24 / 0.010 msである。
この入力ではforward/evalが大きいが、attentionとFFN等を分離したkernel profilingではない。
local graph構築はlazyであり、cache hitの微小な時間をGPU計算・転送時間とは扱わない。

共有案はembeddingの2048/4096/limit/mixed/multichunkでMetal peak中央値を
約204/256/256/204/244 MiB減らした。mixedのwarmは10,888.95→10,676.97 ms、
multichunkは14,816.61→14,477.79 ms。長文の限定候補として検討する根拠はある。
一方、512境界のpeakは全3processで2,485.17→2,805.17 MiBと320 MiB増えた。
rerankerのmixedはpeakが同じ3,953.52 MiB、warm中央値は6,867.01→6,911.48 msで、
sharedのprocess中央値範囲は6,859.05〜7,707.32 msと広い。全経路への共有採用は見送る。

GPU生成はdense N²の容量を減らさず、全ケースに共通する速度改善も得られなかった。
8192のmask単体は計算上256 MiB、全要素oracleではlocalとbatch合成を含む
Metal peakがCPU/GPUとも805,371,912 bytesだった。CPU→GPU転送だけを分離した観測ではない。
単体oracleの`local_readback_ns`は`as_slice`取得であり、全要素走査・コピー時間ではない。

64 MiB buffer案は各forward前にsetterの成功を確認し、cleanup時の超過と失敗ではclearする。
現行のpool回収タイミングを含むため、64 MiBはGPU全資源や瞬間peakのhard limitではない。
reranker mixedのwarmは7,240.70 msと現行より約5.4%長く、全体採用を見送る。
compile案は8 forwardごとのclearで保持期間を制限したが、現行srcにcompile transformは見つからず、
entry/bytesも観測できない。mixed warmはembedding 10,881.79 ms、reranker 6,888.16 msで、
意味のある全体改善を示す根拠にならない。無制限保持を推薦しない。

内部のlocal mask cacheは観測したケースで最大4 keyだった。公開ModernBertは任意seq_lenを
無期限cacheする現行契約を維持する。4 keyの内部保証を公開APIの容量上限へ拡張しない。
継続系列は各processの実elapsedとcleanup後active/cacheで確認する。31回から長時間を推測しない。
例えばembedding mixedの現行process末尾は349.51〜351.60秒、sharedは342.85〜343.80秒。
この有限範囲の復帰確認は無期限リーク・複数thread競合の証明ではない。

## 検証、停止記録と証拠の適用範囲

CPU/GPU全要素oracleは48設定（長さ1/8/9/17/128/512/2048/8192 × half 0/3/64 × 2）で成功した。
local全要素と合成maskを照合し、対称性・window端点・右padding・内部masked keyを確認した。
既存real_model_cleanup_memoryは全5候補で各93試行、合計465試行が成功し、
Forward/Pool/Eval/Readbackの注入失敗360回では両FFI clearを確認した。
既存assertionによるArray解放→cleanup一回、有限出力、元エラー保持を再利用した。
取得/clear/freeの失敗コードとArray寿命の標準テストを削除・重複追加していない。

初回verifyはpressure悪化でshared以降を止め、成功済みcurrentのmask/cleanupと
新しい保存先で成功した残り4候補を合わせた。失敗したrun全体を成功と読み替えていない。
測定ではpressure悪化でreranker limitのtrial 1、mixedのtrial 0 gpuを停止し、
未完了の保存先を残して新しいprocessで再検証した。自身のprocess groupはすべて回収した。
測定開始時の所有権監視ではsetuidのpsを実UIDと取り違えて止めた記録も保存した。
以後はreal UIDで帰属を照合し、signal対象のsession leaderのeffective UID確認は維持した。
他の所有者のprocessは終了していない。

30分・所有RSS 8 GiB・pressure=1・swap増分256 MiBの停止条件を緩めていない。
長いcaseはtrial単位、mixed/multichunkは候補単位で分割し、同じ入力・反復数を維持した。
結合時に全5候補×3組の条件・ファイルhash・正常終了・process回収を照合した。
現行source、候補source/binary、モデルhashは各測定前後で一致した。
同じチャットの他のbuild/GPU作業は重ねなかったが、他アプリのGPU無負荷は証明できない。
起動時はAC接続・thermal warningなしであり、全測定期間の熱条件固定は主張しない。

RSSはprocess範囲の約0.1秒間隔の標本最大、Metal peakはcall範囲のallocator観測で、
範囲が異なる。全候補に同じ観測を入れ、wallにはその負担を含む。
3process・開始順・ホスト負荷の限界から、小さな速度差の統計的保証は主張しない。
backend abort、panic、runnerへのSIGKILL後の回収は未検証である。

モデル不要の追加2テストはreadback欠落と失敗時の片方だけのclear、候補混合・source drift等を拒否する。
初回個別0.018秒、当時の標準Python全25件は5.053秒で成功した。
対応する検査を一時的に外した反証ではテストが失敗した。環境エラーを検出実績に含めていない。
分割collectorの追加2テストは15組の結合と欠落・重複・条件変更・改変・未回収の拒否を確認する。
重い実モデル反復や8192 oracleは標準CIへ追加しない。件数やこの時間から保守費削減を主張しない。

測定後は報告・結果JSON・再現文書と、外部で使ったcollectorの同一bytes・軽量テストを追加した。
ホスト復帰時点では測定したRustと候補patch、モデル、lockは変更していなかった。
下記の修正後は、測定版と現在版の差を明示して証拠の適用範囲を評価する。
変更版の既存verify-fixtureは3 workloadともNFR-001内で成功した。
モデル不要の標準Python全27件も7.483秒で成功した。
固定natural-japaneseのlintは成功し、環境条件の列挙を表へ整理した。
技術語の反復による語彙多様性の指摘は、条件名の一貫性を保つため残した。
原資料との数値・条件・採用状態を手動で照合し、標準checkと独立評価は公開前の別工程で確認する。
実測原本・停止記録はホストのcheckout/旧run外へ保持し、公開JSONには私的path・PIDを含めない。

既存local mask cacheコメントは、内部callerの4区分と公開forwardの任意長を区別する説明へ整理した。
数値、Array寿命、Metal固有の理由を説明するコメントは保持した。

## assessments: 初回ホスト復帰後の再評価（2026-10-13、過去の記録）

標準checkはClippyの`absolute_paths`違反3件で失敗した。追加oracleの呼出しを
`use`で取り込む既存規則への照合が不足していた。`spike310.rs`のsandbox確認、
cleanup関数、Componentをimportし、同じ参照先を呼ぶ形へ修正した。
oracleの入力、演算、assertion、Arrayの解放順、cleanup回数は変更していない。

元の差分基準は`4cd349961e5d656dac0e272ec8d9b6573a854da3`を維持する。
初回停止時からホスト復帰時までの既存ファイルの変更は、#310の報告・操作説明と
`model.rs`のcacheコメントだった。実測JSON、分割collectorとその軽量テストも追加された。
今回の修正はoracleのimportと本報告である。推論演算、候補patch、モデル、lockは変えていない。
古いprepared記録のsource hashを現在版のhashとして再利用しない。

停止記録のfindingsに残った事項を、新しい証拠へ次のように照合した。
旧runは実装または準備で停止しており、提供された記録に独立評価のacceptedはない。
証拠の`status: passed`はホスト実行の主張として扱い、受入を引き継いでいない。

| 過去の未確認事項・失敗 | 現在の評価と限界 |
| --- | --- |
| 最新main・親Issueのオンライン照合 | ホストの`main-observation.json`は開始HEADと一致し、`parent.json`で検証・保存条件を確認した。2026-10-13の観測であり、将来のmain一致は保証しない |
| lock固定APIと報告の版 | 上記5資料は現在も指定blobと一致する。現行sourceとlockを照合し、buffer poolと現在threadのcompile cacheを区別した。ADR-0009のcompile cacheを4 kernelとする説明は、entry/bytesの実測保証として使わない |
| Rustビルドと標準check | ホストで5候補のreleaseビルドと標準テスト実行は確認できるが、check全体はClippyで失敗した。今回の修正後の標準checkは未確認で、受入状態はneeds_changesのまま |
| 全mask・実モデルcleanup・数値比較 | ホスト原本と集計を照合した。48設定、465試行、差0と順位一致は上記条件の測定版の観測である。今回のimport変更は同じ呼出先と順序を保つが、現在版のGPU再実行とは扱わない |
| cold/warm・peak・shape・継続系列 | 公開JSONのcasesとsourcesは`aggregate.json`に一致し、17ケース・255process・3,285 callを確認した。有限の継続時間とRSS/Metalの測定範囲を維持し、無期限運転やattention単独の支配性へ拡張しない |
| 日本語lint | ホストでの固定版lint成功は過去の結果として保持する。今回の変更文書の結果はhandoffに記す。lintだけでは数値・条件・合意の意味を保証しない |
| 独立評価 | 今回の修正・追記とホスト原本をread-onlyで評価した。85ケース/候補群のcold/warm・Metal peak再集計と報告表に不一致はなく、新たな修正必須欠陥はなかった。標準check成功と最終受入は未確認 |

`host-evidence-final.json`の参照89ログと`host-log-manifest-final.json`の原本2,019件は
すべて記録されたSHA-256と一致した。これは改変の有無の照合であり、内容の受入ではない。
manifestのSHA-256は`6373b14beabe7c1fce07daef7fae8e3b12a8221bf868026177e3f94bcf23845f`。
未完了のverify・pressure停止と成功した再試行は別記録として残す。

## handoff: 初回修正時の検証と残る確認（過去の記録）

関連するモデル不要Python4件は0.545秒で成功し、formatと差分検査も成功した。
変更報告の日本語lintは固定版`9a78a42964096da509b8f3e011f0085a5f080151`で試したが、
offline cacheにsudachipyがなく依存解決に失敗した。今回のlintは未完了である。
同版の手動チェックリストで追加文の結論・条件・否定・数量・参照を原資料へ照合した。
Clippyの限定再確認は共有build先の`.cargo-build-lock`への書込みがsandboxで拒否され、
コンパイル開始前に終了した。修正後のClippy成功とは扱わない。
設定済み`bash scripts/check.sh`は同じClippy規則でoracleをビルドするため、
この失敗条件を確認する新しいテストは追加しない。import表記の専用テストは既存lintと重複する。

既存の軽量検証は候補の混合、readback欠落、失敗時の片方だけのclear、
分割証拠の欠落・改変・未回収を検出するため維持する。全mask oracleはwindow端点・対称性・
paddingの誤り、既存cleanup反復は失敗後の解放順と復帰を確認するため維持する。
大容量readbackと実モデル反復はホスト限定とする。今回テストの削除・統合はなく、
失う検出条件もない。保守費用や不安定さ、変更による実行時間の改善は測定していない。

今回の同一参照先へのimport変更から、性能比較を全件再測定する必要は認めなかった。
上記のホスト実測は測定版の調査証拠として使い、現在版の全体検証はホストが設定済みcheckで行う。
captureはnullのままで、IssueはUI媒体を要求しない。製品policyの採用、許容差や実行上限の変更、
commit・push・公開は行わない。backend abort、panic、SIGKILLと他所有者のGPU負荷に関する限界は残る。


## assessments: R1指摘に対する修正（2026-10-13、過去の記録）

R1-1〜R1-3を現行ファイルで再現・確認し、修正した。独立再評価と変更後の標準checkは未完了である。
前節の「新たな修正必須欠陥はなかった」は初回修正時の判断として残す。
その後の独立評価は、待機後の回収を実行中の資源監視と区別し、固定期待値の再計算と
buffer候補の未使用状態も指摘した。初回判断から受入を継承せず、今回の修正へ戻した。
根拠はホスト復帰runの`verification/review-1.json`、対象ID
`810e14b528b2f559ba11c0275d0233ba97074006e05bd0291a3c36e839be8f31`である。
履歴にこのR1評価が1件残っており、3指摘の独立した解消判定は次の評価担当へ渡す。

| 指摘 | 原因・修正と今回の証拠 | 残る限界 |
| --- | --- | --- |
| R1-1 | verifyが無期限の`owned_run`待機を使い、外側wrapperの監視を公開入口へ接続していなかった。現行verifyに既存samplerと所有group回収を接続した。全検証共通の30分期限、RSS 8 GiB、pressure=1、開始時からのswap増分256 MiBを維持し、基準値・部分ログ・monitor・終了値・回収を保存する。資源値と時計を模擬し、実子processと同一groupの子孫を使って、各制限の停止と正常終了、group消失を確認した。verify入口が同じ基準値・期限を渡し、停止後にcompleteを作らずincompleteを残すことも確認した | 今回の軽量検証ではmacOS sensor値を実測していない。GPU負荷下の現在verify再実行ではない。runnerのSIGKILL後回収は保証しない |
| R1-2 | 固定headerからのlengths・histogram・padded tokens・期待shapeをcallループ内で再生成していた。単一validate内で一度算出し、各callのmetricsへ照合する。3chunk・4callの読取りプローブは12回から3回となった。既存の形状不一致検証に後続warm callの異常も含め、欠落・有限性・順序・実readbackの検査を維持した | 速度や保守費の削減量は未測定。収集・結合・別processでの再検証は保持し、工程をまたぐcacheは導入しない |
| R1-3 | 共通patchがbuffer候補にもcompile専用のSPIKE_CALLSを生成していた。counterの定義をcompileだけに限定した。隔離source生成の既存検証でbufferのcounter不在とcompileのcounter、周期分岐・失敗判定を確認した | 生成bufferの未使用定義だけを削除した。失敗時の両clear、compile周期とreset、製品policyは維持した。現在候補binaryの実モデル再実行はしていない |

指定5資料はstartCommitの指定blobと今回も一致した。版・合意状態・適用条件は
[出典の表](#出典版適用条件)を維持する。#303の解放順、ADR-0010のmask fill、ADR-0014のMLX-onlyを
現行sourceに照合し、#308限定の比較許容差を今回へ移していない。
開始HEADとCargo.lock hashも上記固定値のままである。

停止記録、初回と追加修正の生応答、評価履歴、ホスト追加証拠を読み直した。
参照89ログと原本2,019件のSHA-256は再照合で一致した。
公開JSONのcases/sourcesは原本aggregateと一致し、85比較群・3,285 callのcold/warm・Metal peakを
保存summaryのcall系列から再集計して一致を確認した。これは保存された測定版の再評価である。
48設定のmaskと各候補72回の失敗clear観測は、以前の実測原本として保持する。
R1評価は実行成功を受入とせず、3件の必須欠陥を残している。

ホストの`check-2.stdout/stderr`は評価targetに記録されたhashと一致し、
import修正後・R1修正前の標準check成功を確認した。以前のClippy失敗はこの証拠で解消したが、
現在差分のcheck成功へ読み替えない。初回のmain・親Issue照合は2026-10-13の観測、
固定API調査、数値一致と有限反復は測定版の条件付き証拠として扱う。
日本語lintと独立受入を、実測成功やcheck成功から推定しない。

測定版からのRust変更は前回の同一呼出先へのimport整理だけである。
今回のR1修正はverifyの監視、比較の固定期待値、bufferコピーの未使用定義と関連テスト・文書であり、
mask演算、readback、cleanupの分岐・解放順、モデル、precision、lockは変えていない。
保存済み数値・時間を現在版の実測へ更新せず、全件の性能再測定は要求しない。
将来の再実行では新しいpreparedを作り、古いplanのsource/binary hashを現在版として使わない。

## handoff: R1修正後の検証と引き継ぎ（過去の記録）

関連するmask/cache・分割検証6件は0.815秒、固定表比較の5件は0.037秒で成功した。
これらは同じsandboxでのモデル不要検証であり、過去の全27件・7.483秒とは対象が異なる。
時間改善率や保守費・不安定さの削減は主張しない。
verifyの監視欠落が無制限の待機や資源悪化を許す不具合に対し、追加2件は
制限到達、部分記録、実group回収、公開入口の接続を守る。既存検証にこの保証はなかった。
約0.8秒の関連検証と小さい子processで確認し、重いモデルfixtureやGPUの再測定は追加しない。
固定表のshape拒否は既存テストへ統合し、候補分離検証はcounterの適用範囲も確認する形で維持した。
テストの削除はなく、失う検出条件もない。readback・有限性・順序・欠落・source drift・
失敗時の両clear・分割証拠の改変や未回収を検出する既存検証を保持した。
Array解放順の模擬寿命検証と実モデル復帰の責任分担も維持する。

[現行再現手順](README.md)をverifyの監視と保存ファイルへ一致させた。
文書の条件・数量・版・合意範囲は原資料と照合し、旧実測・停止記録・評価runは変更していない。
固定natural-japanese版の手動チェックリストも今回追記へ適用した。
固定版の機械lintはoffline実行を試みたが、sudachipyのcache不足で依存解決に失敗した。
指摘0件やlint成功とは扱わない。条件名・技術語の反復は意味の一貫性のため保持した。
基準値の保存を加えた後も監視・候補分離4件は0.333秒で成功し、差分検査も成功した。
5候補の生成cache sourceを測定preparedの原本へ照合し、bufferの未使用counter定義の削除以外は
すべて同じbytesであることを確認した。これはsourceの比較であり、現在binaryの実行ではない。

今回の修正を含む全体検証は、ホストが設定済み`bash scripts/check.sh`で実行する。
Python discoveryは今回の監視・比較・候補分離検証を含む。captureはnullで、IssueにUI媒体の要件はない。
残る受入は現在差分と変更文書に対する独立評価であり、保存実測のpassedから引き継がない。
今回のsensor模擬の限界、有限shape/F32/反復、他所有者のGPU負荷、panic・abort・runnerのSIGKILLの
未確認を残す。長文共有の採用候補と512境界のpeak増、mixed rerankerの利益不足の判断は維持する。
製品採用、新しい許容差・資源hard limit・consumer移行はユーザー判断が必要な別の作業である。
commit・push・公開、実行上限とcheck/capture契約の変更は行わない。

## assessments: R2指摘に対する修正（2026-10-13、過去の記録）

未解決のR2-1・R2-2を修正した。現在差分の独立受入と全体checkはホストへ引き継ぐ。
R2評価の対象IDは`98064faafdfd1e07c9376bb8215451f9e3ef84de0dcc9ac87e16d703487e502e`。
ホスト復帰runの`verification/review-1.json`・`review-2.json`、修正1・2の生応答、
初回の停止記録と追加証拠を読み、以前の解消判定を現在版の受入へ引き継いでいない。
元の差分基準は`4cd349961e5d656dac0e272ec8d9b6573a854da3`のままである。

| 指摘 | 現在の修正と再確認 | 限界 |
| --- | --- | --- |
| R2-1 | caller、準備完了hash検査、JSON検査がplanを別々に読んでいた。`verify_plan`が新しいbytesを一度取得し、hashとJSONを検査してplanとhashを返す。#308 measure、#310 verify・measure・partition.collectは返されたplanを使い、同じhashをcontextへ記録する。次の境界では再取得する。既存準備検証へ読取り差替えを統合し、一境界の読取りが1回で、次の境界の改変を拒否し、返したhashがそのplanに対応することを確認した | HEAD・source・候補binary・prepared完了の検査は維持する。工程をまたぐcacheは導入しない。速度効果は未測定 |
| R2-2 | hash照合後に内容を読み直したため、検査後に別の時間値を集計できた。必要ファイルを一度取得し、そのbytesのhashを確認してからJSONとstderrイベントを解析する。phase-completeの解析と記録hashも同じbytesを使う。stdout等のhash検査だけのファイルは内容を保持せず、全ファイルの生bytesを常駐させない。2回目のrecord読取りがcold wallへ100秒を加えた内容を返すfixtureで、再読取りなし、元summaryとの一致、phase-complete hashの対応を確認した | ファイルごとの観測であり、保存先全体の原子的snapshotや改変者の認証ではない。読取り後の変更を集計へ取り込まない保証で、原本が改変されていたという主張ではない |
| R1-1 | 現行verifyの監視接続と共通期限を再確認した。模擬sensor・時計と実process groupの検証、および公開入口の停止・incomplete保存の検証を再実行した。今回の入口検証は、contextに返却済みplan hashを記録することも確認する | macOS sensor値と現在GPU負荷下の再実行は未確認。panic・backend abort・runnerのSIGKILL後回収は保証しない |
| R1-2 | 固定headerの期待値を単一validationで算出する現行compareはR2対象版と同じhashである。初回と後続warmの形状異常、有限性・順序・欠落・readback検証を再実行した | 計算の局所共有を維持し、別process・収集・結合の再検証は残す。速度と保守費の削減量は未測定 |
| R1-3 | 現行patchはR2対象版と同じhashである。候補分離検証を再実行し、bufferでのcounter不在とcompileでの周期・失敗分岐を確認した | 新しい候補binaryの実モデル実行ではない。失敗時の両clearと元エラー保持、製品policyは維持する |

指定5資料は現在もstartCommitの指定blobと一致し、[出典の表](#出典版適用条件)を維持した。
#307の数値・検索品質は固定モデルの過去実測、#308の比較承認は同Issue限定、
ADR-0009/0010/0014はそれぞれ内部bucket・mask fill・MLX-onlyのaccepted規則として照合した。
#303の模擬寿命・実モデル復帰の責任分担も[現行方針](../../../CONTRIBUTING.md#推論失敗時のcleanupとメモリ観測)に従う。
開始HEAD、Cargo.lock hash、推論source、cleanup、候補patch、Rust oracleはR2対象版から変わっていない。
今回の差分は読取りと解析の共用、関連テスト、操作説明と評価記録に限定した。

追加証拠の参照89件とmanifestの原本2,019件のhashを再確認し、一致した。
R2対象のcheck-3 stdout/stderrも記録hashと一致したが、これは修正前の成功である。
保存済みembedding mixedの15 phaseを現行combineで新規一時保存先へ再集計し、
元のsummaryと全項目一致した。cold/warm・Metal peak・RSS系列・数値診断・mask/cleanup系列を含む。
この再集計は5.674秒だった。GPU・モデルは起動せず、原本と旧runは変更していない。
他caseの85比較群全体を今回再集計したとは扱わず、R2評価の全群照合を過去の証拠として保持する。
48設定のmask、5候補のcleanup、有限shape/F32/反復の実測は保存版の証拠であり、
現在のrunnerを実sensorとGPUで再実行した結果ではない。

## handoff: R2修正後の検証と残る限界（過去の記録）

関連する固定表・mask/cache・分割証拠の軽量検証12件は成功した。
同じsandboxで固定表5件は0.040秒、mask/cacheと分割7件は1.136秒だった。
plan改変・source drift・binary改変、全候補×3 trial、条件混在、欠落・重複・保存hashとの不一致、
終了とgroup回収確認を維持する。収集から結合までの再検証も削除していない。
今回の追加検証は、hash検査後の別内容が収集完了として受け入れられる現実的な不具合を防ぐ。
既存の事前改変拒否では検出しなかった読取り間の差替えを、モデル不要の小さいfixtureで確認する。
plan観測は既存準備検証へ統合し、入口検証も既存のものを拡張した。重いモデルfixtureは追加しない。
削除したテストと失う検出条件はない。既存の形状、実readback、候補分離、失敗時両clear、
資源監視・部分ログ・回収の検証は、それぞれ異なる失敗経路を守るため維持する。
実行時間・不安定さ・保守費の改善は未測定であり、過去の11件と今回の12件を速度比較に使わない。

[現行操作説明](README.md)は、一度検査したbytesの解析・集計という保証へ更新した。
初回collectorと同じbytesという記述は過去の状態へ限定した。過去の測定値・停止・評価は保持する。
変更文書の事実、数量、条件、権限、参照と未確認事項を今回のsourceと保存原本へ照合した。
固定natural-japanese版`9a78a42964096da509b8f3e011f0085a5f080151`のHEADと変更なしを確認し、
クイック手順の執筆・通読と手動チェックリストで推敲した。report.mdだけを指定したoffline機械lintは、
sudachipyのcache不足で依存解決に失敗した。lintとクイック手順全体は未完了であり、
指摘0件や成功とは扱わない。技術語・条件名の反復は保証範囲を正確に保つため残した。

ホストは設定済み`bash scripts/check.sh`と、変更文書を含む独立評価を更新する。
同checkのPython discoveryに今回の検証が含まれ、captureはnull、IssueにUI媒体の要求はない。
今回の読取り修正は推論演算・cleanup・測定条件を変えず、保存版の実測を現在版へ改称しない。
全性能測定の再実行は追加の完了条件にしない。実sensor、他所有者のGPU無負荷、
panic・abort・runnerのSIGKILL後回収、無期限のリーク保証は未確認として残す。
長文共有の候補、512境界のpeak増とmixed rerankerの利益不足による全体採用見送りを維持する。
新しい品質許容差・製品hard limit・consumer移行は別の判断であり、今回採用していない。
commit・push・公開、旧run、check/capture契約と実行上限は変更しない。


## assessments: R3指摘に対する修正（2026-10-13、過去の記録）

R3-1・R3-2の重複を修正した。現在版の独立受入と全体checkはホストで更新する。
評価対象`e821b98114718ab4cd2e1b2661ae65c2c5fb83e4a6c60e390197264f600094bb`の
`verification/review-1.json`から`review-3.json`、`repair-3.stdout`、停止記録、元成果物と追加証拠を照合した。
差分基準は`4cd349961e5d656dac0e272ec8d9b6573a854da3`を維持する。
前回は読取りを統合したが、同じbytesの解析・hash計算と、閉じたstderrの取得境界の重複が残った。
今回は境界内で結果を共用する。実行前後・保存完了・結合での新しい観測は残す。

| 指摘 | 現在版での評価と証拠 | 限界 |
| --- | --- | --- |
| R3-1 | `verify_plan`でplanを一度読み、JSON解析とSHA-256計算を各一回行う。準備完了hashの検査と返却は同じhashを使い、source検査へ検査済みplanを渡す。prepareの保存planとの一致検査を入口へ移し、既存準備検証に解析・hash計数を統合した。plan改変・HEAD/source/binary変更の拒否を確認した | 次の境界では再観測する。速度改善量は未測定 |
| R3-2 | 回収後のstderrを一度読み、一回の行分割で数値prefixとmask310を抽出する。measureとcollectは返却イベントを使用する。模擬processで一回の取得、両prefixの内容、#308既定のlist返却、model identityと回収状態を確認し、両入口からstderrを再読取りしないことも確認した | process/sensorはこの追加検証では模擬する。現在GPUと実sensorでの再実行ではない |
| R1-1 | verifyの共通30分期限、RSS 8 GiB、pressure=1、swap増分256 MiBと停止後保存・回収は変更していない。模擬sensor・時計／実子process groupと入口接続の既存検証を再実行した | runnerのSIGKILL後回収と実GPU負荷でのsensor再実行は未確認 |
| R1-2 | compareの固定header期待値はR3対象版と同じ。初回・後続warm形状の拒否、順序・有限性・欠落・readback検証を再実行した | 別processと収集・結合の検査を維持する。時間削減量は未測定 |
| R1-3 | patchはR3対象版と同じ。既存候補分離検証でbufferのcounter不在、compile周期、失敗分岐を再確認した | 新binaryの実モデル検証ではない |
| R2-1 | 各境界のplan読取り一回と返却plan/hashの対応、次境界の改変拒否を既存検証で再確認した。R3-1はこの保証を保って計算を統合した | 工程をまたぐcacheは導入しない |
| R2-2 | combineはR3対象版と同じ。hash検査後の再読取り差替えを防ぐ検証、元fixture summaryとの一致、全5候補×3 trial、欠落・重複・条件混在・改変・未回収の拒否を再実行した | ファイルごとのbytesと集計の対応を守る。原子的snapshot・認証ではない |

指定5資料のblobは現在も[出典の表](#出典版適用条件)と一致する。
#307は固定モデルの過去の数値・検索品質、#308の許容差承認は同Issue限定であり、今回の採用許可にしない。
ADR-0009/0010/0014のaccepted範囲と、#303のArray寿命・Result失敗時cleanupの責任分担を維持する。
R3対象manifestとの比較では、今回編集したrunnerと関連テスト以外の差はなかった（文書追記前）。
推論source、候補patch、Rust oracle、Cargo.lockは変えていない。
追加証拠89参照のhashとR3の修正生応答・check-4ログのhashを今回照合し、一致した。
2,019原本と85比較群の全分布照合はreview-3の過去の評価であり、今回は全面再計測・再集計していない。
保存版の同一入力/shape/F32、17ケース・255process・3,285 call、48 mask設定と各候補72注入失敗は
従来の調査根拠として保持する。現在runnerのGPU再実行や新しい受入へ読み替えない。

## handoff: R3修正後の検証と残る限界（過去の記録）

最終差分で関連Python検証14件が成功した。固定表5件は0.043秒、mask/cache・分割9件は1.479秒だった。
同じsandbox・Python discoveryで実行した局所検証の時間であり、性能改善の比較には使わない。
planの重複計算検査は既存準備検証へ統合し、stderr検証は小さい模擬processのログと既存fixtureを使う。
数値記録とmask観測の対応を失う不具合、両入口が再読取りへ戻る不具合を防ぐ保証を追加した。
既存の改変拒否・readback・形状・候補分離・失敗時両clear・部分ログ保存・group回収は維持する。
削除したテストと失う検出条件はない。大容量maskや実モデルfixtureは追加しない。
費用はモデル不要の局所検証に限定し、実行時間・不安定さ・保守費の改善量は未測定とする。
行数や件数の変化を改善の根拠にしていない。

[操作説明](README.md)を現在の解析・hash・stderr共用へ合わせ、過去の結果は履歴として残した。
変更文書の条件、数量、権限、参照、未確認事項をsourceと原資料へ照合した。
固定natural-japanese版`9a78a42964096da509b8f3e011f0085a5f080151`のHEADと変更なしを確認し、
クイック手順の執筆・通読と手動チェックリストで推敲した。report.mdを指定したoffline機械lintは
sudachipyのcache不足で依存解決に失敗した。機械lintとクイック手順全体は未完了であり、
成功や指摘0件とは扱わない。技術語と条件の反復は保証範囲を保つため残した。
差分の空白検査も成功した。全体checkはsandboxでは実行していない。

ホストは設定済み`bash scripts/check.sh`と、変更文書を含む独立評価を更新する。
Python discoveryに今回の検証が含まれる。captureはnullで、Issueの媒体要求はない。
今回の局所修正は測定演算・cleanup・実測条件を変更しないため、全GPU測定の再実行は要求しない。
現在GPU・実sensor、自然発生GPU障害、panic・abort・runnerのSIGKILL後回収、無期限リークは未確認である。
長文共有の後続候補、512境界のpeak増とmixed rerankerの利益不足による全体採用見送りを維持する。
新しい数値許容差・製品hard limit・consumer移行は別の人の判断であり、今回採用していない。
旧run、check/capture契約、実行上限を変更せず、commit・push・公開も行わない。

## assessments: R4指摘に対する修正（2026-10-13、過去の記録）

R4-1のverify正常経路のNameErrorと、R4-2のbinごとのbucket再探索を修正した。
前回の受入は継承せず、現在のコードと関連検証で全9指摘を再確認した。以下は修正担当の評価であり、
変更後の独立受入はホストで更新する。要求・合意は[Issue #310](https://github.com/thkt/rurico/issues/310)のままである。

比較基準は開始HEAD `4cd349961e5d656dac0e272ec8d9b6573a854da3`＋未commit差分。
停止記録、元成果物、修正生応答`repair-4.stdout`、評価履歴`review-1.json`〜`review-4.json`、
追加証拠`host-evidence-final.json`を読んだ。評価・修正原本はホスト内の
`rurico-310-host-return-20261013/verification/`、追加証拠は`rurico-310-host-20261013/`にあり、
今回受領した停止記録が絶対pathとhashを特定する。
R4対象IDは`f4159b1d151f74108927427d283e473c9504b66de2cd728a4f0dc623f36dde16`。
その全ファイルmanifestとの差は、文書更新前にはcompare/runと既存テスト2ファイルだけだった。
指定5資料のblobは[出典・版・適用条件](#出典版適用条件)と現在も一致し、根拠の変更はない。
#308限定の比較許容差、ADR-0009の内部bucket、ADR-0010のglobal fill、ADR-0014のMLX-onlyを
今回の新しい品質許容差・容量保証・方式採用へ拡張していない。

| 指摘 | 現在の成果物と新しい確認 | 適用限界 |
| --- | --- | --- |
| R4-1 | R3のstderr共用時に削除したヘルパーのcallerがverifyに残った。閉じたstderrを`parse_observations`へ直接渡す。既存入口テストに正常経路を統合し、修正前のNameErrorを再現した。同じ条件で修正後は5候補cleanup、48mask記録、model identity、最終plan/model照合、complete保存を確認した。両clear不足・identity違い・mask欠落・最終plan/model変更はincomplete保存へ進む | 子process・sensor・plan/model取得は模擬し、ログ解析と記録保存は実行する。実GPU verifyの再実行ではない |
| R4-2 | lengthsごとに最小bucket indexを一度求め、histogramへ加算する。3chunkの計数プローブは各表で3探索になり、4binでは従来12探索から減った。同じhistogram、padded tokens、forward形状を既存の3表fixtureで確認し、上限外chunkの拒否を既存検証へ統合した | 実行時間の改善量は未測定。工程間cacheや新helperは追加しない |
| R1-1 | 共有30分期限、所有RSS 8 GiB、pressure=1、swap増分256 MiBと停止後の部分保存・回収をsourceへ再照合し、模擬sensor・時計／実子process groupと入口の停止検証を再実行した | 実GPU負荷下のsensor、runnerのSIGKILL後回収は未確認 |
| R1-2 | headerの固定期待値はcallループ外のまま。初回と後続warmの形状異常、identity・欠落・有限性・readbackを既存検証で再確認した。R4-2はbin間の再探索だけを除去した | callごとの検査、別process・収集・結合の再検証を維持 |
| R1-3 | patchのbuffer counter不在、compileだけの8forward周期、失敗時分岐と候補混合拒否を既存候補分離検証で再確認した | 新binaryの実モデル再実行ではない |
| R2-1 | planの一境界一読取りと返却plan/hashの対応、次境界の改変、HEAD/source/binary drift拒否を既存準備検証で再確認した | 準備・実行前後の再観測は維持 |
| R2-2 | combineが取得bytesのhashを照合して同じ内容を解析する経路を読み、読取り差替え・元fixture summary一致、全5候補×3trial、欠落・重複・条件混在・改変・未回収の検証を再実行した | 今回は小さい合成fixture。保存15phaseの再集計はR4独立評価時の過去証拠。原子的snapshotや認証ではない |
| R3-1 | planのJSON解析とhash計算各一回、検査と返却への共用、prepareの保存plan整合をsourceと既存計数検証で再確認した | 速度改善量は未測定 |
| R3-2 | observeの一読取りで両prefixを返すこと、#308の既定list返却とmodel identity、measure/collectの再読取り防止を既存検証で再確認した | verifyは数値prefixを収集しない別入口。今回の直接解析接続は共用済みmeasure/collectを変更しない |

追加証拠のhashは指定値`be2bedab…`と一致し、89参照のhash、repair-4応答とcheck-5ログのhashも一致した。
これは保存整合の確認であり、`status: passed`や旧check成功を現在版の受入へ読み替えない。
2,019原本の再hashと85群の独立再集計は今回繰り返していない。R4独立評価の実施範囲を過去証拠として残す。
今回の修正は推論Rust・候補patch・Cargo.lock・入力・F32を変更しない。
保存された17ケース／255process／3,285callは固定測定版の演算比較に適用できるが、
変更後runnerの実実行証拠としては不十分であり、その用途にはneeds_changesと評価する。
長文共有を後続候補に留める判断、512境界のpeak増とmixed rerankerの利益不足、attention単独の支配性未確認は維持する。

## handoff: R4修正後の検証と残る限界（過去の記録）

関連Python14テストは成功した。mask/cache・分割9件は1.598秒、固定表5件は0.040秒。
同じsandboxのdiscoveryによる局所検証で、変更前後の性能比較ではない。
既存verify入口テストへ正常終了と照合失敗を統合した理由は、監視停止だけでは
今回のNameErrorや検査を飛ばしたcomplete保存を見逃すためである。既存の監視・回収検証を維持する。
histogramの内部整理には既存の複数表・初回/後続warm検証を使い、上限外拒否だけを同じ検証へ補った。
テスト削除と失う検出条件はない。追加の実モデルfixtureや大容量GPU実行は不要と判断した。
模擬ログと短い実子processに限定した費用は妥当であり、実行時間・不安定さ・保守費の改善量は未測定。
行の圧縮やファイル移動を改善に数えていない。

[操作説明](README.md)をverifyの解析・最終照合・保存へ合わせた。
文書の数量・条件・権限・未確認事項・参照を現在sourceと上記原資料へ照合し、過去の結果は保持した。
固定natural-japanese版`9a78a42964096da509b8f3e011f0085a5f080151`のHEADと変更なしを確認し、
執筆・通読・手動確認を適用した。report.mdを指定した機械lintはcacheへの書込み権限で失敗した。
書込み可能な一時cacheでoffline再試行したが、sudachipyのcache不足で依存解決に失敗した。
機械lintとクイック手順全体は未完了であり、成功や指摘0件とは扱わない。
条件・限界を表の各行へ残し、文書の見出しと段落先頭も通読した。技術語の反復は保証範囲の特定に必要なため維持した。
差分の空白検査は成功した。全体checkはsandboxで実行していない。

ホストは設定済み`bash scripts/check.sh`と変更文書を含む独立評価を更新する。
同checkのPython discoveryは今回の入口検証を実行する。captureはnull、Issueの媒体要求もない。
製品演算と実測条件を変更しない局所修正のため、全性能測定や実GPU fixtureの再実行は完了条件に追加しない。
現在GPU・実sensorの再実行、自然発生GPU障害、panic・backend abort、runnerのSIGKILL後回収、
他所有者のGPU無負荷、無期限リークは未確認である。新しい許容差・製品hard limit・consumer移行は採用しない。
旧run、実行上限、check/capture契約を変更せず、commit・push・公開は行わない。

## assessments: R5指摘に対する修正（2026-10-13）

R5-1の計測側の再割込みによる所有process残存を修正した。
`observe`の回収とmonitor/status保存をSIGINT/SIGTERMから保護し、外側のfinallyでhandlerを復元する。
回収失敗は伝播し、completeはFalseのまま、group_absentは回収成功後だけ記録する。
共有イベント取得、部分ログ、既存の監視条件、製品演算とcleanup policyは維持した。
以下は修正担当の確認であり、変更後の独立受入はホストで更新する。

要求・合意は[Issue #310](https://github.com/thkt/rurico/issues/310)、差分基準は開始HEAD
`4cd349961e5d656dac0e272ec8d9b6573a854da3`＋未commit差分のままである。
停止記録と元成果物、修正生応答repair-1〜5.stdout、review-1〜5.json、追加証拠を読んだ。
原本の所在は[前回の記録](#assessments-r4指摘に対する修正2026-10-13過去の記録)と受領した停止記録で特定できる。
R5対象IDは`609f2bc623e5887356caf301607f6d705611981c0b6988ac0e542978771b5fdd`。
修正前のファイルはR5のmanifestと一致した。今回の差は共有runnerのfinally、既存process検証と本資料・操作説明だけである。
指定5資料は[出典・版・適用条件](#出典版適用条件)のblobと現在も一致し、根拠の変更はない。
#303の解放後cleanup保証、#308限定の比較合意とADRの適用条件を新しい許容差・容量保証・採用許可へ拡張しない。

| 指摘 | 現在の成果物に対する確認 | 限界 |
| --- | --- | --- |
| R5-1 | 同じ実子process・模擬sensor・実ファイル保存で、TERMを無視するchildに2回のSIGINTを与えた。修正前はalive=True、complete=False、group_absentなし、修正後はalive=False、complete=False、group_absent=Trueだった。プローブ終了時は両版の所有groupを回収した。既存process検証へSIGINT/SIGTERMでの停止、回収とmonitor/status保存中の再送、handler復元、回収失敗の伝播を統合し、verifyとobserveの両経路で成功した | 実GPU・実sensorは使わない。回収失敗は実回収後の結果へ注入した条件であり、OSの自然発生故障の証拠ではない |
| R1-1 | verifyの共有30分期限、所有RSS 8 GiB、pressure=1、swap増分256 MiBをsourceへ再照合した。既存の実child・同groupの子孫を使う検証で正常終了と各制限の停止、部分ログ・group不在・終了値を再確認した。再割込み条件も同じ検証で通過した | sensorと期限は模擬。SIGKILL後回収は未確認 |
| R1-2 | 固定header由来の期待値はcallループ外で共用したまま。既存の初回／後続warmの形状・欠落・identity・有限性・readback検証を再実行した | 各callと別工程の検査は維持。速度効果は未測定 |
| R1-3 | bufferにSPIKE_CALLSを生成せず、compileだけに8forward周期を残すsourceと候補分離検証を再確認した。失敗時両clearの照合も通過した | 新binaryのGPU再実行ではない |
| R2-1 | verify_planの一境界一読取り、返却plan/hashの対応、次境界でのplan改変とHEAD/source/binary drift拒否を既存検証で再確認した | 準備・実行前後の再観測を維持 |
| R2-2 | combineのhash確認済みbytesからの解析と集計をsourceへ再照合した。既存の読取り差替え・fixture summary一致、全候補×3trial、欠落・重複・条件混在・改変・未回収の検証が成功した | 小さい合成fixture。原子的snapshotや認証を保証しない |
| R3-1 | planのJSON解析・hash計算各一回と検査／返却への共用を既存計数検証で再確認した。prepareの保存整合検査も残る | 実行時間の改善量は未測定 |
| R3-2 | observeの閉じたstderrから数値記録とmaskイベントを一度取得し、#308の既定list返却、measure/collectの解析結果共有と再読取り防止を既存検証で再確認した | 今回は小さいログ。保存完了hashと結合時の再検証を維持 |
| R4-1 | verifyの直接parser接続をsourceと既存入口検証へ照合した。5候補cleanup、48mask記録、model identity、最終plan/model照合とcomplete保存が成功し、5種類の照合失敗はincompleteへ進んだ | process・sensor・plan/model取得は模擬。実GPU verifyではない |
| R4-2 | 各lengthの最小bucket indexを一度求めてhistogramへ加算するsourceを再確認し、3表、上限外入力、padded tokens、初回／後続warm形状の既存検証が成功した | 探索回数の過去プローブは過去証拠として保持。時間効果は未測定 |

ホスト復帰の14参照記録、追加証拠本体と89参照、repair-5応答とcheck-6ログのhashは指定値と一致した。
過去のcheck成功や追加証拠のpassedは現在版の受入へ読み替えない。
R5独立評価による85群・255process・3,285callの再集計、48mask設定と各候補72注入失敗の両clear照合は過去証拠である。
今回は2,019原本の全再hashと全性能分布の再集計を繰り返していない。
製品演算、候補patch、Rust oracle、入力・shape・F32・Cargo.lockはR5版から変わらず、
保存実測は固定測定版の候補比較に適用できる。変更後runnerのGPU実行証拠としては不十分で、
その用途にはneeds_changesと評価する。今回の局所修正は軽量process検証で確認し、全GPU再測定を追加の完了条件にしない。
長文共有の限定検討、512境界のpeak増加、mixed rerankerの利益不足、attention単独の支配性未確認という判断を維持する。

## handoff: R5修正後の検証と残る限界

関連Python14テストは同じsandboxで成功した（2.776秒）。対象はmask/cache、分割証拠、固定bucket表である。
実行は`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python3 -m unittest -v test_mask_cache_spike test_mask_cache_partitions test_bucket_tables`。
局所ログは`/tmp/rurico-310-r5-local-tests.log`、変更前後の実子processプローブは
`/tmp/rurico-310-r5-interrupt-probe.json`へ保存した。一時領域の記録で、恒久公開は保証しない。

既存のprocess検証へobserveと再割込み・回収失敗条件を統合した理由は、単発の停止検証だけでは
TERMを無視する所有processの残存や、部分証拠保存の中断、回収失敗の成功扱いを見逃すためである。
同じ短い実childとgroup、実signal・実保存を使い、正常終了と既存の全資源制限条件も維持した。
独立した大容量モデルfixtureや重複した新テストは追加せず、失う検出条件はない。
追加の短いchildと模擬sensorの費用はこの停止保証に見合う。時間・不安定さ・保守費の改善量は未測定で、
上の2.776秒は変更前後の性能比較ではない。行数や移動を改善として扱わない。

[操作説明](README.md)の停止保証と現行評価への参照を更新し、過去の結果は残した。
数量・条件・権限・未確認事項をsource、実子processの結果、上記原資料へ照合した。
固定natural-japanese版`9a78a42964096da509b8f3e011f0085a5f080151`のHEADと変更なしを確認し、
クイック手順の執筆・通読・手動確認を適用した。report.mdを指定したofflineの機械lintは、
書込み可能な一時cacheにsudachipyがなく、依存解決で終了1となった。
ログは`/tmp/rurico-310-r5-japanese-lint.log`へ保存した。機械lintとクイック手順全体は未完了で、
成功や指摘0件とは扱わない。保証範囲を特定する技術語と条件の反復は維持した。
差分の空白検査は成功した。全体checkはsandboxで実行していない。

ホストは変更後の設定済み`bash scripts/check.sh`と、文書を含む独立評価を更新する。
同checkのPython discoveryは今回の実子process・再割込み検証を実行する。captureはnullで、Issueの媒体要求もない。
追加の権限や上限変更は不要である。現在GPU・実sensorの再実行、自然発生GPU障害、panic・backend abort、
runnerのSIGKILL後回収、他所有者のGPU無負荷、無期限リークは未確認のままである。
新しい許容差・製品hard limit・consumer移行は採用せず、旧runの記録・check/capture契約を変更しない。
commit・push・公開は行わない。後続の方式採用に必要な人の判断は[従来の引き継ぎ](#handoff-r4修正後の検証と残る限界過去の記録)を維持する。
