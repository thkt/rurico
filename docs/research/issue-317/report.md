# Issue #317: query batchと共通query再利用の試作

最新の比較結果は[results-revised.json](results-revised.json)に対応する。
モデル不要の検証接続、公開照合の重複推論、query結果のコピーを修正し、
修正版の全7ケースの実モデル比較を完了した。追加評価で未使用の内側wall計測を除去し、
変更後のcoverageは967/1006行、96.12%で既存95%条件を満たした。
最新のホスト確認は[内側wall除去後の確認](#内側wall除去後のホスト確認2026-10-13)に示す。
標準check・独立評価・新headの全CI・公開後確認はこれから行う。
収集完了を製品方式の採用や独立評価の受入へ読み替えない。
旧結果・失敗・追加検証前のhandoffは以下の履歴に残す。製品方式の採用はこの調査に含めない。

推奨は、query batchを複数query用途の次の設計候補として残し、公開APIへの採用は保留することである。
修正版の32件queryのwarm中央値は1184.132 msから1010.827 msへ14.64%短縮したが、
Metal peakは1444.2 MiBから2804.2 MiBへ増加した。単発・境界入力には明確な利益がない。
共通query再利用の今回の方式は見送る。修正版の256pairでも準備時間が2.098 msから23.238 msへ増え、
wallの0.11%差はwarmの範囲が重なり、tokenize削減による利益の証拠にならなかった。
固定F32 310m、修正版の全7ケース・14 processの比較は完了した。製品API・tokenizerは変更していない。

## 根拠と適用条件

要求の正本は [Issue #317](https://github.com/thkt/rurico/issues/317)
（提供本文のupdatedAt `2026-10-07T17:05:46Z`、2026-10-08の合意記録）。
開始commitとローカル `origin/main` はともに `a5471c83fd4f48255a71ee77b223968c2cf1015b`。
remoteの最新状態はsandboxでは再取得していない。Cargo.lockは変更していない。

引き継ぎ3資料は開始commitの指定blobと現行ファイルが一致した。
[既存#307報告](../issue-307/report.md)のblobは `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`、
[ADR-0011](../../decisions/0011-adopt-uniform-vector-length-contract-for-the-embed-trait.md)は
`e313d9b8c2c205484b937939c5d9a54edf778ca1`、
[ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md)は
`5c2840daf74b8364a88b37f864ed5717d6bbf403`。確認済みIDは置き換えていない。
#307の観測はrurico `d0639cc815d03f44dc7c80abebe40e9157deaca1` と固定310mモデル、FP32、
公式参照環境およびamiciの限定fixtureに対する過去の実測である。
今回の変更版の数値・速度・検索品質の確認結果として流用しない。

Issueの2026-10-08根拠版 `c8f250d60a5afb9944b9008d22ad6f4dda2d7103` からは、
#311の準備コピー削減、#363の識別可能な委譲/組立検証、有限logitの拒否、
関連調査と依存lock更新が入っている。単発query traitとquery prefix込みtokenize、
pairごとのencodeという比較の前提は現行コードでも成立する。
現在のforward・準備資産を使用し、旧コードへ最適化を重ねた仮想差分は利益に数えない。

[親#296](https://github.com/thkt/rurico/issues/296)の保存済み取得本文
（updatedAt `2026-10-07T17:08:32Z`、取得日は未照合）には、
開始版/lock/model/環境固定、同条件でcold/warmと順序・forward・wall・RSS/Metalを分けること、
通常checkを実モデル結果へ読み替えないこと、公開可能な要約を残すことが記載されている。
これは既存取得版の根拠で、最新原文の確認ではない。
scoutはDNS失敗、別取得入口もcache missで、今回sandboxからの親原文再取得はできなかった。
ホスト実行前に原文と実行条件を再照合する。不足を調査で解消し、要求・権限変更へは進まない。

採用済みADR-0011の同一implementor内の次元契約と、ADR-0014のMLX-only条件を守る。
[ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md)の
固定bucketを既存の製品関数から使い、chunking/precision/bucket変更を混ぜない。
[ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)に従い検索評価はamiciが所有する。
今回consumerや検索構成は変更せず、#307のamici品質を今回の改善証拠にはしない。
[README](../../../README.md#要件)と[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)の
MLX環境・通常/ignored検証の分担を適用した。

## 試作と結果契約

`query.rs`のobject-safeな `QueryBatch: Embed` は試作限定の入口である。
既存implementorを包むdefault経路はqueryを入力順に単件処理し、最初の失敗で停止する。
返却は全件成功時だけで、途中結果や再試行はない。空sliceは空結果でproviderを呼ばない。
空文字列は1queryとしてprefix付きで処理する。defaultはmetricsなしで、一般implementorにoptionsを要求しない。
`src/embed/embedder.rs`のMLX候補は同じ入口をoverrideし、`entry.rs`の共有測定関数を通して
1回のlockでqueryを準備する。
query prefixは文字列全体に付加し、既存truncateでEOSを維持する。
既存の文書bucket実行/pooled readbackを、1query=1chunkの内部計画として再利用する。
文書prefixや文書chunkingは使用しない。unpadded tokenizerを適用条件とし、不適合maskは拒否する。
製品のEmbed trait・公開APIへこの入口を採用していない。

`pairs.rs`は直前のqueryのraw Encodingを1件保持し、同じqueryが続くと再利用する。
tokenizerのcloneからpost-processor・padding・truncationを除いてrawを作り、
元のtokenizerの `post_process(query, Some(doc), true)` でpairを処理する。
query/docsのtoken列を連結せず、元のpair処理の特殊token・truncate/pad設定を維持する。
製品のpair truncateとforwardを `rerank.rs`から使う。入力順でscoreを返し、
空sliceはencode/forwardなし、空文字は通常pair、tokenizer/forward失敗は元の分類を伝播する。
baselineは製品の `encode_pairs` を直接使う。候補も各pairのEncodingをIDs/maskへ変換後に解放し、
測定のために全Encodingを残さない。結果組立や順位付けは製品の公開入口とも照合する。
queryが交互に変わる条件には再利用利益がなく、clone/post-processの追加費用だけが残る可能性がある。

## 検証の価値と確認結果

既存のplanner、bucket組立、pooling、readback、cleanup、metrics default、LazyReranker委譲は維持する。
既存spyは単なる固定件数mockよりも値・引数・順序を区別でき、新しい候補でも同じ考え方を使った。
query defaultのspyはquery/document/text入口の取り違え、入力の並替え、途中失敗後の処理続行を検出する。
prefix/truncate検証は候補が文書prefix・chunkingへ流れる誤りを防ぐ。
pair検証はraw生成時にpost-processingが二重に働く誤り、SEP/type ID消失、
query切替時の古いEncoding再利用、truncate/pad/overflowの相違を検出する。
モデル不要の小さな入力を使い、GPU・ネットワークや長いtimeoutの費用は追加していない。
追加保証は候補入口との接続にあり、tokenizers内部の一般的な正しさを再証明するテストではない。

Pythonの記録検査は、成功したように見える欠損/余剰・非有限・trial順序違いと、
query行の置換/次元違いを拒否する。実モデル結果の代用品ではない。
既存検証の削除・統合は行わず、失う既存検出条件はない。
試作専用の検証が増える保守負担はあるが、採用前に入力/結果契約を誤認する現実的なリスクを守る。
件数やcoverageのための重複テストは追加していない。時間・保守費用の改善は主張しない。
一時コピーで特殊token追加を除く変異とquery切替時のcache更新を除く変異を入れ、
どちらもpair検証の値/境界assertionが失敗した。正常版は成功した。
この確認は実tokenizersと候補sourceを使うモデル不要harnessで、実モデル受入の代用ではない。

sandboxでは既存依存成果物を読み取り専用で使って変更版libtestをコンパイルし、
新規のRustモデル不要3件が成功した。pair検証0.01秒、prefix/truncate検証0.02秒、
spyは表示上0.00秒だった。これは局所実行の表示値で、標準check全体の比較ではない。
Pythonの既存discoveryは新規2件を含む34件が8.002秒で成功した。
workspaceと同じlint指定での既存依存を使う局所Clippyも成功した。
日本語確認は固定natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151` の
クイック手順と手動checklistで意味・主旨・段落を照合した。lintはPyPIのDNS失敗で未完了であり、
指摘0件やクイック手順完了とは扱わない。既存の文書を含む独立評価へこの限界も渡す。
Cargoの新規checkは、MLX CMakeのGit取得がDNS制限で失敗した。
既存依存での局所コンパイルを、今回lockでのCargo check成功や全体検証の代わりにはしない。

## 旧測定版の採否提案と検証の分担

query batchは32件のforwardを1回へ集約し、この固定入力ではwarm wallが短縮した。
ただしMetal peakが1360.0 MiB増える。単発queryのwarm範囲は重なり、
長短混在の境界入力はforwardが4回から3回になってもwallの改善を示さなかった。
default付きのobject-safeな入口を設計候補として残すが、公開APIの採用は別の合意と検証が必要である。
並列呼出しや別モデル・異なる件数・入力長、consumer検索品質へ今回の結果を一般化しない。

共通query再利用は今回の実装を見送る。query raw Encodingを再利用しても、
各呼出しのtokenizer cloneとpost-processの費用を含む準備時間が全ケースで増えた。
修正版の単発wallは41.39%、交互queryは2.38%増えた。process RSS最大の差は約6〜8 MiBだった。
共通queryと境界入力のwall範囲は重なり、繰り返しqueryだけの高速化を示せなかった。
今回の方式の結果であり、すべてのtokenization再利用方式を否定するものではない。
採用のためにclone費用を測定外へ逃がしたり、別方式やbucket・precision変更を混ぜたりしていない。

通常checkはモデル不要の契約検証を担当し、ignored laneは固定モデルの互換性・比較を担当する。
収集完了と採用判断を区別する。変更文書を含む独立評価・新headの全CI・PR公開はこの後に確認する。

## 履歴: 担当AIのホスト準備と停止（2026-10-13）

最新mainは開始commitと一致し、親Issue #296の最新本文・updatedAtは引き継ぎ版と一致した。
共通queryケースは全256pairに同じqueryを使うよう修正した。元の試作入力は異なるqueryを混ぜていたため、
同一queryの利益と公開rerank順序を比較する入力として不適切だった。変更前の入力で性能を計上していない。
変更ファイルのコメントも確認し、Embedderの固定GPUメモリ量と倍増という記述を除き、
インスタンス別のweight保持とinstance内のlockという理由を残した。

Apple M3・24 GiBのホストでrelease libtestをビルドし、98.30秒で終了0、所有groupは回収済みである。
環境情報のsandbox取得に失敗した後、context未作成の状態で測定を起動して終了1になった。
この試行はモデルを起動しておらず、失敗記録を保持した。ホストでcontextを準備し直し、
固定snapshotの6モデルファイルを#307 manifestのhashへ照合して、新規保存先で測定を開始した。

既存の30分・RSS 8 GiB・pressure=1・swap増分256 MiBの外側ガードを維持した。
query/single/baselineは終了0・7記録・回収確認まで完了した。
candidateの実行中にpressureが1から2へ悪化し、ガードが約3秒で停止した。
candidateには7記録があるが完了statusではなく、全体のcomplete・summaryもないため成功扱いしない。
選択された所有groupはすべて回収済みで、他所有者のprocessを終了していない。
停止後のruntime sourceは開始時と一致した。モデルの停止後hashは未確認で、再開前に再確認する。

部分的な単発記録で利益・数値互換性・API採用を提案しない。メモリ確保後、
同じガードと新規保存先で全7ケースを実行し、必要な要約を更新して標準check・独立評価へ戻す。
方式、許容差、資源上限を通過目的で変更せず、既存fixtureを再生成しない。
追加の固定版日本語lintはREADME・本報告をホストの既存環境で確認する。
新headの全CIとPR公開は未完了である。

## メモリ確保後の旧版再測定（2026-10-13）

ユーザーのメモリ確保と再測定許可後、pressure=1、同じbinary hashと6モデルファイルのhashを確認した。
前回の部分ログは使わず、新しい保存先で全7ケースを再実行した。
同じ30分・RSS 8 GiB・pressure=1・swap増分256 MiBのガードで424.77秒、終了0となった。
全14 processの終了0・記録7件・group回収を確認し、所有groupはすべて不在だった。
source・lock・binary・モデル・contextの前後hash、収集manifest、集計と生ログを照合した。
今回の測定後は本報告と公開要約だけを更新し、測定したruntime sourceは変えていない。

Apple M3（Mac15,3）・24 GiB、macOS 27.0.1、Rust/Cargo 1.99.0、Xcode 27.0、
release・F32の固定310mで比較した。AC電源、熱警告なし、担当AIの別ビルド・GPU作業は停止した。
他所有者のGPU負荷が完全に不在だったとは証明していない。停止もしていない。
各方式・ケースは1 process、ロード後cold 1回とwarm 3回であり、統計的に安定した差の証明ではない。
公開可能な数値・環境・版・hashとcold/warm最小・中央値・最大、実forward形状は
[results.json](results.json)に保存した。rawログと個人pathは外部のホスト証拠に保持する。

次表はwarm中央値で、差はcandidate / baseline − 1である。
tokenize・forward・Metal・RSSの2値はbaseline / candidate。時間はms、メモリはMiB。
RSSはprocess全体の観測最大で、ロードと計測後の製品入口の照合を含む。
Metalはロード済みweight/cacheを含む各呼出しの絶対peakで、RSSと同じ指標ではない。

| ケース | baseline wall | candidate wall | wall差 | tokenize | forward数 | Metal peak | process RSS最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query-single | 35.252 | 34.690 | -1.59% | 0.025 / 0.023 | 1 / 1 | 1444.2 / 1444.2 | 1476.3 / 1477.6 |
| query-multiple | 1161.698 | 981.669 | -15.50% | 0.877 / 0.163 | 32 / 1 | 1444.2 / 2804.2 | 1479.9 / 1478.7 |
| query-boundary | 4979.220 | 4990.952 | +0.24% | 1.689 / 2.095 | 4 / 3 | 3289.3 / 3289.3 | 1995.1 / 1995.0 |
| pair-single | 37.276 | 54.564 | +46.38% | 0.027 / 19.042 | 1 / 1 | 1446.5 / 1446.5 | 1477.9 / 1582.5 |
| pair-multiple | 1021.242 | 1016.763 | -0.44% | 0.221 / 23.088 | 1 / 1 | 2806.5 / 2806.5 | 1477.1 / 1583.9 |
| pair-common | 7851.528 | 7806.092 | -0.58% | 2.108 / 27.582 | 1 / 1 | 3522.6 / 3522.6 | 1476.6 / 1582.0 |
| pair-boundary | 18896.456 | 19295.914 | +2.11% | 1.856 / 22.932 | 1 / 1 | 4786.5 / 4786.5 | 1990.3 / 2094.2 |

queryの最大絶対差は32件で4.172325134277344e-7、最小cosineは0.9999999999975016だった。
単発・境界入力は完全一致で、既存#307の同一token診断を満たした。
pairは全ケースでEncodingのIDs・mask・type ID・特殊token・offset・sequence ID・overflowが一致し、
scoreは完全一致した。公開score入口とも一致し、同一queryの単発・256pairでは公開rerankの順位も一致した。
各process内のcold/warm一致、空slice、queryの出力次元・順序も確認した。
モデル不要検証はdefault委譲・最初の失敗・prefix/truncate・tokenizer設定差を別途守る。
許容差・製品方式・資源上限・coverage条件・既存fixtureは変更していない。

前回の環境取得失敗・pressure停止は履歴として残し、今回の成功と混同しない。
固定版natural-japaneseの既存環境でREADME・停止時報告のlintは終了0だった。
本更新も同じ固定版のlintと意味確認を行ってから独立評価へ渡す。

## 旧版の標準coverage不足

CIと同じfeatures・除外regexのlcov生成は37.96秒、終了0だった。
開始commitとの全差分をdiff-coverへ渡すと変更行coverageは45%で、既存95%条件を満たさなかった。
主な未実行箇所は実モデル用のhost・entry・forward接続で、pair境界検証は100%だった。
ホスト測定成功を通常CIのcoverage成功へ読み替えない。
この不足も独立評価へ渡し、検証の配置と接続を見直す。閾値・除外・fixtureは変更しない。


## 履歴のassessments: 必須指摘への修正と証拠の適用範囲

比較基準は開始commit `a5471c83fd4f48255a71ee77b223968c2cf1015b` のままである。
旧評価の対象IDは `32517878b80a178399c77107cb719859bc3a498dbec18de3d4a681c33901d992`。
停止記録、実装の生応答、成果物manifest、review-1のassessmentsと追加ホスト証拠を読んだ。
追加証拠の参照81件のhashを照合し、全14 processのcold/warm集計・RSS・数値と公開要約の一致を
独立に再確認した。これは旧版の観測の確認であり、証拠の `passed` を現行版の受入へ変換しない。
引き継ぎ3資料は上記の指定blobと現在版が引き続き一致する。採用済みADRやIssueの要求は変更していない。

| 指摘 | 原因と現行修正 | 現行版の判定と限界 |
| --- | --- | --- |
| R1-1 | 記録・比較・候補組立が実モデル用ignored入口に閉じていた。queryのbucket組立・所有結果返却、pairの準備・forward接続、lockの解放・metrics、hostの反復・記録・公開照合を共有関数として通常検証へ接続した。実モデルのロード・実forward・Metal観測は既存MLXモジュールの薄いadapterに分離した。 | 構成を修正し、モデル不要の局所検証は成功した。CI同条件coverageは後述のホスト再検証で通過した。除外regex・95%条件・CI・checkは変更していない。 |
| R1-2 | 公開score_batch後のrerankが同じpairを再推論していた。single/commonは公開rerankを1回だけ実行し、全index順と全scoreをoriginal[index]へ照合する。multiple/boundaryは直接score_batch照合を維持する。 | sourceと共有protocolの呼出し回数・内容の検証では修正を確認した。実モデルの公開照合とprocess RSSは後述の修正版記録で確認した。余分な呼出しだけに生じる一過性失敗の観測機会は減るが、専用cold1/warm3回は維持する。 |
| R1-3 | 所有ChunkedEmbeddingを消費する段階で唯一のchunkをcloneしていた。既存の1query=1chunk条件で内部の所有Vecを移動する。 | 複数bucketの識別値・入力順とforwardが返したallocationの同一性を確認した。shape・有限性検査を行う製品forwardは維持する。修正版全体は再測定したが、コピー除去だけの速度差は分離していない。旧candidate wallを修正版の効果としない。 |

今回の追加検証は、準備したqueryのprefix/EOSがforwardへ渡らない、bucket実行順で返す、
不要なcloneが復活する、lock/forward失敗後に処理を続ける、公開照合でscoreやtie順を取り違える、
cold/warmや実forward記録を誤って組み立てる不具合を防ぐ。
識別tokenを使う候補境界検証と、観測処理に値を供給するprotocol fixtureを区別する。
後者のshape・peakはfixtureであり、モデルの数値一致・速度・Metal/RSSの実測ではない。
既存のpair特殊token・padding/truncate/overflow・例外、readback/cleanupの検証は維持する。
既存テストの削除はなく、失う既存検出条件はない。
公開照合は二つの観測を一つへ統合し、反復安定性は専用の計測反復に残した。

既存依存を読み取り専用で使い、修正版libtestを局所コンパイルした。
モデル不要Rust検証9件は成功し、候補接続4件は0.03秒、host protocol3件は0.10秒、
default委譲とpair境界はそれぞれ表示上0.00秒だった。Python記録検査2件と同じlint指定の局所Clippyも成功した。
追加検証は短いCPU入力で既存の通常laneに載り、実モデル・download・timeout待ちを追加しない。
候補と同じ処理を使うことで、fixtureを別実装として維持する費用を抑える。
これは修正版の局所値で、旧版の実行時間とは同条件で比較していない。
テスト件数・行数・配置変更自体を改善とせず、全体時間・不安定さ・保守費用の削減量は未測定である。

局所のinstrumented libtestで同じ9件を実行し、既存CIの除外regexをそのまま使った。
開始commitとの全差分をlcovの行へ独立に照合すると、実行は969/1008行（96.13%）だった。
主な未実行はignored入口の環境取得、Legacyの非query委譲、誤入口を拒否するfixtureのpanicである。
この比率には今回のテスト行も含まれる。必要な保証を増やさないassertionや行圧縮は加えていない。
Cargoは読み取り専用の既存targetのbuild lockで拒否されたため、これは現在lockでのCargo再ビルドではない。
インストール済みdiff-coverもPython module欠損で実行できず、局所の独立集計をCI gate成功とはしない。
実モデルの接続、GPU失敗/cleanup、固定モデル互換性は追加ホスト検証へ残る。

固定natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151` のcleanな取得版で推敲した。
uvのoffline解決はSudachi依存のcache不足で失敗したため、既存のlint用Python環境で同じスクリプトを実行した。
lintは終了0で、情報・読解負荷の指摘は残る。技術用語の反復と検証条件の列挙は意味を保つため維持し、
過去の失敗条件を読みやすさだけの理由で省かなかった。原資料との意味・条件・数量・参照も照合した。
別担当の読取りによる独立評価で局所coverageの集計を再現し、修正コードと変更文書に新しい必須欠陥は確認されなかった。
これは追加ホスト検証前の構造評価で、上の未確認事項を解消した受入ではない。


## 履歴: 追加ホスト検証前のhandoff

この節は追加ホスト検証前に渡した要求の履歴であり、現在の待機条件ではない。
下の修正版ホスト再検証で実行結果を示す。
対象は開始commitと本修正を含む未commit差分で、実行前に差分・Cargo.lock・runtime sourceのhashを固定する。
設定済み `bash scripts/check.sh` とcaptureなしの契約は維持する。媒体はIssueでも不要である。
checkはignored実モデル比較とCI coverage jobを実行しないため、担当AIの追加ホスト検証が必要である。

Apple Siliconの権限あるホストで、CIと同じfeatures・既存除外のlcovを生成し、開始commitからの全差分を
既存diff-coverの95%条件へ照合する。未実行行の失敗条件と対象版・環境・所要時間も残す。
続いて[READMEの固定モデル手順](README.md#ホストで追加比較する)で全7ケースを新規保存先へ収集する。
モデル再ロード、内部score経路との比較、公開照合、cold1/warm3回、前後hashと既存の資源ガードを維持する。
R1-2によりprocess全体の観測範囲が変わり、R1-3によりquery候補の返却費用が変わるため、
旧結果を投影せず、新しいJSONL・RSS/Metal記録・終了/group回収・集計を確認する。
旧run、`results.json` と過去の失敗/観測は証拠として保持し、新しい結果は別の版付き要約へ保存する。
その実測に対応する報告と採否提案を更新してから、変更文書を含む既存の独立評価へ戻す。
一般的な性能安定性、他モデル、consumer検索品質、公開API採用は今回の完了条件に加えない。


## 修正版のホスト再検証（2026-10-13）

R1-1〜R1-3の修正版をCIと同じfeatures・既存除外regexでCargoから再ビルドし、lcovを生成した。
全通常テストが終了0で、26.53秒、所有groupは回収済みだった。
開始commitとの全差分を既存diff-coverへ渡し、1008行中969行、96.13%で95%条件を通過した。
未実行39行はLegacyの非query委譲、ignored入口の環境取得とdispatch、拒否するfixtureのpanic等である。
モデル不要のpure処理と実際の測定protocolは共有し、必要な境界・失敗・順序の検出を維持した。
coverage regex・閾値・CI・標準checkは変更していない。モデル推論を通常CIへ持ち込んでもいない。

release libtestを同じ条件で再ビルドして71.31秒で終了0となり、compiler artifactを照合して保全した。
同じ固定6モデルファイルを再確認し、pressure=1で全7ケースを新規保存先へ再測定した。
旧版の成功や部分記録を流用せず、同じ30分・RSS 8 GiB・pressure=1・swap増分256 MiBのガードで実行した。
413.45秒で終了0、全14 processの各7記録と正常終了、所有groupの不在を確認した。
source・lock・binary・モデル・contextの前後hashと収集manifest、全56計測呼出しの集計を照合した。
機種・ソフトウェア・F32 snapshot・入力・cold1/warm3回は旧版と同じで、AC電源・熱警告なしだった。
他所有者の背景負荷の完全な不在は未証明である。別所有者のprocessを終了していない。

公開要約は[results-revised.json](results-revised.json)へ別の版として保存し、旧results.jsonは保持した。
source差分とbinaryのhashは修正版に対応し、測定後の変更は報告・README・新しい要約だけである。
次表は修正版のwarm中央値。差の定義とMetal・RSS観測の範囲は上の旧版表と同じである。
旧版とのprocess RSS差を公開照合削減の改善量とは扱わない。呼出し以外の割当・sampling状態も異なる。

| ケース | baseline wall | candidate wall | wall差 | tokenize | forward数 | Metal peak | process RSS最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query-single | 37.993 | 36.326 | -4.39% | 0.039 / 0.028 | 1 / 1 | 1444.2 / 1444.2 | 1477.5 / 1477.2 |
| query-multiple | 1184.132 | 1010.827 | -14.64% | 0.888 / 0.171 | 32 / 1 | 1444.2 / 2804.2 | 1478.4 / 1479.3 |
| query-boundary | 5055.092 | 5058.193 | +0.06% | 1.717 / 1.774 | 4 / 3 | 3289.3 / 3289.3 | 1994.9 / 1994.7 |
| pair-single | 40.365 | 57.072 | +41.39% | 0.035 / 20.118 | 1 / 1 | 1446.5 / 1446.5 | 1577.9 / 1584.9 |
| pair-multiple | 993.568 | 1017.224 | +2.38% | 0.251 / 25.179 | 1 / 1 | 2806.5 / 2806.5 | 1576.8 / 1583.0 |
| pair-common | 7861.631 | 7852.670 | -0.11% | 2.098 / 23.238 | 1 / 1 | 3522.6 / 3522.6 | 1578.2 / 1585.0 |
| pair-boundary | 19282.610 | 19259.222 | -0.12% | 1.829 / 30.279 | 1 / 1 | 4786.5 / 4786.5 | 2091.8 / 2099.3 |

query32件の最大絶対差4.172325134277344e-7、最小cosine0.9999999999975016、単発・境界は完全一致だった。
pairのEncoding・scoreも全ケースで一致した。single/commonでは公開rerank1回の全indexとscoreを
元のscore列へ照合し、multiple/boundaryでは公開score_batchとの一致を確認した。
専用cold/warm反復・内部score照合・モデル再ロードは維持した。
不要なベクトルコピーを除いた候補でも次元・入力順と既存の数値診断を満たした。
公開API採用や製品品質の新しい許容差へこの結果を読み替えない。

推奨は変わらない。query batchは複数queryの設計候補として残すが、Metal peak増加を含め採用は別途検討する。
単発・境界のwarm範囲は重なるため、単発高速化は主張しない。
今回の共通query方式はclone費用を含む準備時間が全ケースで増え、共通queryのwall改善も明確でないため見送る。
1 process/方式・ケース、warm3回という限界は残る。別モデル・件数・入力長・検索品質への一般化はしない。
この再検証と文書更新を既存の独立評価へ戻し、標準check・新head全CI・PR公開は後続記録で確認する。


## 最終評価での局所整理

追加の評価で見つかった冒頭の旧待機表記を更新し、最新結果と履歴の適用範囲を分けた。
query観測のcold結果と自身の比較1回を削除した。3回のcold対warm比較と別方式oracle比較は維持し、
cold結果の有限性・非ゼロnorm・次元・順序もそれらの比較で検査する。
モデル実行4回・計測wall・実forward記録・公開照合は変更していない。
共通の誤scoreとtie逆順の否定検証は外側のケースループから出し、各1回だけ実行する。
4ケースの正常な委譲先・入力・一度の呼出しと、2種類の不正結果の拒否を維持する。
新しい抽象化や検証のためだけの行は追加せず、速度・メモリ削減量は主張しない。

results-revised.jsonはこの整理前の測定版に固定した証拠である。
整理後の差は計測後の自己比較と通常テストの重複除去、文書の案内であり、
計測内のtokenize・forward・返却・4反復を変えていない。
現在版のprotocolとcoverageを再検証し、測定版との対応と限界を次の独立評価へ渡す。
旧runの評価応答は過去runのR1 IDを現在runの更新IDとして返し、unknown IDで形式不正となった。
その応答は受入結果ではない。旧記録・予約・上限は変更せず、新しい通常runでは
targetが渡す現在のreviewHistoryだけを更新IDの正本とし、過去のR1対応は説明として照合する。

局所整理後もCIと同じ条件の全通常テスト・lcov生成が26.77秒で終了0だった。
全差分diff-coverは969/1008行、96.13%で既存95%条件を通過した。
共有protocolの正常委譲・4反復・記録内容と、誤score・tie逆順の拒否もこの実行に含まれる。
測定内の実装を変えていないため、14 processの性能値は整理前の測定版として保持する。
計測後CPU処理の削減やprocess RSS変化は再測定しておらず、改善量を主張しない。


## 履歴のassessments: protocol検証の再修正（2026-10-13）

今回の評価対象は `c6ca94584146b786246674f45171fd7607a4dce331f1db5c7d829934203f4a8f`。
この対象へのR1-1〜R1-3は、履歴の同名IDとは別の指摘である。
評価原本は `rurico-317-tidy-host-return-20261013/verification/review-1.json`、
追加証拠は `rurico-317-tidy-host-20261013/host-evidence.json`
（SHA-256 `b4f7901a58f9896bde64cff39d8da1309a7a0940fe3fa5935267bbe8cc3c42ac`）。
停止側27件と追加証拠106件の参照hashを再照合し、不一致はなかった。
評価原本・停止理由・修正の生応答を読み、過去の受入判定は引き継いでいない。
比較基準と引き継ぎ3資料のblobは冒頭のままで、開始版・現行ファイルとの一致も再確認した。
Issueと採用済みADRの条件は変更していない。

R1-1の原因は、不変なquery入力から同じ期待値を記録ごとに作り直していたことである。
[host.rs](host.rs)ではケースごとに期待値とJSON形を一度作り、両方式の全4記録へ共有した。
期待値の生成は24回から3回になるが、記録ごとの値・trial・cold/warm・shape・tokenize検査は残す。
観測側の4反復と別方式比較も維持した。共有はこのテストのケース内だけで、状態を持つcacheは導入していない。

R1-2は、記録を検査する期待値に検査対象の返却scoreを使っていたため、両方が同じ誤った列になる回帰を見逃していた。
外側の識別score fixtureをcallbackへ供給し、返却値と全4記録をそのfixtureへ直接照合する。
callbackの最初の空入力と、後続の全query/doc内容・順序も照合する。
一時コピーで観測処理が全scoreを42へ置換する変異を入れると、返却値の照合で失敗した。
正常版は成功した。実モデル検証を通常laneへ追加していない。

R1-3は、任意のpanicを拒否成功とし、callbackのpanicも受理していたことが原因である。
callbackは呼出しを記録して正常に戻るfixtureへ変更し、catchの外で未実行を要求する。
dispatchの拒否内容もdomain/mode別に照合し、別のpanicを成功扱いしない。
未知modeをbaselineへ、未知domainをqueryへ流す変異をそれぞれ入れると、callback呼出し数の照合で失敗した。
正常4組の委譲・context追記と、元の不正2条件は維持した。
変異用コピーの最初の準備はsymlinkを辿って失敗したため、symlinkを保持して作り直した。
準備失敗を回帰検出として数えていない。元checkoutと過去runへ変異は適用していない。

テストの追加・削除はなく、既存protocol検証を補強した。
同一期待値の再生成だけを統合し、異なる記録・反復・方式の検査は残したため、失う検出条件はない。
返却値と記録の同時置換、誤った実験識別でadapterを開始する不具合に対する保証が加わる。
これはCPU fixtureの小さな検査であり、モデルの数値・性能の代用品ではない。
通常テスト3件は現在sourceを既存依存で局所コンパイルして成功した。実行表示は0.11〜0.14秒だった。
同じworkspace lint指定の局所Clippyとfmt、差分の空白検査も成功した。
実装・テスト・文書は同じ開始版と測定版へ照合したが、整理前後の実行時間は同条件で測定していない。
速度・メモリ・不安定さ・保守費用の改善量は主張しない。行数やファイル配置を改善の根拠にしていない。

履歴の未解決指摘も現在の証拠から見直した。共有処理の通常検証接続は
[embed.rs](embed.rs)・[entry.rs](entry.rs)・[rerank.rs](rerank.rs)と実モデルadapterで維持される。
旧lcovと開始版からの変更行を独立に照合し、修正前の969/1008行（96.13%）を再現した。
ただし現在のhostテストの変更には古いlcovを適用できず、現在版のcoverageは未確認である。
公開照合は[host.rs](host.rs)のsingle/commonでrerankを一度だけ呼び、全index・scoreを照合する。
複数query条件のscore_batch照合も残る。query所有Vecの移動とallocation同一性の検証は
[embed.rs](embed.rs)に残り、今回変更していない。
形式不正応答の追加3指摘についても、冒頭の現在案内と履歴の区別、cold自己比較の除去、
共通の誤score/tie順検証のループ外配置を確認した。現在の待機条件は本節と次のhandoffで更新した。

全7ケース・14 processの生記録からcold/warmの値・forward形状・RSS最大・数値診断を再集計し、
[results-revised.json](results-revised.json)と一致した。終了0とgroup回収は保存済み実行記録から確認した。
contextのhashと現在sourceを比較すると、測定後の変更はhost.rsと報告・READMEだけである。
今回のコード変更はhost.rsの通常テスト内に限られ、計測処理・製品adapterは変えていない。
測定版からの計測後の自己比較の除去は前節の範囲に限定される。
実測は整理前の測定版に適用し、現在版全体を再測定したとは扱わない。
単発・複数query・共通query多数pairを分けた推奨案と、少数反復・RSS sampling・
背景負荷不在の未証明・他モデルとconsumer検索品質の未確認という限界は維持する。
追加証拠のpassedは収集に関する主張で、現在版の独立受入ではない。

報告は上記の固定版natural-japaneseのクイック手順で推敲し、原資料と意味・数量・条件・参照を照合した。
既定PythonはSudachi未導入でlintが失敗したため、既存のlint環境で同じ固定版スクリプトを実行して終了0を確認した。
情報として出た文頭反復は修正版と履歴の適用範囲を説明するため維持した。lint成功を内容の受入とは扱わない。

## 履歴: protocol検証修正後のhandoff

この節は追加coverage実行前の引き継ぎの履歴である。実行結果は次節に示す。

対象は開始commit `a5471c83fd4f48255a71ee77b223968c2cf1015b` と現在の未commit全差分。
修正後host.rsのSHA-256は `3599b9a2e1cf2afde6732835f05e3b8149ab37308921916305ba705f5bdafa0f`。
Cargo.lockはSHA-256 `d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7` のままである。
設定済みcheckは[check.sh](../../../scripts/check.sh)で、captureはnull。IssueでもUI媒体は不要である。
標準checkはホストが変更後に実行する。sandbox内で全体check・ブラウザー・サーバー・実モデル推論は起動していない。

追加で現在版のCI同条件coverageが必要である。
[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)と[CI](../../../.github/workflows/ci.yml)が定める
同じfeatures・除外regex・変更行95%条件を維持し、開始commitとの全差分を照合する。
check.shはCargo check・Python・nextest・doctest・clippy・fmtを実行するが、lcov生成とdiff-coverは実行しない。
capture nullにもこの検証はなく、旧coverageをテスト変更後の版へ流用できないため、担当AIの追加ホスト検証へ渡す。
局所コンパイルは既存依存を読み取り専用で使う確認であり、現在lockでのCargo再ビルドやCI gateの成功ではない。

実行前後の差分・source・lockのhash、環境・実行範囲・時間・終了結果・lcov・diff-cover結果と
未実行行の失敗条件をcheckoutと過去runの外へ保存する。
結果に応じて現在の要約を更新し、変更文書と全差分を既存の独立評価へ戻す。
今回のテスト内修正だけを理由に固定モデルの再測定は要求しない。
計測処理を追加で変更した場合はその適用性を再判断する。現在版の標準check・coverage・独立評価は未完了である。
commit・push・公開、要求・品質許容差・資源上限・除外・閾値の変更は行わない。


## protocol検証修正後のホスト確認（2026-10-13）

この節は今回のpair検証修正・局所整理前の版の記録であり、現在版のcoverage結果ではない。

現在版をCIと同じfeatures・既存除外regexでCargoから再ビルドして全通常テストを実行し、lcovを生成した。
28.02秒で終了0、pressure=1を維持し、所有groupはすべて回収した。
開始commitとの全差分を既存diff-coverへ渡すと、982/1021行、96.18%で95%条件を通過した。
未実行39行はLegacyの非query委譲、ignored入口、誤った入口を拒否するfixture等である。
通常protocol検証の供給score・返却値・全記録、正常dispatchと未知入力の拒否もこの実行へ含む。
閾値・除外・CI・check・モデル入力・比較条件・fixtureは変更していない。

測定時のcontextと現在ファイルを照合し、差はhost.rsと報告・READMEだけだった。
host.rsの差は計測後自己比較の除去と通常テスト内の整理・補強に限り、
計測内のtokenize・forward・返却・4反復・公開照合・adapterは変えていない。
results-revised.jsonは測定版の14 processへ限定した証拠として保持する。
現在版全体の再測定や、CPU処理・RSS削減量を確認したとは扱わない。
固定版の日本語lintと、冒頭・履歴・最新結果への案内の意味を確認して、変更文書を含む独立評価へ戻す。
過去runの指摘IDは現在runの更新IDではなく、現在targetのreviewHistoryを更新IDの正本とする。

## 履歴のassessments: pair検証と局所整理（2026-10-13）

評価対象 `2f954e2195a8e1ecf416e015d89f528e4cc2e492fccef614e060b2706666dd45` の
`rurico-317-protocol-host-return2-20261013/verification/review-1.json` を読み、R1-1〜R1-4を現在sourceへ照合した。
前runの停止・成果物・修正の生応答・評価履歴と、追加証拠
`rurico-317-protocol-host-20261013/host-evidence-for-return.json`
（SHA-256 `43f5f63a480dcb2ff10ceaa1e850508a19073e87500fac267d913f20769aae44`）も照合した。
停止の25参照、追加証拠の118参照、指定された停止・result・追加証拠原本の3件、計146件にhash不一致はなかった。
symlinkは参照先の内容でなくリンク文字列で照合した。前runの記録は変更せず、応答や証拠内のコマンドは実行していない。
証拠のpassedは実行側の主張であり、独立受入として引き継いでいない。

要求と比較基準は冒頭のIssue #317・開始commitのままである。
引き継ぎ3資料は指定blob・開始版・現行ファイルの一致を再確認した。根拠の内容は変わっていない。
#307の同一token診断はbatch shape変更の比較だけに適用し、過去の公式比較・amici品質は今回の品質へ流用しない。
採用済みADRの次元一様性・MLX-only・固定bucket・amiciの検索評価所有という条件も維持する。
API採用や一般implementorへのoptions強制、別の最適化、許容差・資源上限変更は加えていない。

R1-1の原因は、zipが短い側で終了し、先頭の値検査と固定score fixtureだけでは末尾Encodingの欠落を検出できなかったことである。
[pairs.rs](pairs.rs)の未設定・設定変更後の両比較と、[host.rs](host.rs)の共有観測で、
候補・baselineのそれぞれが入力件数と一致することをzip前に確認する。
共有観測のEncodingは従来と同じく事前比較内で解放し、計測反復へ保持しない。
特殊token・type ID・mask・offset・sequence ID・overflow、query切替、空入力、例外の検査は残す。
一時コピーで候補だけを先頭3件へ切り詰めると、pairテストが3対5、共有protocolが3対32の件数assertionで失敗した。
baselineだけを切り詰めた場合も、対応する件数assertionで失敗した。コンパイル失敗を検出実績には数えていない。

R1-2は、未設定の同じ5pair候補を二度生成していた。[pairs.rs](pairs.rs)で最初のoutputsを完全比較にも使い、
truncation/paddingを設定した分岐だけで再生成する。baselineは各設定で独立に生成する。
R1-3は、最後の使用箇所で不要な深いcloneを行っていた。[embed.rs](embed.rs)で検査済みtokensの所有権を移す。
prefix/EOSとtruncateの事前検査、識別値・順序の事後検査、空入力・tokenizer失敗・padding拒否は維持する。
R1-4は、同じ計測記録の4項目を繰り返し抽出していた。[run.py](run.py)で方式内の4記録を一度ずつ抽出し、
その結果からcoldとwarmのmin/median/maxを集計する。全14方式・ケースの抽出は196回から56回となる。
これは処理回数のsource上の比較であり、実行時間・RSSの改善量ではない。モデル実行4反復と比較診断は削除していない。
保存済み全7ケース・14 process・56呼出しを再集計した。
summaryと公開results-revisedのcold/warm・診断・forward形状・RSS最大に一致した。
warm記録のtokenize必須項目を欠落させる局所確認ではKeyErrorとなり、欠損を成功扱いしないことも確認した。

過去の未解決指摘も、同名IDを別runへ流用せず再評価した。
対象 `32517878b80a178399c77107cb719859bc3a498dbec18de3d4a681c33901d992` の
通常検証接続は共有entry・query/pair実行の通常テストで維持され、
公開rerankの重複推論はsingle/commonで1回だけの公開照合、所有query結果のコピーはmoveとallocation同一性検査で解消を確認した。
対象 `c6ca94584146b786246674f45171fd7607a4dce331f1db5c7d829934203f4a8f` の3指摘は、
ケース内の期待値共有、供給score fixtureへの返却値・全記録の直接照合、
未知domain/modeでのcallback未実行と拒否内容の検査が現在も残り、今回の局所実行でも成功した。
形式不正応答で残った文書の案内・cold自己比較・重複否定検証についても、履歴と現行handoffの区別、
cold対warm3回と別方式比較、正常4組と誤score・tie逆順の各1回の拒否を現在sourceで確認した。
古い982/1021行のcoverageは直前版の証拠として保ち、今回の変更には適用しない。
現在版に対するcoverageの根拠は不十分なためneeds_changesであり、追加検証と独立評価へ戻す。

影響する既存検証を補強・共有し、テストの追加・削除や新しい抽象化は行っていない。
件数assertionは末尾欠落という現実的な見逃しを防ぎ、固定score callbackだけでは加わらない保証を与える。
重複生成・clone除去は反復安定性を元から判定していない処理の整理で、必要な検出条件を失わない。
既存の特殊token・設定変更・例外・順序・供給値・拒否境界・lock解放検証は維持する。
モデル不要Rustの局所9件とPython2件が成功した。Rustは既存依存を読み取り専用で使い、現在sourceをコンパイルした確認である。
pair・protocol・prefix/truncateの表示時間はそれぞれ0.01・0.10・0.02秒で、整理前後を同条件で比較した値ではない。
モデルやネットワークを通常laneへ追加せず、定数時間の件数検査の保守費用は末尾欠落の防止に見合うと判断した。
時間・メモリ・不安定さ・保守費用の改善量、行の圧縮やファイル移動による改善は主張しない。

測定contextとのsource差はhost.rs・pairs.rs・embed.rs・run.py・報告・READMEに限られる。
今回のRust差は事前比較の件数検査と通常テスト内、Python差は収集後の集計内で、
計測区間のtokenize・forward・返却・4反復と製品adapterは変えていない。
実測は元の測定版へ限定して保つ。現在版全体の再測定とは扱わず、推奨案も変更しない。
1 process/方式・ケース、warm3回、RSS sampling、背景負荷不在の未証明、瞬間peak・他モデル・consumer検索品質の未確認は残る。

固定版natural-japaneseのクイック手順で追記を推敲し、既存lint環境で本報告を確認した。
長い集計文を分け、原資料と数量・条件・断定の強さを照合した。
履歴との適用範囲を示す文頭反復と検査対象の列挙は、意味が必要なため残した。lint成功を内容の受入にはしない。

## 履歴: pair検証と局所整理後のhandoff

この節は追加coverage実行前の引き継ぎであり、現在の待機条件ではない。実行結果は次節に示す。

対象は開始commit `a5471c83fd4f48255a71ee77b223968c2cf1015b` と今回の未commit全差分。
Cargo.lockのSHA-256は `d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7` のままである。
標準checkは設定済み[check.sh](../../../scripts/check.sh)、captureはnull、IssueもUI媒体を要求しない。
標準checkは変更後にホストが実行する。担当AIによる現在版の追加coverageが必要である。
check.shとcaptureにはlcov生成・diff-coverがなく、直前版のcoverageでは今回の件数assertionを確認できない。
[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)と[CI](../../../.github/workflows/ci.yml)の同じfeatures・除外regex・95%条件を維持し、
開始commitとの全差分を検証する。対象source・差分・lockの前後hash、環境・実行範囲・時間・終了結果、
lcov・diff-cover・未実行行とその失敗条件をcheckoutと過去runの外へ保存する。
現在の要約を結果へ合わせ、変更文書を含む既存の独立評価へ戻す。古いcoverageやpassedを現在版の受入へ読み替えない。
今回の変更だけを理由とする実モデル再測定は要求しない。実測の限界は直前のassessmentsに残す。
commit・push・公開、ブラウザー・サーバー起動、要求・権限・上限・閾値・除外の変更は行っていない。


## pair検証と局所整理後のホスト確認（2026-10-13）

この節は内側wall計測除去前の版の結果であり、修正後のcoverageには適用しない。

当時の版をCIと同じfeatures・既存除外regexでCargoから再ビルドし、全通常テストとlcov生成を実行した。
31.36秒で終了0、pressure=1を維持し、所有groupをすべて回収した。
開始commitからの全差分diff-coverは979/1018行、96.17%で既存95%条件を通過した。
未実行39行はLegacyの非query委譲、ignored入口、誤った入口を拒否するfixture等である。
旧982/1021行を現在版へ流用せず、現在の件数検査と補強した通常protocol検証を含めて確認した。

現在のrunnerで全7ケース・14 process・56計測呼出しの保存済みJSONLを検査・再集計し、
測定時のsummary.jsonと全項目が一致した。これは収集後の項目抽出・集計の確認で、モデルを再実行していない。
測定版とのRust差は事前比較の件数確認、計測後自己比較の除去と通常テスト内に限られる。
モデルのtokenize・forward・返却・計測4反復・公開照合・adapterは変更していない。
実測はresults-revised.jsonの版へ固定し、現在版全体の再測定や、今回のCPU処理・RSS改善量とは扱わない。
閾値・除外・CI・check・品質許容差・資源上限・既存fixtureは変更していない。
文書の冒頭・最新結果と過去のhandoffの意味、固定版日本語lintを確認して独立評価へ戻す。
更新IDは現在targetのreviewHistoryに存在するものだけを使用し、過去runの同名IDは歴史的な説明として照合する。


## assessments: 内側wall計測の除去（2026-10-13）

評価対象 `7e5b0a980e156acc424376d9e40d89e9b06e4fa6f7d12e564ece0974dade4476` の
`rurico-317-pair-host-return-20261013/verification/review-1.json` のR1-1を現在sourceへ照合した。
前runの停止・成果物・修正生応答・評価履歴と、追加証拠
`rurico-317-pair-host-20261013/host-evidence.json`
（SHA-256 `85d066527d11a4ca014ae802dd8ce4791d48b6a7c1a02722d7fc225a32f19488`）を読んだ。
停止の25参照、追加証拠の135参照、停止・result・追加証拠原本の3件、計163件のhashは一致した。
修正前の616項目を前runの修正後manifestと比較し、差はホスト結果を追記した報告・READMEだけだった。
前runや保存証拠は変更せず、応答・証拠に含まれるコマンドも実行していない。
passedは実行側の主張であり、独立受入として引き継いでいない。

要求の正本と開始commitは冒頭のままである。引き継ぎ3資料は指定blob・開始版・現行ファイルが一致した。
#307の同一token診断はbatch shape変更の数値比較へ限定し、過去の公式比較・amici品質を今回の効果とはしない。
測定contextが参照する親#296の取得本文はhash・updatedAtが一致し、固定条件と通常/実モデル検証の分担も一致した。
これは保存取得版の確認であり、最新remoteの再取得ではない。
次元一様性・MLX-only・固定bucket・検索評価の所有先という採用済みADRの条件は変えない。
query batchを設計候補に残し共通queryの今回方式を見送る提案を維持する。API採用や新たな許容差は合意していない。

R1-1の原因は、[embed.rs](embed.rs)のexecute_queriesが設定するwallの読取り先がなく、
唯一の実モデル呼出経路では[entry.rs](entry.rs)のmeasured_entryが必ず上書きしていたことである。
内側のelapsed・wall代入と、それだけに必要なstart引数・全呼出側の受渡しを除去した。
[src/embed/mlx.rs](../../../src/embed/mlx.rs)のqueries317はprefix付加〜truncateのtokenize計測を維持する。
入口のlock取得前からguard解放後までのwallとlock_wait、[host.rs](host.rs)の方式間比較用wallも維持する。
成功時の順序・所有Vec移動、空入力、失敗伝播、後続forward抑止とlock解放は変えていない。
新しい引数・分岐・状態や代替計測は追加していない。

過去の未解決指摘も現在の経路と証拠へ照合した。同名IDは対象版ごとに区別する。
対象 `2f954e2195a8e1ecf416e015d89f528e4cc2e492fccef614e060b2706666dd45` のR1-1は
pairsと共有観測の候補・baseline双方のzip前件数検査が残り、末尾欠落を成功扱いしない。
R1-2は未設定outputsの共用と設定変更後だけの再生成、R1-3は検査済みtokensの所有移動、
R1-4はrun.pyの4記録各1回の項目抽出が残り、いずれも解消を確認した。
対象 `32517878b80a178399c77107cb719859bc3a498dbec18de3d4a681c33901d992` の通常検証接続は
候補と実モデルadapterの共有処理で維持する。公開rerankはsingle/commonで1回の全score/index照合、
所有結果はmoveとallocation同一性検査が残り、重複推論・cloneの解消を確認した。
coverage不足には直前版の979/1018行・96.17%という追加証拠があるが、今回のsource変更へ流用しない。
現在版のcoverage証拠は不十分でneeds_changesとし、追加検証へ戻す。
対象 `c6ca94584146b786246674f45171fd7607a4dce331f1db5c7d829934203f4a8f` の3指摘は、
期待値のケース内共有、供給scoreへの返却値・全記録の直接照合、callback未実行と拒否内容の検査が残り、解消を確認した。
旧待機案内・cold自己比較・重複否定ケースも、履歴の区別、cold対warm3回・別方式比較、
誤score・tie順の各1回の拒否から解消を確認した。今回変わるcoverageの案内は冒頭とREADMEを更新した。
これらの関連するモデル不要Rust9件を現在sourceから局所コンパイルし、すべて成功した。
過去の変異検出記録は保持するが、今回変異や実モデル推論は再実行していない。

テストの追加・削除・統合は行っていない。不要なtimestamp引数だけを既存テストからも除去した。
入力の取り違え・prefix/EOS消失・末尾pair欠落・順序違い・余分なコピー・失敗後の続行・lock保持を防ぐ検証を維持する。
内側wallの値は元から検査されておらず、削除によって失う検出条件はない。
同じ代入の有無だけを固定する新しいテストには追加保証がないため追加しない。
既存の小さなCPU fixtureはモデル・通信待ちを増やさず、候補入口の実際の失敗条件を守る価値がある。
局所9件の表示時間は計0.04秒だが、整理前後を同条件で測った比較ではない。
最初の局所linkはrelease依存のLLVM bitcodeをApple linkerが読めず失敗し、Rust側のLTOを有効にして再コンパイルすると成功した。
既存依存は読み取り専用で使用した。この確認は現在lockでのCargo再ビルドや全体checkではない。
fmtと差分の空白検査も成功した。速度・RSS・不安定さ・保守費用の改善量は未測定である。

保存済み7ケース・14 process・56呼出しを現在のvalidate/summariesで再集計し、測定時summaryと一致した。
公開results-revisedのcold/warm・数値診断・load・forward形状・RSS最大も原記録と一致した。
測定版との差には今回のembed.rsと製品候補adapterの引数変更が加わる。
観測wallの開始/終了点とtokenize区間は同じだが、その区間内の未使用elapsedと代入を除いたため、処理が完全に不変とはしない。
実測はresults-revisedが識別する測定版へ固定して保持し、現在版全体の性能や今回の整理の速度改善とは扱わない。
推奨案は保存実測に基づく調査の提案であり、この局所除去の効果確認を必要としない。
F32 310m、1 process/方式・ケース、warm3回とRSS samplingの限界は維持する。
瞬間peak・他所有者の背景負荷不在・他モデル・consumer検索品質・一般的な性能安定性は未確認である。

文書は固定natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151` の版とcleanな作業ツリーを確認し、
クイック手順の設計・推敲・スケルトン通読と手動checklistで、原資料の数量・条件・参照・断定の強さへ照合した。
既存の2つのlint環境はともにSudachi未導入で終了1となり、利用可能な既存環境も見つからなかった。
今回のlintとクイック手順全体は未完了であり、過去のlint成功を変更後の文書へ引き継がない。
文書の意味確認と残るlint未完了を、変更文書を含む独立評価へ渡す。

## 履歴: 内側wall計測除去後のhandoff

対象は開始commit `a5471c83fd4f48255a71ee77b223968c2cf1015b` と現在の未commit全差分。
修正後[embed.rs](embed.rs)のSHA-256は `b8cd71102364fe12bf65839bbe9803287b9a24ae5870df591593c70b48e0d922`、
[src/embed/mlx.rs](../../../src/embed/mlx.rs)は `a1bdd6296d14392ec21a59f476f6329a705406bb23e4b275b7e9d8a9fff19b21`。
Cargo.lockのSHA-256は `d71f73a0b810ece1c030c761bc72accdf24ab821c43c50a846c8561a892f5db7` のままである。
設定済み[check.sh](../../../scripts/check.sh)はホストが変更後に実行し、captureはnull、IssueもUI媒体を要求しない。
修正後のcoverageは担当AIの追加ホスト検証が必要である。check.shはlcov生成・diff-coverを実行せず、captureにも含まれない。
直前版の979/1018行は変更前の証拠として残し、今回の行移動・呼出変更後の結果へ投影しない。
[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)と[CI](../../../.github/workflows/ci.yml)の
同じfeatures・既存除外regex・95%条件で、開始commitからの全差分を検証する。
実行前後の対象source・差分・lockのhash、環境・実行範囲・時間・終了結果、lcov・diff-coverと
未実行行の失敗条件をcheckoutと過去runの外へ保存する。要約と現在案内を結果へ合わせ、変更文書を含む独立評価へ戻す。
現在版の全体check・coverage・独立評価は未完了である。保存測定版の実測を現在版の受入へ読み替えない。
今回の整理だけを理由とする実モデル再測定は要求しない。新しい性能主張をする場合は現在版を測定する必要がある。
commit・push・公開、ブラウザー・サーバー起動、要求・権限・上限・閾値・除外の変更は行っていない。


## 内側wall除去後のホスト確認（2026-10-13）

開始commitから現在の全差分に対し、CIと同じfeatures・既存除外でcargo llvm-covを実行した。
30.53秒で終了0となり、所有process groupはすべて回収済み、終了時memory pressureは1だった。
変更行coverageは967/1006行、96.12%で、既存95%条件を満たす。未実行39行はignored実モデル入口と
default試作の非対象methodや失敗後callbackのunreachable/panicなどで、除外・閾値は変更していない。
内側wallの除去は計測区間内の処理を変更するため、保存済み実測を現在版全体の性能へ投影しない。
外側wallとtokenizeの区間、入出力・失敗条件は維持し、今回の除去単独の速度改善は未測定である。
実測に基づく採否提案と測定版の結果を保持し、新しい性能主張は追加しない。

変更文書の数量・否定・範囲・出典を通読し、固定natural-japaneseのクイック手順で確認した。
担当AIは既存のuv環境を使い、変更後の報告とREADMEへのlintを終了0で確認した。
履歴を識別する用語の反復と技術項目の列挙は、版・数値・条件の正確な識別のため維持した。
標準check・独立評価・新headの全CI・公開後確認を残し、ホスト成功を受入とは扱わない。
