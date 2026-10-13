# Issue #316: 文脈選択とscore集約の比較報告

既定APIを維持し、短queryの低コスト用途ではchunk周辺、長文の根拠保持ではwindow/maxを次の採用検討候補にすることを推奨する。
2026-10-13の固定4query・12候補の比較ではchunkの短queryが約4.3秒、windowが約44～62秒だった。
query枠確保は長queryでdocumentが全消失する状態を防いだが、末尾根拠は失った。
方式の製品採用・品質/応答時間の許容条件は未合意であり、本比較から実利用全体の品質を保証しない。
要求の正本は [Issue #316](https://github.com/thkt/rurico/issues/316) と合意記録。
方式採用、consumer実装、品質・時間の新しい許容差は含めない。
[README](README.md)に比較方式、実行入口、測定条件、解釈の限界を示す。

## 固定入力での実モデル比較（2026-10-13）

[model-results.json](model-results.json)に全桁の指標、全候補の順位・score、初回とwarmの各記録、
時間の最小/中央値/最大、process RSS、元JSONLとbinaryのhashを示す。
同じ12候補と4queryを各方式で初回1回・warm3回測定した。
表の品質値は初回、時間・forward・Metal peakはwarm3回の中央値。
各方式・queryの全4回でscore・順位・品質値は一致した。共有JSONでは一致するwarmの集約値をtrial 0への参照で保持し、時間・forward・peakは各回を残した。
Recall@10は全組合せで1だった。

| 方式 | query | Recall@5 | nDCG@10 | warm時間（秒） | forward | Metal peak（MiB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 日本語・短 | 0.75 | 0.876849 | 29.611 | 12 | 3290.50 |
| baseline | 日本語・長 | 1.00 | 1.000000 | 58.814 | 12 | 3290.50 |
| baseline | コード・短 | 0.75 | 0.961999 | 29.599 | 12 | 3290.50 |
| baseline | コード・長 | 0.00 | 0.483639 | 58.812 | 12 | 3290.50 |
| query-prefix | 日本語・短 | 0.75 | 0.876849 | 29.537 | 12 | 3290.50 |
| query-prefix | 日本語・長 | 1.00 | 1.000000 | 33.445 | 12 | 3306.50 |
| query-prefix | コード・短 | 0.75 | 0.961999 | 29.603 | 12 | 3306.50 |
| query-prefix | コード・長 | 1.00 | 0.982892 | 33.450 | 12 | 3306.50 |
| chunk-neighbors | 日本語・短 | 1.00 | 0.904717 | 4.341 | 12 | 2486.48 |
| chunk-neighbors | 日本語・長 | 1.00 | 0.955830 | 29.279 | 12 | 3306.50 |
| chunk-neighbors | コード・短 | 1.00 | 1.000000 | 4.334 | 12 | 2742.48 |
| chunk-neighbors | コード・長 | 1.00 | 0.982892 | 33.509 | 12 | 3306.50 |
| windows/max | 日本語・短 | 1.00 | 0.955830 | 44.157 | 18 | 3291.50 |
| windows/mean | 日本語・短 | 1.00 | 0.904717 | 44.157 | 18 | 3291.50 |
| windows/max | 日本語・長 | 1.00 | 1.000000 | 61.942 | 18 | 3307.50 |
| windows/mean | 日本語・長 | 1.00 | 1.000000 | 61.942 | 18 | 3307.50 |
| windows/max | コード・短 | 1.00 | 1.000000 | 44.452 | 18 | 3307.50 |
| windows/mean | コード・短 | 1.00 | 1.000000 | 44.452 | 18 | 3307.50 |
| windows/max | コード・長 | 1.00 | 0.982892 | 62.089 | 18 | 3307.50 |
| windows/mean | コード・長 | 1.00 | 0.982892 | 62.089 | 18 | 3307.50 |

**長queryのbaselineは全12候補が同点で、document tokenは全候補で0だった。**
日本語のnDCG@10=1は元index順がラベル順に合った結果であり、文脈評価の成功ではない。
コードのRecall@5=0との違いも、この候補順の影響を含む。macro平均だけで方式を選ばない。
query-prefixは候補を区別でき、長いコードqueryではRecall@5=1、nDCG@10=0.982892になった。
同時に全文の末尾根拠は残らず、主題の一致だけで高scoreになる候補もある。

短queryのchunk-neighborsは約4.3秒で、baselineの約29.6秒より約85%短かった。
短queryでは長文6件も2048 bucket以下へ収まった。長queryではquery枠1024 tokenに
選択documentと特殊tokenを足すと2048を超え、日本語5件/コード6件が8192 bucketへ入った。
例えば2052 tokenでも8192までpaddingするため、長queryの時間は約29～34秒だった。
これは固定bucketの観測であり、本Issueでpadding方針を変更していない。

windowは全位置の根拠を残し、forwardが12→18回になった。
短queryは約44秒、長queryは約62秒で、query-prefixより約1.5倍/1.85倍かかった。
日本語短queryではmaxのnDCGがmeanより約0.051高く、他3queryのnDCGは同じだった。
同じ指標でも順位は異なるため、全候補順をJSONに残した。max/meanで推論を二重に実行していない。
日本語短queryでは誤答の短文がmaxで3位、meanで2位だった。
関連主題の順位と回答の正しさは一致しない。maxも虚偽回答の除外を保証しない。

process全体のRSS観測最大は次のとおり。

| 方式 | RSS観測最大（MiB） |
| --- | ---: |
| baseline | 2098.98 |
| query-prefix | 2099.50 |
| chunk-neighbors | 2037.59 |
| windows | 2099.70 |

これは方式ごとのprocess最大であり、queryごとの瞬間peakではない。
Metal peakはweight/cache込みの呼出しpeakであり、RSSとの加算やGPU全量への換算をしない。
比較の順序・熱・他所有者の負荷を無作為化した因果実験ではなく、製品batchの性能保証でもない。

### 再現条件と期限停止

Mac15,3/Apple M3、RAM 24 GiB、macOS 27.0.1 (26A434)、Rust/Cargo 1.99.0、
Xcode 27.0 (27A266a)、Metal 32023.921、Python 3.15.0を実出力へ照合した。
AC電源、測定前のthermal warningなし。担当AIの別ビルド・GPU作業は同時実行していない。
他所有者のGPU作業がないことは未証明で、他所有者のprocessは終了していない。
固定F32 rerankerとtokenizer、release binaryは前後hash一致。入力は独立fixtureの固定hash
`a5c7e0806da5882988968df1d5af9d3f17546955d13706ff59d0b4c7af433837`。
実際のCargo compiler-artifactと保存binaryはSHA-256
`f47b7f34d74ef80cbad9f5e79979c57539444a59a2e2b9def5e6f5b34512e27d`で一致した。

modelを全4方式まとめて実行した最初の測定は1800秒で停止した。
baseline/query-prefix/chunkの各16回は終了0とprocess group不在を確認済み。
windowは未完了で、旧runにcompleteを追加せず、部分結果を最終比較に使わなかった。
同じcollectorの対象をwindowsだけに限定する外部driverで、全4query・各4回を別保存先へ測定した。
このdriverは対象方式だけを限定し、入力生成・token処理・forward・集約・回収・hash照合は変更していない。
別測定は926.76秒で終了0。各phaseの期限は1800秒のまま。所有RSS 8 GiB超、pressure悪化、swap増分256 MiB超で停止する条件も維持した。
両phaseの最後に所有group不在とpressure=1を確認した。元の期限停止を成功へ変更していない。
[README](README.md#方式を分けて収集する)に分割収集の再現方法を示す。

ホストで実測JSONLの候補/境界/特殊token/8192上限と実forward数を確認し、
全方式・全trialのraw token列一致、短文正常対照48組のIDs/mask一致を照合した。
Recall/nDCGを固定ラベルから別計算してcollectorの値と一致した。
生ログ・環境・driver・監視・監査はcheckoutと旧runの外へ保存した。
共有要約のcollection_completeは収集完了を表す。通常check・独立テスト・独立評価・CIの合格を代用しない。
この測定直後は報告・要約・コメントだけを整理した。2026-10-14の観測構築の修正と、
旧測定を適用できる範囲は下記「検証状態」で区別する。

## 根拠と適用範囲

開始commit `1d81211bf7571598f0a5c98663703c21f13d056b` を確認した。
引き継いだ3報告の現在のGit blobは指定版と一致し、内容の変化はない。

| 出典 | 固定blob | 今回適用すること |
| --- | --- | --- |
| [#307報告](../issue-307/report.md) | `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 公式数値比較と固定amici品質の完了済み観測、実測と診断・製品許容差の区別 |
| [accepted ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md) | `1e2403400b586d3a09907eb12998f4b6c2f2fa81` | end-to-end検索評価はamici所有。本調査で検索pipelineを複製しない |
| [accepted ADR-0009](../../decisions/0009-adopt-bucketbounds-as-cross-module-forward-pass-invariant.md) | `e96818842c029af9466cb4b0891d10d20d68f489` | 既存bucketをforward経路で維持し、padding変更と文脈変更を混ぜない |

#307は当時の版・モデル・fixture・環境の観測であり、今回の文脈方式の品質保証ではない。
同報告の未採用仮説、既存amici baselineの許容差、同一token数値診断を、
新方式の品質・応答時間の合意へ昇格させない。
Issueの旧source `32513da...` / `c8f250d...` に代わって開始版の
[`processing.rs`](../../../src/reranker/processing.rs)も照合した。
現行は結合pairの末尾切断であり、query枠の予約やwindow集約は既定動作にない。
ローカルGitで`c8f250d...`から開始版への差分を確認した。
`truncate_pair`の切断処理は同じで、score有限性検査の分離、公開batch組立の共通化、
dispatch telemetryと#309/#317の研究経路が追加されている。
今回の比較は開始版のforward/有限性検査/cleanupを使い、旧版の実測へ投影しない。
独立fixtureはこの損失量を調べる条件を補うもので、末尾保持を製品の回帰期待値にしない。

親 [#296](https://github.com/thkt/rurico/issues/296) は実装セッションでDNSエラー/Cache missとなったが、
実測前にホストが本文を取得して実行・保存条件を確認した。
updatedAtは`2026-10-07T17:08:32Z`、取得JSONのSHA-256は
`515b12dd4e34c42af8399e03d5f86dbf65a31cbb7858f82518c813f52a6e506d`。
同入力・同モデル・cold/warm・shape/時間/peak・固定環境・公開syntheticのみという条件を適用した。
公開remote mainも実測前に取得し、開始commit `1d81211bf7571598f0a5c98663703c21f13d056b`と一致した。
実装セッションの取得失敗と、ホストの取得成功を別記録として保存している。

## 比較で確認すること

query-prefixはdocument枠を確保するが末尾根拠を失う可能性がある。
chunk-neighborsは局所文脈を取得できるが、token一致という代理尺度で誤ったchunkを選ぶ可能性がある。
windows/maxは局所根拠を拾う可能性があり、無関係windowで高scoreが出る機会も増える。
windows/meanは複数windowの関連性を反映するが、少数の根拠を希釈する可能性がある。
これらは仮説であり、採用や効果確認ではない。

同じ12候補について4queryの位置別順位、Recall/nDCG、残るtoken列・境界・根拠、
forward回数、時間、Metal peakとprocess RSSを比較する。
根拠だけの短文はモデル経路に到達する正常対照、誤答と無関係文書は誤った選択・scoreを区別する対照。
特殊token・境界・最大長を保つ候補と、現行baselineの境界消失の観測を分ける。
詳細・集約の意味・同点規則・測定範囲は[README](README.md#比較する方式)を参照。

## 検証状態

2026-10-14の最初のホスト検証では、独立担当が準備した研究CLIのRustテスト3件が成功した。
coverageの生出力は、当時の`context.rs`の変更228行中218行、95.6%の被覆を示した。
ただし、測定時のcargo・rustc・cargo-llvm-cov・LLVM・diff-coverの解決実体と版、
diff-coverの完全なargv・終了結果を対応付ける記録が不足している。
この最初の出力だけでは95%条件の受入確認を完了できず、旧環境の観測やactorのCLI版では補完しなかった。
その後、全文根拠index共有後の全差分を同じ除外・閾値・開始commitで再測定し、223行中213行、95.5%で終了0を確認した。
Cargo/Rust 1.99.0、cargo-llvm-cov 0.8.4、LLVM 23.1.1、diff-cover 10.6.0の解決実体と版、
完全なargv・終了結果・source/lock・lcovを新しい外部記録へ対応付けた。
Cargo/RustとLLVMを固定実体で指定し、diff-coverのwrapperが使うPythonの実体・版も確認した。
測定前後のsourceとツールhash・版は一致し、所有processは全て回収した。
通常の`bash scripts/check.sh`はこのcoverage測定を実行しない。
独立評価と同じheadのCIは未完了で、PRは未作成。

独立テスト工程には、ユーザーの今回限りの承認により研究コードを修正前の固定版として渡した。
元の開始commitとIssue、資料3件、95%条件を維持し、実装の会話・判断・測定結果は渡していない。
`tests/context_plan.rs`は合成tokenizerと隔離cacheで実際のplan入口を呼び、
短文・空文書の正常対照、UTF-8 offset、特殊token・境界・最大長、windowの全体被覆、
不正spanの拒否と既存出力の保全を検証する。
最初の3件では選択規則と根拠分断の判定が不足していたため、後続の独立工程で次を補った。

- 長queryの異なるprefixと末尾1024 tokenの内容・個数・offset、document先頭の固定範囲を判定する。
- chunkの一致回数最大、同点時の先頭、破棄queryの一致を無視する条件、零一致の対照を判定する。
- overlapを超える300 token根拠の分断を完全保持と報告しない条件を、単一windowへの包含、overlap内の包含、端点一致の正常対照と区別する。

現在のRust対象は5件。全文包含の処理は追加ケース以前からあり、独立工程での5件成功は
実装不具合の修正前Redではなく、検出条件を補った正常版の実行である。
`scripts/test_context_spike.py`の3件は人工scoreからmax/mean・同点順・Recall/nDCG・warm集計を判定し、
候補・call・forwardの欠落、非有限score、token境界・bucketの不整合、planの測定誤表示を拒否する。
既存Python discoverに接続され、最初の独立工程では追加3件を含む37件が成功した。
合成入力と人工scoreは、実tokenizer固有の分割・実モデル順位・検索pipeline品質を保証しない。
旧queryの同一token反復だけの条件は置換されたが、上限超過と切断条件は維持し、誤選択を識別する内容を得た。
小さな集計検査と実CLIの境界・保全検査には異なる保証があり、統合・削除は提案しない。
独立工程のRust5件の所要は20.44秒。同条件での費用増分、不安定さ、保守費用は未測定である。

2026-10-14のR1-3修正では、全文根拠token indexの走査・確保をwindowループの外へ移し、
同じdocument処理内で一度だけ計算して各windowへ共有する。
局所根拠・EOS検査、範囲包含判定、各forwardと観測、collectorの工程前後hash照合は維持する。
選択・推論・集約とfixtureは変更していないが、wallに含まれる観測構築費用は変わる。
上の時間・peakは旧binaryの観測として保存し、現行binaryの測定値や改善量には読み替えない。
修正後、固定実tokenizer・同じ4query/12候補を現在のdebug binaryのplanで全4方式実行し、
load/wall以外の全記録が旧planと一致した。モデルforwardや性能改善の測定ではない。
既存Rust5件とPython集計3件も終了0で、選択・分断・境界・保全の正常動作を確認した。
これらの生ログとhashはcheckout外へ保存し、独立評価へ渡す。
測定来歴を補った同時点のcoverageは上記のホスト再検証で確認した。後述の追加修正後のcoverageには引き継がない。
前回のimport・コメント整理時のbinary byte一致は、その時点に限る証拠である。
同時点の全体check成功（34.98秒）も今回版のcheck成功には引き継がない。
テスト・fixture・check接続はこの修正セッションで変更していない。

2026-10-14の後続修正では、同じwindowのsequence ID列を`observe`と呼出元で
二度構築していた箇所を、呼出元で一度生成して借用共有する形へ変更した。
EOS補正用の別コピー、局所根拠・全文包含検査、raw観測、forwardと工程前後hash確認は維持する。
既存のRust5件は19.63秒で成功した。固定実tokenizer・同じ4query/12候補の全4方式planについても、
load/wall以外の記録一致を再確認した。旧時間・peakは旧binaryの値であり、今回の速度改善量は未測定である。

同じ後続の独立テスト工程で、集計器が合法bucketへの所属だけを検査し、
windowの実token長との対応を検査していない欠陥を再現した。
修正前は正常なround-up対照が成功し、合法だが不一致の24条件は期待した拒否が起こらず失敗した。
これは対象集計器まで到達したRedで、MLX取得失敗やテスト被覆の不足とは区別する。
集計器を各windowの実長から最小の既存bucketを判定するよう修正し、
独立担当が追加した2件を含むPython5件が成功した。正常境界・不一致拒否と既存集約の正常動作を確認した。
保存された旧モデル64 call・864 windowも現在の厳しいbucket判定を通り、
再集計した保持文脈・score・順位・品質・時間・forward・peakは全16組の共有要約と一致した。
この再検査は新しいモデル推論や現在版の性能測定ではない。
今回の実装修正ではテスト・fixture・検証接続を変更していない。
追加修正後の全差分も、同じ固定実体・版、features・除外・開始commitでホスト再測定した。
223行中213行、95.5%で`--fail-under=95`は終了0。完全なargv・終了結果・source/lock・lcov、
前後の実ツール版・hash、diff-coverが使うPython実体・版を新しい外部記録へ対応付けた。
記録されたsourceとツールは測定前後で一致し、所有processは全て回収、ガードも維持した。
通常checkにはcoverage測定が含まれないため、独立評価には別の測定証拠として渡す。
子Bunの来歴不足は、2026-10-14の新しいホスト検証で補った。
source snapshotの取得に実際に使うBun 1.4.3の入口・realpath・版・hashとargvを記録し、
呼ぶハーネスのGit commitと直接関係する4ファイルの限定hashも前後で照合した。
これは依存全体の再現性を保証する記録ではない。同じfeatures・除外・開始commit・95%条件と
所有process・メモリガードを維持した新規coverageも223行中213行、95.5%で終了0だった。
source・ツール版・限定ハーネスhashは前後で一致し、diff-coverの子Python 3.12.13も実体と版を観測した。
旧記録は保持する。通常check・独立評価・公開後CIの結果は、この後の各runと対象commitで照合する。

その後のcollector修正では、方式ごとのprocess RSS最大値をqueryごとに再集計せず、
同じsamplesから一度だけ求めて全queryへ保存する。方式フィルタと空観測時の`None`、
実行中のsnapshot、process回収、工程前後のhash確認は維持する。
保存monitorを使った局所照合では、各方式の4queryへの値は修正前と一致し、
samplesの走査は4回から1回になった。空観測・別方式だけの観測では`None`を維持した。
この修正後に既存Python5件が成功し、独立工程が4方式へ拡張した正常境界と
4方式×24条件の不一致拒否、既存の集約・不正記録拒否・plan区別を確認した。
旧モデル64 call・864 windowの再集計も共有要約に一致した。これらはモデルの再実行ではない。
保存済みの時間・peak・RSSは旧収集の観測であり、今回の速度改善量は未測定である。

2026-10-14の保持数の修正では、decode用に抽出したquery/documentのtoken列の長さを
保持数として保存し、同じwindowを数え直す二つの走査を除いた。
EOS補正、局所根拠と全文包含の検査、raw観測、forwardは維持する。
修正後の既存Rust CLI5件とPython集計5件は成功した。これは正常動作の確認であり、
重複走査を修正前の機能テストのRedとして扱わない。
固定実tokenizer・同じ4query/12候補の全4方式planも、load/wall以外の記録が旧planと一致した。
保持数はbaselineを含む全216windowのsequence列とも一致した。モデルforwardや速度改善の測定ではない。
旧モデル64 call・864 windowの再集計は全16組の共有要約と一致したが、新しい推論ではない。
上記の変更前測定とは別に、保持数の修正後の全差分をホストで再測定した。
同じ開始commit・features・除外・95%条件で223行中213行、95.5%となり、
llvm-covとdiff-coverは終了0だった。未被覆は136・183・279～286行である。
実子Bun・diff-coverの子Pythonを含む固定ツールの実体・版・hashと完全argvを記録し、
sourceとツール、限定ハーネスhashの前後一致、所有processの回収を確認した。
通常check・独立評価・公開後CIは、それぞれの後続runと対象commitの記録で照合する。
局所実行の生ログ・対象hashと残る条件は、checkout外の新規記録から独立評価へ渡す。

2026-10-14の後続修正では、truncationとpaddingがないことを検査した直後の
`with_truncation(None)`と不要な`mut`を削除した。設定のassertと後続処理は維持する。
既存Rust CLI5件とPython集計5件は成功し、選択・境界・根拠保持・失敗時の記録保全と集約を確認した。
この冗長処理は機能テストの修正前Redではなく、速度改善量も未測定である。
削除後の全差分も、同じ開始commit・features・除外・95%条件と資源ガードでホスト再測定した。
222行中212行、95.5%でllvm-covとdiff-coverは終了0、未被覆は136・182・278～285行だった。
子Bunと子Pythonを含む固定実体・観測版・hash・argvを記録し、sourceとツール、
限定ハーネスhashの前後一致と所有processの回収を確認した。
この記録を渡し、後続の全体check・独立評価・公開後CIは各対象版で照合する。

最初のCI相当coverageは研究経路を実行せず、228行全て未被覆で95%ゲートに失敗した。
後続checkも入口の完全修飾呼出しがClippy `absolute_paths`で拒否され、終了101だった。
入口をimportへ整理し、研究経路の不足は上記の独立テストで解消した。
旧失敗記録と未完了の測定記録は保存し、成功へ書き換えていない。
初回のテスト判断と後続のlint向け「追加不要」は、研究経路全体の保証へ読み替えなかった。

独立環境の新規targetではMLX取得のDNSエラーによりRustテストへ到達できなかった。
同じ依存を使う一時crateの型検査・Clippyは成功したが、実接続の証拠には数えない。
最初のホストの既存build環境では実CLI接続とcoverage出力を得て、所有processを回収した。
この実行に対応する測定ツールの版・来歴は、上記のとおり未確認が残る。
メモリpressureは正常を維持し、メモリ・期限ガードの緩和は行っていない。

### 固定tokenizerでの局所観測（2026-10-13）

[plan-results.json](plan-results.json)は修正版のJSONLを`run.py`の`summarize`で集計した観測と、
正常対照・境界の照合結果。正本のJSONLのhash、source/lock/input/tokenizerのhashを含む。
Rust 1.99.0、Python 3.15.0、固定reranker tokenizer、モデルforwardなしの手動ビルドで実行した。
研究sourceはSHA-256 `dae690cc0653bb03d0a2ca558bbd2423d1846dae14bb59d592c4dfef51761a97`、
runnerは `49d74740af590e37e0b910b58ae08164f0623c9925037c09b1ff1c1382a82a84`。
生ログはcheckout外の新規保存先に保ち、独立評価への参照はfindingsへ渡す。

全方式でraw token列は同一。日本語queryは12/8204 token、コードqueryは14/12302 tokenで、
長queryの上限超過を実際に確認した。位置違いの日本語documentは8205 token、コードは12304 token。
全方式の短queryと短い正常/誤答/無関係documentの48組は、raw pairのIDs/maskと一致した。
候補168windowのquery/document sequence、特殊token数、BOS/EOS、8192上限を確認した。
これはモデル出力の一致や全入力の境界保証ではない。

| 方式 | 短query・同話題の先頭/中央/末尾根拠 | 長query・同話題の先頭/中央/末尾根拠 |
| --- | --- | --- |
| baseline | 保持/保持/欠落 | 欠落/欠落/欠落 |
| query-prefix | 保持/保持/欠落 | 保持/保持/欠落 |
| chunk-neighbors | 保持/保持/保持 | 保持/欠落/欠落 |
| windows | 保持/保持/保持 | 保持/保持/保持 |

日本語とコードの両方でこの表の観測は同じだった。別話題の候補も同じ集合のまま比較している。
windowsでは長文ごとに2windowを選んだが、forwardは実行していないため、実測forward回数とは呼ばない。
長queryの多くが背景語であるとtoken一致chunk選択が根拠を選べない、という条件も確認できた。
根拠の保持だけでモデル品質が上がるとは判断しない。

初回の局所観測では、短いdocumentの候補記録からqueryのsequence IDが消える不具合を確認した。
tokenizers 0.23.2の`set_sequence_id`は旧rangeを消さず追加するため、単独encodeしたdocumentの
sequence-0 rangeがpair再構築でqueryのrangeを上書きしていた。
sliceはtoken・offsetを保ち、古いsequence rangeを引き継がずpostprocessorに再割当てさせるよう修正した。
旧JSONLは保存し、修正版の同じ条件で境界と短文の正常動作を確認した。
この不具合と修正は研究の観測記録に限り、製品truncationの変更ではない。

この局所観測を記録した2026-10-13時点では選択・集約の独立テストは未作成だった。
現在の追加済み条件と実行・未確認事項は上記「検証状態」に統合した。
固定実tokenizerの局所観測、合成tokenizerの独立ケース、実モデルの比較は別の保証である。
恒久的な全GPU再測定や生成処理の写しを、件数やcoverageだけを理由に追加することは推奨しない。

## 推奨案と移行判断

既定APIを維持し、採用は別Issueで用途・品質/時間の条件を合意してから進めることを推奨する。
短queryで低コストが必要ならchunk周辺を次の検討候補とするが、今回のtoken一致代理尺度を
consumerの検索top chunkへそのまま採用しない。実際の文脈取得はconsumerが所有する。
根拠の位置を事前に絞れない長文ではwindow/maxを比較候補にする。全位置の根拠を残した一方、
18 forwardと約44～62秒が必要だった。meanは今回maxを上回らず、希少な根拠の希釈にも注意が必要。
長queryのdocument消失対策としてquery枠の予約は候補となるが、query末尾1024という条件は
質問を末尾に置いた今回のfixtureに限る。query先頭の重要条件を失う用途への適用は未確認。

採用判断では、利用側の固定候補・文書分布で位置別の品質と応答時間を比較し、許容値を先に合意する。
#307の固定amici結果は当時のreference compositionの根拠であり、今回の未採用方式の検索品質ではない。
今回のsynthetic Recall/nDCGは同候補集合のrerank観測で、候補生成や検索pipelineを置き換えない。
採用する場合も、現行`Rerank`を維持したopt-in方式の別IssueでAPIとconsumerの責任、
文脈取得、model/tokenizer識別、移行とrollbackを合意する。本Issueでは採用・移行を実装していない。
