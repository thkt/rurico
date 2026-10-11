# Issue #366 の候補検証結果

2026-10-11、修正版の検証入口で候補 `266fc41ff6bdf7ca5110e55a7315f8cbb5022cc6` を通し測定し、14:55:51 JSTに終了0で完了した。embeddingの23 batch・27行、rerankerの12 batch・16行は既存の数値基準を満たした。amiciの60文書・168クエリも既存baseline検証を通過した。数値・検索の両工程を新しい出力先で実行し、過去の成果物を合算していない。

## 入力と精度を維持して参照processを分離した

Apple M3・24 GiBのホストで、Python 3.12.13と現行requirementsを使用した。固定revisionの310mモデルはFP32、CPU参照はeager attention・4 threadsで実行した。既存inputs・比較器・amici固定commitを使い、入力、fixture、baseline、許容差を変更していない。候補source・lock・解決済みpackage・モデルと完了成果物のhashを照合した。製品sourceは指定commitから隔離したコピーで、検証入口の未コミット変更と区別する。終了後の遠隔mainも候補SHAと一致した。終了後の確認では、過去の着手時点の遠隔状態までは確かめられない。

公式参照のnative入力16件とbatch35件を、51個の新しい子processで順に実行した。前のprocessが終了してから次を開始し、元のwrapper・入力ID・順序・全batch shapeを保持した。混在batchを行単位へ分割していない。8190／8192 tokenの長文、chunkのpadding、5×512の混在batchも成功した。`reference-events.jsonl`で51件の完了と異なるworker PIDを確認した。

14:36:00〜14:55:49 JSTの5秒間隔の監視では、memory pressureは237標本すべてで通常（1）だった。別process groupも含めた測定配下のRSS最大値は約9.69 GiBで、終了後は両groupにprocessが残っていなかった。このRSSは瞬間peakやGPU・ホスト全体の使用量を示さない。モデルを毎回読み直すため、速度改善を示す測定でもない。

## 新しい隔離fixtureでamiciの既存評価を通過した

amiciは公開固定commit `547f9ee2ed734a2eab316fdbd62f194849875ee4` の隔離コピーを使った。通常/devのrurico依存だけを候補SHAへ差し替え、lock・metadata・取得sourceを照合した。元consumer、評価コード、fixture、baselineは変更していない。

以前の共有build cacheを残した状態で、出力先専用の空のtargetを依存解決・metadata・buildへ渡した。ビルドは終了0で270.05秒。生成した実行ファイルでcapture-baselineとverify-baselineを実行し、ともに終了0だった。両private logの実fixtureパスが、新しい隔離コピーと一致した。以前のcheckoutに埋め込まれたパスへの依存は、今回の実行で解消できた。

| 指標 | 候補の値 | 既存baselineとの差 |
| --- | ---: | ---: |
| Recall@5 | 0.6299603174603176 | 0 |
| Recall@10 | 0.7414682539682539 | 0 |
| nDCG@10 | 0.8929771775329307 | +0.00026194524855382983 |

Hit@1、Hit@3、MRR@10はいずれも1で差0。既存許容差内という結果であり、改善や実アプリ全体の品質は示さない。検索評価の所有権はADR-0006どおりamiciにある。

## 独立評価の5件を修正して再測定した

初期の通し測定はmemory pressure悪化で3回中断し、incompleteとして保存した。参照processを入力・batchごとに分離した版は13:31:23 JSTに通し実行を完了したが、検索評価は旧cacheに埋め込まれたfixtureパスへ依存していた。実入力9ファイルの内容一致は確認できたものの、その結果を修正版の隔離経路の証明には使っていない。

独立評価では次の5件が見つかった。

- 固定consumerが読まないfixture指定。
- 中断時の孫process残留。
- 同じ観測内での重複hash計算。
- 候補SHA否定テストが別の理由で通る条件。
- dirty lockによる誤失敗。

今回の修正版は、新しいbuild targetと実fixtureパスの検査を使う。数値工程の所有groupを停止・回収してからincompleteを記録し、同じidentityのlock・入力hashには取得済みsource mapを使う。工程をまたぐ再検査は維持する。

不正候補のテストは未使用の出力先と固有の失敗理由を確認するよう直した。dirtyな一時cloneから候補を隔離し、期待するlockを指定commitのblobから取得する。実SIGTERM後の子孫終了と書込み停止、旧targetからの分離と誤fixtureログの拒否も検証した。SHA検査を個別に外した改変と、dirty lockを期待値に戻した改変では、意図どおりテストが失敗した。既存テストの削除や検出条件の喪失はない。

追加評価で、検索工程でも中断時に孫processが残る欠陥が見つかった。数値と検索で回収処理を共有し、検索の依存解決・metadata・build・capture・verifyに適用した。metadataのJSONとstderrは別に保持し、baseline検証の終了1を許す条件は維持する。

修正後、固定Python 3.12.13の軽量入口では、参照17件と候補制御7件の計24件が成功した。実SIGTERMによる数値・検索両工程の子孫終了と書込み停止、metadataの出力分離、許可・拒否する終了コードを確認した。実Cargo metadataと既存targetのビルドも新しい回収処理を通って成功した。参照テストには、小さなローカルcheckpointでの分離前後の数値一致と、失敗後に次の入力を開始しない検証を含む。新規targetはビルドとディスク費用を増やし得る。今回はビルド時間を記録したが、同条件での前後比較や長期保守費用は測定していない。

[report.md](report.md)の指定blobと旧環境の[validation-353.md](validation-353.md)は保持した。今回の測定は、更新後のTorch 2.14.1・Transformers 5.18.0・Sentence Transformers 6.1.0による新しい観測である。過去の成功や保存JSONの再集計を、新しい候補の推論成功には読み替えない。

## 再実行と適用範囲

[READMEの候補検証入口](README.md#候補commitの継続検証issue-366)で候補SHAと新しい出力先を指定する。14:55:51の通し測定は、検索側へ回収処理を追加する前の版である。追加修正は外部commandの所有・出力保持に限り、候補source、モデル、入力、wrapper、batch shape、推論引数、環境、比較基準を変えていない。修正後の実モデル通し測定は未実施で、品質の実測と中断制御の検証を区別する。

今回の完了は、このホスト・候補・固定入力での結果であり、他の候補やホスト、実アプリ全体を保証しない。SIGKILLで入口自体が終了した場合の回収と、長時間安定性も未確認である。通常check・独立評価・同じheadのCIは別途確認する。

生ログ、モデル、個人pathを含む環境記録はホスト側で保管する。途中の成果物やbaselineの書換えで不一致を消さない。

## assessments: 軽量laneの中断修正

2026-10-11、search-guard-host-returnの評価対象
`fcb374158553d8e1942adf5f1aad6f5fc16ca9630317959194eff3bcc3c795bd`のR1-1を修正した。
軽量laneの通常の`subprocess.run`はunittestだけを停止し、そのテストが起動した検証用入口と
別groupのworkerを残した。モデル工程だけの中断テストではこの外側の取消しを検出できなかった。
現在は軽量runnerがSIGTERMをKeyboardInterruptへ変換し、共有runnerが後片付けを待つ。
テスト側も検証用入口を即座にkillせず、SIGTERMによる所有groupの回収を待つ。
10秒の待機は後片付けの猶予で、全体checkの実行上限変更ではない。

既存の実process中断テストへ軽量laneの階層を統合した。推論・Cargoと候補準備は合成だが、
外側のcandidate.main、light分岐、unittest runner、別sessionの検証用入口、共有runnerと
書込みworkerは実processで動く。外側PIDへ実SIGTERMを送り、未完了・完了記録なし、
runner・検証用入口・workerの不在と書込み停止を確認した。
軽量分岐だけを旧subprocess.runへ戻したメモリ上の対照では、同じ子孫不在のassertionが失敗し、
数値・検索の中断条件は成功した。再現processは後片付けした。

Python 3.15.0で、修正後の軽量runner経由の制御7件は3.323秒、既存比較器7件は0.011秒で成功した。
制御テストは、既存比較器では検出できない取消し後の書込み・資源消費と後続検証への干渉を防ぐ。
新しい独立テストを増やさず既存の中断テストへ条件を統合し、候補照合・dirty lock・旧fixture拒否等も維持した。
削除による検出条件の喪失はない。追加階層のprocess起動・同期・後片付けには費用があるが、
実際に見逃した中断経路を守るため維持する。旧ホストのPython 3.12.13による2.155秒とは
Python版と実行範囲が異なるため、速度や保守費用の改善量は算出しない。長期の不安定さも未測定である。

### 過去の指摘と証拠の適用性

unit-host-returnの停止記録・生応答・評価履歴、search-guard-host-evidenceの34記録と
その通し測定の完了成果物を読み直した。停止記録に列挙された25記録、元runを参照する14記録、
追加証拠34記録のhashは一致した。passedの申告を受入として引き継いでいない。
今回の変更前の成果物はsearch-guard評価の全ファイルに一致し、今回の実装修正は
candidate.py、processes.py、test_candidate_quality.pyに限る。手順と本要約も今回更新した。

unit-host-returnの旧R1-1は新規targetと実fixture参照検査を現在のコードへ照合し、
新しい隔離コピーを読んだ両実測ログを確認した。旧R1-2は現在の数値工程のgroup回収と
今回の実SIGTERM条件で再評価した。旧R1-3はidentity/prepareが取得済みsource mapを使い、
工程間の再検査を維持することを確認した。旧R1-4は未使用出力先と固有の拒否理由、
旧R1-5はcommit blob由来のlock期待値を確認し、今回の制御7件でも成功した。
後続の検索工程残留も、現在の共有runner呼出しと今回の実SIGTERM条件で再評価した。
これらの修正条件は維持されている。新しい軽量laneのR1-1は上記の修正・対照検証によって評価する。
各IDは別runの来歴であり、新しい評価の更新IDや受入状態を自動生成する根拠にはしない。

両モデルの生JSONを現行比較器で再集計し、保存comparisonと値が一致した。
固定件数と診断も満たした。検索のglobal metricsと既存baselineの差分も再計算し、
上記の値に一致した。これは保存結果の整合確認であり、新しい推論ではない。
通し測定は同じ製品候補に対応し、今回も製品source、入力、モデル、wrapper、shape、
数値計算・検索設定・基準は変更していない。共有runnerの既定のモデル工程回収も変更していない。
旧実測は変更のない品質経路へ適用できるが、今回の軽量取消しや現行版全体の実測成功を証明しない。
今回の局所検証も固定Python参照環境の全17件、実Cargo・実モデルの中断や実アプリ全体を検証しない。

[report.md](report.md)は開始commit `266fc41ff6bdf7ca5110e55a7315f8cbb5022cc6`の
指定blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`と一致し、旧観測・未採用仮説は保持した。
[CONTRIBUTING.md](../../../CONTRIBUTING.md)の通常checkと実モデルを区別する方針、
[README](README.md#候補commitの継続検証issue-366)の候補検証条件、
[accepted ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)の所有権を適用した。
Issue指定mainから開始版への参照環境の変更と[validation-353.md](validation-353.md)の
旧実測は、今回の新しいホスト観測と区別する。着手時の遠隔mainの最新性は引き続き未確認である。

## assessments: 照合済み入力hashの記録

2026-10-11、search-guard-host-returnの`verification/review-2.json`にある
評価対象`0706cf1f9bbe9f677a3dc9d21654d2424805ed840a49defc613820e573afbddc`のR2-1を修正した。
前の修正はidentity内のlockとprepare内の候補入力の重複読取りを除いたが、
prepareで照合した直後のcandidate.json生成には、入口側の同じ入力の再読取りが残っていた。
記録式だけを、prepareから返されたsnapshotの入力hashを使う式へ変更した。
prepareの入口側入力との照合、モデル工程の入力取得、工程前後のsource検査、成果物hash検査は維持する。
新しい引数や工程間キャッシュは追加していない。

変更前後のexecute→prepareをメモリ上で比較した。候補準備のGit操作、identity、
工程実行と記録書込みを置換し、入口側inputs.jsonへの直接hash呼出しが2回から1回となり、
candidate.jsonの記録値が照合済みsnapshotと一致することを確認した。
これは呼出しと記録値の診断であり、実推論や速度改善の測定ではない。
Python 3.15.0で既存の候補制御7件は3.303秒で成功した。

旧revでの誤実行、dirty lockの混入・誤拒否、旧fixture参照、成果物改変、
中断後の子孫残留・書込みを防ぐ既存テストを維持した。
比較器に加えて、候補との束縛と実processの終了経路を確認する保証がある。
今回の単純な記録式に対し、hash呼出し回数へ依存する恒久テストは追加しない。
局所診断で十分であり、内部構造へ結び付いたテストの保守費用を増やす価値は小さい。
既存テストの追加・削除はなく、失う検出条件もない。今回と過去の実行は同条件の費用比較ではなく、
時間削減量、長期の不安定さ・保守費用は未測定である。

参照された停止記録、生応答、評価履歴と追加証拠を再読し、記録hashを照合した。
修正前はreview-2の458ファイルと一致し、今回の差分はcandidate.pyの記録式と本要約だけである。
unit-host-returnの旧R1-1は新規targetと両ログの実fixture参照を再確認した。
旧R1-2と後続の検索工程残留、軽量laneのR1-1は、現行の共有runnerと今回の実SIGTERMテストで再評価した。
旧R1-3のsource map共有、旧R1-4の固有のSHA拒否理由、旧R1-5のcommit blob由来のlock期待値も、
現在のコードと今回の制御テストへ照合した。これらの修正条件は維持されている。
R2-1は上記の変更前後の診断で評価し、過去の受入やpassedの申告を引き継がない。

追加証拠の数値・検索の完了成果物hashを再検査し、両モデルの生JSONを現行比較器で再集計した。
保存comparisonとの一致、固定件数と診断、検索のglobal・分類別metrics差分の一致を確認した。
同じ製品候補・固定入力への証拠であり、推論条件と品質基準は今回も変わっていない。
旧測定は変更のない品質経路には適用できるが、現行入口全体の実測成功の証拠には不足する。
開始commitのreport.mdは指定blobと一致する。旧観測と未採用仮説、上記の版境界と限界を保持し、
本更新も通常checkと実モデル検証を区別する文書方針・accepted ADR-0006の範囲に従う。

日本語確認は固定版`9a78a42964096da509b8f3e011f0085a5f080151`の手動チェックリストで行い、
数値・条件・参照と段落の主旨を再照合した。lintは既定cacheへの書込み権限がなく失敗し、
一時cacheでの再試行もofflineではSudachi依存が見つからず実行できなかった。
機械lintを含むクイック手順は未完了であり、指摘0件として扱わない。

## assessments: 検索集計の共通条件を一度照合する

2026-10-11、search-guard-host-returnの`verification/review-3.json`にある
評価対象`12f367da15dcd2a5f2fc00d99829818840c01aa6df8350e5a8a76cfa76674bad`のR3-1を修正した。
同じbaseline・measuredの分類別集計で、包絡辞書を16個複製し、共通6条件を9組検査していた。
数値列の比較を内部関数へ移し、globalの`metrics_delta`で共通条件を一度照合してから、
8分類にも同じ数値比較を使う。外部の`metrics_delta`は共通条件を引き続き拒否する。
分類集合、各列の非空・指標名・cutoff・有限性・CIの検査を維持し、検証済みflagや工程間キャッシュは追加しない。
複数の不正が同時にある場合は、分類別数値よりglobal数値の不正を先に報告する。
いずれも成果物の書込み前に停止し、通過条件は変わらない。

修正前後の実際の集計境界をASTから取り出し、保存された同じbaseline・実測JSONをメモリ上で比較した。
共通条件の検査は54回から6回となり、globalと8分類の差分・CI・順序は保存結果と一致した。
共通6条件の不一致、分類欠落、globalと各分類の空列・指標名・cutoff・逆転CI、
3数値欄のNaN・正負Infinityを個別に与えた124例は、前後とも対応する理由で拒否した。
これは集計の診断であり、新規推論や時間削減量の測定ではない。
局所診断はホストの一時領域に`rurico-366-r3-diagnostic.py`と同名のJSONとして保存した。

既存の比較器テストへ共通6条件と数値列の拒否例を統合した。
今回移した比較本体で、別pipelineや空・不正な数値を成功扱いする退行を防ぐ。
既存の正常差分・CI確認に加え、両側の不正値を拒否する保証を持つ。
Python 3.15.0で比較器7件は0.011秒で成功した。
候補制御の既存7件も、工程接続への拒否例追加前は3.304秒、追加後は3.314秒で成功した。
既存の模擬Cargo接続テストにも、正常なglobal・分類別結果と、分類欠落・空列・非有限値の拒否を統合した。
集計経路で不正な分類を無視して完了を保存する退行を防ぎ、保存抑止を既存の関数テストへ加える。
最初のNaN注入はテスト用writerで拒否され、比較器まで届かなかったため、consumerの生JSONとして渡すよう修正した。
模擬工程を3回追加する費用で、実Cargo・推論や新しいprocess・同期を増やさない。
呼出し回数に依存する恒久テストも作らない。
既存テストの削除はなく、失う検出条件はない。所要は今回の検証費用であり、
同条件の修正前後の速度比較、長期の不安定さ・保守費用は未測定である。

停止記録、生応答、評価履歴と追加証拠を再読し、参照された98記録のhashを照合した。
修正前の成果物はreview-3の458ファイルと一致する。
今回の変更は`search_quality.py`、既存の`test_compare.py`と`scripts/test_candidate_quality.py`、本要約に限る。
開始commit `266fc41ff6bdf7ca5110e55a7315f8cbb5022cc6`と指定reportのblob
`7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`を維持する。
[report.md](report.md)は現在もそのblobと一致し、旧観測・未採用仮説を追加の許可にしていない。
[CONTRIBUTING](../../../CONTRIBUTING.md)と[現行手順](README.md)は軽量検証と実測を区別する規則、
[validation-353.md](validation-353.md)は旧参照環境の観測、
[accepted ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)はamiciの評価所有権の根拠として適用した。

過去の指摘も現在のコードと今回の証拠で再評価した。
unit-host-returnの旧R1-1は新規targetの伝播と誤fixture拒否を制御テストで確認し、
実測のcapture/verify両ログも新しい隔離fixtureと一致した。
旧R1-2、後続の検索工程残留、search-guardのR1-1は、現行の共有runnerと
実SIGTERMの制御テストで数値・検索・軽量laneの子孫不在と書込み停止を再確認した。
旧R1-3はidentityのlock・prepareの入力hashが同じsource mapを使う現行コード、
旧R1-4とR1-5は固有のSHA拒否理由とdirty cloneからcommit blobのlockを隔離する制御テストで確認した。
R2-1はprepare直後の記録が照合済みsnapshotを使い、新しい入力照合や工程前後の検査を保つことを再確認した。
今回はその記録式を変更していない。これらの修正条件は維持され、R3-1は上記の前後診断で解消を確認した。
過去の受入や証拠のpassed申告を引き継いだ判定ではない。

製品候補の452 source hashとlockをcommitの内容へ照合し、固定consumerの依存2ファイル以外の
93ファイルが原manifestと一致することも確認した。実測の解決済みpackage・lockと、
後続の実Cargo metadataは同じ候補へ束縛されていた。
通し測定時のmetadata stdout自体は保存されておらず、後続の実metadata証拠と区別する。
数値・検索の完了成果物hashを再検査し、両モデルの生JSONを現行比較器で再集計した。
embedding 23 batch・27行、reranker 12 batch・16行の診断と保存結果が一致した。
検索のglobal・分類別結果も一致したが、再集計を新しい推論成功とはしない。
測定時のsource mapとの差は入口制御・出力保持、軽量取消し、入力hash記録、今回の集計整理とテスト・文書に限る。
製品候補、数値推論、公式参照・入力・環境・検索評価の品質条件は変わっていない。
品質証拠はその範囲へ適用できる。「現行入口全体を実モデルで再測定済み」という主張には
証拠が不足するため、その主張はneeds_changesと評価する。今回の修正の受入は独立評価へ戻し、版境界と限界を保持する。

日本語確認は固定版`9a78a42964096da509b8f3e011f0085a5f080151`の手動チェックリストで
数値・条件・参照と段落の主旨を照合した。一時cacheを指定したofflineのlintは、Sudachi依存がcacheに存在せず終了1となった。
機械lintを含むクイック手順は未完了であり、指摘0件として扱わない。

## handoff

R3-1は同じ実測入力による結果一致と拒否条件、既存の比較器・制御テストで確認した。
全体checkと本更新を含む独立評価はホストで更新する。今回の独立受入はまだ行っていない。
設定済み`bash scripts/check.sh`は候補制御テストを実行し、比較器テストは別のPython参照CI stepにも含まれる。
比較器の影響するテストは今回局所実行済みで、check/capture契約は変更しない。UI媒体は不要のままである。
今回の変更は同じ集計境界内の整理に限るため、この修正のための全モデル再測定を追加の完了条件にはしない。
現行入口全体の通し再測定が未実施であることは、上記の品質証拠の適用範囲と分けて評価する。

実Cargo・実モデル推論中の取消しを後続で検証する場合は、合成process検証の成功を代用せず、
実工程で子孫不在と書込み停止を確認する。入口自体のSIGKILL、応答しない後片付け、
任意に分離したsession、他候補・他ホスト、長時間安定性は保証しない。
過去runと実測原本、fixture、baseline、閾値は保持し、commit・push・公開は行わない。
