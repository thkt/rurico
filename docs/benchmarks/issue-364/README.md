# Issue #364の検証記録（5000ペアは未完了）

50ペアの前後比較と小規模の実モデル検証は完了した。5000ペアはメモリ圧迫の安全監視で中断したため、Issue #364の受入検証は未完了である。下書きPRで変更を共有し、マージ可能とは扱わない。

## 測定結果

2026年10月10日18:41:28–18:44:25 JST、Apple M3・24GiBで実行した。

| 対象 | 基準版 | 変更版 |
| --- | ---: | ---: |
| 50ペア p50 | 1.612105秒 | 1.606127秒 |
| 50ペア p95 | 1.777296秒 | 1.784018秒 |
| 小規模の実モデルテスト群 | 8件成功・58.31秒 | 1件成功・0.95秒 |
| 5000ペア | 実行せず | メモリ圧迫で中断・未完了 |

推論の高速化や性能非劣化は、この一組の順次比較では判断できない。小規模テスト群の短縮には、旧50ペア性能測定を独立benchmarkへ移した効果が含まれる。新旧でテスト群の構成が違うため、推論速度の改善として扱わない。

測定値の正本は[基準版JSON](baseline.json)と[変更版JSON](current.json)。30回のraw値、モデルrevision・資産SHA、ソース識別、toolchain、ビルド条件、warmup回数と比較比を保存している。p50は昇順の16番目、p95は29番目。比較比は変更版/基準版で、p50=0.996292、p95=1.003782である。

## 対象版・方法

基準はmainの`16e5ca97e4079917b7f16b48c7dc003ffdd77407`で、今回と同じbenchmarkファイルとCargoのbench登録だけを追加した。変更版は同commitに今回の差分を加えた未commitソース。測定後に検証記録と案内リンクを追加し、独立評価で指摘された重複helperテスト1件を削除した。固定budgetテストを指すdoc commentとADRの参照も修正した。実モデルテスト・benchmark・製品動作・Cargo.lockは測定時から変更していない。

両側とも`cargo bench --locked --bench reranker_latency --no-run`でビルドし、benchプロファイルのopt-level=3、LTO有効、codegen-units=1を使用した。Cargo並列数は2。同じF32のruri-v3-reranker-310m、revision `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`を使用し、モデル3ファイルとCargo.lockのSHAを照合した。

入力は50個の`("query", "doc")`、bucket=128。モデル生成後に3回warmupし、public `score_batch`のwall timeを30回測定した。各呼出しの件数50・有限値・[0,1]も確認した。`constructor_calls=1`と`score_batch_api_calls=33`はbenchmarkハーネスの実行数であり、内部ロード・GPU処理の独立した計測カウンターではない。

基準benchmark、変更benchmark、基準の小規模テスト、変更の小規模テスト、変更5000ペアの順に、ビルドを止めて単独実行した。0.1秒間隔を目安に、所有するプロセスグループのRSS、システムのメモリ圧迫、swap増分、既知の競合ビルド/GPUプロセスを監視した。測定枠は30分、RSS 8GiB超・pressure正常以外・swap 256MiB増加・期限到達では自身の測定を終了する条件とした。この条件は今回の測定保護用であり、製品の資源上限ではない。

小規模テストでは5000ペアを別実行へ分け、基準8件と変更1件を`--ignored --nocapture --test-threads=1`で実行した。変更版の通し検証は1つのモデルでsingleton・入力順・降順rerank・空入力・dispatchログを確認した。5000ペアには同じlibtestバイナリの対象名を`--exact --ignored --nocapture --test-threads=1`で指定した。手順は[CONTRIBUTING](../../../CONTRIBUTING.md#重複検証の整理とreranker遅延の計測issue-364)を参照する。

## 停止結果と限界

変更5000ペアの開始2.25秒後、`kern.memorystatus_vm_pressure_level`が1から2になったため、所有する測定プロセスグループだけにSIGTERMを送り、18:44:25 JSTに終了した（exit -15）。この対象の観測RSSピークは約1.46GiB、swap使用は0。RSSだけではGPU/共有メモリ全量を判断できず、原因は未特定である。基準版5000ペアは開始していない。

18:45:00 JSTの読み戻しでは測定プロセスの残存なし、pressure=1、swap使用0を確認した。入力資産・測定ソースは前後で一致した。8GiB上限の緩和、テスト削除、製品token budget変更は行っていない。前回の再起動前の一時ログは失われており、今回の成功証拠として使用していない。

電源はAC、測定前後の熱・性能警告は記録されていなかった。ただし実行順・背景負荷の全量・熱状態の厳密な同一性は保証できない。安全監視の負荷もwall timeに含まれる。検索品質や全入力の性能改善は今回の測定範囲外である。

測定前の`bash scripts/check.sh`はRust 410件、doctest 3件、Python 4件、clippy、format、visibilityを通過した。公開前の重複helperテスト削除後も同じcheckを再実行し、Rust 409件、doctest 3件、Python 4件、clippy、format、visibilityが成功した。標準checkでignoredの27件を実モデルの成功へ加算していない。独立評価と公開headのCIは別途確認する。

次は現行のbucket=128・一度に2000ペアを処理する経路のメモリ使用を切り分け、安全に5000ペアを検証できる条件を決める必要がある。未完了の検証をskipや閾値緩和で通過させない。

コマンド、stdout/stderr、監視時系列、停止後の読み戻し、ソース照合はホストに保全している。個人の絶対パスと生の実行ログは公開資料へ含めない。
