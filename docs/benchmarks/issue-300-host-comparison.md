# Issue #300: ホスト上の比較

## 環境

Apple M3、24 GiB、macOS 27.0 (26A428)、Rust 1.98.1 (48a229cea)、Xcode 27.0 (27A266a)、Metal toolchain asset 27.1.266.1 / metalfe-32023.921.6。Cargo.lockのmlx-rs 0.32.0 / mlx-sys 0.6.0を使用した。計測用プログラムも既存のCargo.lockを流用し、debug buildで実行した。

## 変更前の測定

2026-09-21、開始版 `1a53b0dc1859b0a432b637faf79b3287304b8875` で実行した。
[公開APIを使う計測プログラム](issue-300-host-probe.rs)と[全スコア・代表embedding・各試行のtime出力](issue-300-before-output.json)を保存する。
同じプログラムを変更後へ向け、同じ入力・モデル・依存・ビルド設定で比較する。
この節は旧版の観測であり、変更後の検証結果ではない。

モデルは公式embed `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` とreranker `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`、ともに310m/F32。
入力はソース内の公開可能な日本語query1件とdocument4件。各モデルを新しいプロセスでseed=42, 7, 42の順に3回loadした。
各rerankerプロセスでは同じbatchを2回実行し、同一インスタンス内の出力は一致した。
embeddingはseed違い・再ロードともquery/batchの全要素が一致したため、完全な配列は最初の試行だけ保存した。

旧rerankerのseed=42と7のscore最大絶対差は0.0057475268840789795。
両seedで順位はdocument index `[2, 0, 1, 3]` だった。この4件で順位が同じでも、一般の順位や検索品質の維持は主張しない。
旧版への一致はreranker修正後の合格条件ではない。今回の公開API測定はsigmoid後のscoreであり、生logitの記録ではない。

| 経路 | seed | constructor load (ms) | load直後のMLX peak (bytes) | プロセス全体最大RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| embed | 42 | 495.423125 | 1,258,611,104 | 1,306,558,464 |
| embed | 7 | 152.073041 | 1,258,611,100 | 1,569,128,448 |
| embed | 42 | 172.584250 | 1,258,611,116 | 1,568,309,248 |
| reranker | 42 | 468.821667 | 1,260,982,360 | 1,570,963,456 |
| reranker | 7 | 155.549916 | 1,260,982,388 | 1,570,930,688 |
| reranker | 42 | 224.610625 | 1,260,982,388 | 1,571,028,992 |

各試行は`/usr/bin/time -l <probe> <embed|reranker> <seed>`で実行した。
constructor時間はcached_artifactsの検証後からnewの完了まで。MLX peakはnew直前にresetして取得したallocatorの値で、RSSやファイルcacheとは別の指標である。
RSSは推論を含むプロセス全体の最大値であり、loadだけの最大RSSではない。
モデルはキャッシュ済みで、OSのファイルcacheをflushしていない。新規プロセスでもdisk cold loadとは扱わない。
最初の試行が遅く、その後にも変動があるため、この3回だけで性能改善や悪化を断定しない。

## 変更後の検証

2026-09-21、同じ公開API probeを変更後にビルドして6試行を実行した。
[変更後の出力とtimeログ](issue-300-after-output.json)、[ビルド時のソースSHA-256](issue-300-measured-source.json)を保存する。
ソースmanifestはcommit前の測定対象を識別するための記録であり、最終commitの代わりではない。計測後に変更したRustソースは差し替え検出のテストファイルのみで、製品コードとCargo.lockは一致する。

queryとdocument4件のembeddingは、3試行とも変更前と全要素一致した（最大絶対差0）。
完全な配列を保存したのは最初の試行だけで、後続2試行は一致の実行記録である。保存配列から全試行の差を独立に再計算できるわけではない。
rerankerは3試行および同一インスタンスの再実行でscoreが完全一致し、seedによる揺れが消えた。
変更後scoreは `[0.0074319318, 0.0035561298, 0.0198229998, 0.0008567868]`、順位は `[2, 0, 1, 3]`。
旧seed=42とのscore最大絶対差は0.0088901892。今回の4件では順位は変わらなかった。

| 経路 | seed | constructor load (ms) | load直後のMLX peak (bytes) | プロセス全体最大RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| embed | 42 | 481.188625 | 1,258,448,284 | 1,568,571,392 |
| embed | 7 | 156.611375 | 1,258,448,284 | 1,568,260,096 |
| embed | 42 | 148.019459 | 1,258,448,284 | 1,568,309,248 |
| reranker | 42 | 448.904250 | 1,260,813,740 | 1,569,964,032 |
| reranker | 7 | 152.118542 | 1,260,813,740 | 1,569,964,032 |
| reranker | 42 | 150.871500 | 1,260,813,740 | 1,570,275,328 |

比較条件と指標の範囲は変更前と同じ。初回と後続の差が大きいため、性能改善の根拠とはしない。
load直後のMLX peakはembedで約159 KiB、rerankerで約165 KiB小さく、checkpoint全体の追加保持は観測しなかった。
CPUで読むのはheaderのみで、約1.26 GBのtensor payloadはMLXで一度だけ読み込む。

### rerankerのlogitとclassifier bias

[内部テストの全出力](issue-300-reranker-runtime.txt)に固定revision・入力・許容差・各試行を保存した。
`official_reranker_reload_contract`は1件成功し、checked、同じ修正後構成のunchecked、旧head再構成の各3回を比較した。
checkedのlogitは3試行とも `[-4.89451, -5.63552, -3.9008904, -7.0614643]`。
uncheckedとのlogit・score・順位は事前定義した基準を満たし、classifier.biasに0.25を足すとlogitにも0.25が加わった。
旧head再構成は変更前の公開API実測scoreと照合してから比較し、旧seed=42とのlogit最大絶対差は0.4161296だった。
これは公式Transformers実装全体との一致や一般の検索品質を評価した結果ではなく、#307の対象範囲は維持する。

検証ありのload時間は90.704 / 94.010 / 103.862 ms、検証なしは129.579 / 92.615 / 113.661 ms。
両経路のload区間MLX peakは1,260,813,740 bytesで同じだった。
3回の交互実行の観測値であり、差を検証処理そのものの速度効果と断定しない。

### embeddingのload比較

[公式embeddingの内部テスト出力](issue-300-embedding-runtime.txt)の1件が成功した。
検証ありのload時間は76.847 / 75.344 / 74.604 ms、検証なしは92.524 / 77.461 / 77.019 ms。
どちらもload区間MLX peakは1,258,448,284 bytesで同じだった。
初回と後続を含む3回の交互実行であり、ここでも速度改善は断定しない。

### 既存embedding基準と実probe

`cargo run --locked --features smoke --bin mlx_smoke -- verify-fixture`は既存fixtureを変更せず成功した。
[W1/W2/W3の出力](issue-300-fixture.txt)は、cosine最小値が表示上すべて1.000000、
最大絶対差が順に9.537e-7、4.172e-7、5.960e-7だった。
既存の閾値cosine>=0.99999かつmax_abs_diff<=1e-5を全workloadで満たした。

2026-09-21の修正時に、[比較処理](../../src/bin/mlx_smoke.rs)、[workload定義](../../src/embed/workloads.rs)、
[数値基準](../../src/embed/fixtures.rs)をログと照合した。W1は長文2件の複数chunk、W2は短文100件、
W3は長短混在10件を検証するため、公開API probeの短いqueryと4文書だけでは見逃すembeddingの退行を検出する。
既存の検証を再利用し、テスト・入力・閾値の変更やfixtureの再生成は行っていない。
`tests/fixtures/phase2_baseline/w1.bin`・`w2.bin`・`w3.bin`は開始版`1a53b0dc1859b0a432b637faf79b3287304b8875`とバイト単位で一致する。
比較処理・workload定義・数値基準・モデル選択を含む製品コードは、上記の測定ソースmanifestと一致する。
[モデル選択](../../src/embed.rs)と[キャッシュ取得](../../src/model_io/hf_backend.rs)はmodel/tokenizerに同じ固定revision
`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`を使う。

前回レビュー対象`c21bd412288be9b2e07c746ce7a6009e5b1c96badf6f726c55c81ced089a6f78`には、
このW1/W2/W3の完了ログがなく、R1-1は実行証拠不足として残っていた。今回の結果を含む成果物で独立評価を更新する。
このログは各workloadの集計値と比較処理の成功記録であり、実行時の全出力配列は含まない。
表示値を厳密なcosine=1と解釈せず、既存閾値を満たした結果として扱う。任意の入力や他のモデルrevisionでの一致は未検証である。

実probeの[統合テスト2件](issue-300-probe-runtime.txt)も成功した。
embedとrerankerの各子プロセスが公式モデルをloadして正常終了することを確認した。
不正checkpointの子プロセス拒否と診断理由は通常の`tests/weight_probe.rs`で別途確認している。

実行したコマンドはCONTRIBUTINGの受入手順に対応し、MLX runtimeはsandbox外のホストで実行した。
rerankerとembeddingの内部テストには`cargo nextest run --locked --lib --features test-support,test-mlx --run-ignored=ignored-only --test-threads=1 --success-output=immediate <test-name>`を使用した。
実probeには`cargo nextest run --locked --features smoke --test mlx_smoke --run-ignored=ignored-only --test-threads=1 --success-output=immediate -E 'test(probe_embed_smoke_binary) | test(probe_reranker_smoke_binary)'`を使用した。



## 通常CI経路で検出した互換性

ホストで通常CI相当のcoverageを測定したところ、正常load経路が未実行だった。
小さな実MLXモデルを保存し、seedを変えて再生成した後に全parameterが復元される`tests/weight_load.rs`を追加した。
このテストは最初、MLXのwriterが生成する`__metadata__: null`を新validatorが誤拒否する問題を検出した。
[Safetensors v0.6.2のreader](https://github.com/huggingface/safetensors/blob/v0.6.2/safetensors/src/tensor.rs#L471-L478)と同じoptional metadataとして扱い、文字列mapまたはnullを受理するよう修正した。
欠損tensor・shape・dtype・未知キーの検査は維持する。
さらに、preflight後・MLXがファイルを開く前に有効な別checkpointへ差し替える回帰テストを追加した。
shape、dtype、未知キー、欠損キーの4条件で実際のロード結果を拒否する。時刻やスレッド競争に依存せず、検査後の差し替えを再現する。
CI相当の`cargo llvm-cov --locked --workspace --features test-support,test-mlx`と既存の除外設定で計測し、
`diff-cover --compare-branch=origin/main --include-untracked --fail-under=95`は182/184行（98.91%）で成功した。
coverageの閾値や除外設定は変更していない。通常テストは380件成功、29件ignored。
[差分coverageの出力](issue-300-diff-coverage.txt)を保存する。
