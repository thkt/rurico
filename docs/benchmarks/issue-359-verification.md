# Issue #359: readbackと性能判定のホスト検証

本番readbackの実測接続と2種類の故障検出を確認した。短時間の実モデル統合テストは成功した。
全W1/W2/W3も推論・既存fixture比較・readback照合まで完了したが、性能gateは3条件で失敗した。
この変更の検証成功を、全workloadの性能目標達成や速度改善とは扱わない。

対象は[Issue #359](https://github.com/thkt/rurico/issues/359)。
開始commitは `4a029333c3b2a64c9483c410d4d722377ae93bea`。
[Issue #306の記録](issue-306-metrics.md#テストと既存報告の訂正)を引き継ぎ、閾値・fixture・Cargo.lockは変更していない。
今回は計測と判定の接続を確認し、bucket変更や新しいSLAは採用していない。

## 実測量と故障の拒否を確認した

2026-10-08（JST）に、Apple M3（24GiB）のsandbox外ホストで実行した。
ツールチェーンはRust/Cargo 1.99.0、Xcode 27.0（27A266a）、Metal 32023.921だった。
default 310mの固定revision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` と既存の公開合成入力を使った。

| 検証 | 結果 |
| --- | --- |
| smoke binaryの単体テスト | 20件成功。無効値・ゼロ時間・tier・比較条件の拒否を含む |
| モデル不要のCLI統合 | summarize/compareの2件成功 |
| MLX Arrayのhost slice記録 | 1件成功。実際の取得量が `[3, 3, 6]` |
| ignoredの短時間実モデル統合 | 1件成功。readbackと空入力を確認 |
| releaseの短時間raw record | 推論33件・集計8件・model load 1件。計測対象のreadback 36回・36,864要素が期待値と一致 |
| releaseの全W1/W2/W3 | 推論25件。計測対象461 callのreadback 464回・694,272要素が期待値と一致 |
| readbackを1回追加した独立変異 | 終了101。観測 `[2304, 2304]`、期待 `[2304]` の回数不一致で拒否 |
| pooled要素数を倍にした独立変異 | 終了101。`BufferShapeMismatch { expected: 2304, actual: 4608 }` で拒否 |
| 変異を戻した同じコピー | 短時間モードが終了0 |

raw recordの `calls[].readback_elements` を、各forwardの `batch_size × 768` の列と独立に照合した。
要素数は各計測callの合計で、同じ文書を反復した取得も含む。文書数や異なる文書の総量ではない。
2変異は一時worktreeで別々に適用・復元し、元の実装sourceは変更していない。

ホストビルドで見つかった可変借用競合は、readbackが記録fieldだけを借りる形に修正した。
比率のテストでは浮動小数点の丸め誤差を考慮し、MLX runtime guardの呼出しも既存の署名へ合わせた。
これらの修正後に上記の実モデル測定を行った。製品の性能閾値は変えていない。

## 全workloadの性能gateは未達だった

`measure-baseline` は全推論後、次の既存primary違反で終了101となった。

| 条件 | 観測 | 既存閾値 |
| --- | --- | --- |
| W2 padding | 6.7724867 | ≤ 1.2 |
| W3 padding | 1.6299626 | ≤ 1.2 |
| forward時間と規模のR² | 0.9206259114 | ≥ 0.95 |

#306でも同じ種類の3条件が未達だった。過去の時間とはbuild・実行条件が一致しないため、版間の速度比較には使わない。
今回もignoredの全性能smokeの成功は確認できていない。
全workloadのbinaryを直接実行してこの失敗を確認したため、同じ推論を繰り返す統合テストは再実行していない。
W1/W3の速度は既存policyどおりprimaryの保証対象外で、R²も速度保証ではない。

検証中、この作業のビルドとGPU測定を重ねず、2秒間隔でprocess名とCPU使用率を記録した。
他アプリのGPU利用や短い負荷はこの観測だけでは証明できない。
本記録の時間は合否判定の観測であり、隔離された条件での速度改善の根拠にはしない。
性能目標の達成判断は、対象の最適化Issueで条件を揃えて測定する必要がある。

## 対象版と再実行方法

実行方法と変異の入れ方は[性能判定と基準rev比較](../../CONTRIBUTING.md#性能判定と基準rev比較issue-359)に記載した。
固定入口 `bash scripts/build-smoke.sh` でrelease binaryを作り、
`measure-overhead`、`measure-baseline` と追加テストを順に実行した。
モデル不要の `compare-records` は合成記録で、方式比が同じまま両方式が100倍遅くなるケースと条件欠落を確認した。
異なる実装revを同条件で再測定した比較は今回行っていない。

測定前後で実装・lockfile・fixture・release binaryのhash一致を確認した。
測定後に本記録と手順からのリンクを追加した。
build情報の空白分割は、clippyの指摘に合わせて同じ判定のmethod参照へ置き換えた。
ホスト検証からの復帰後、標準checkは384件のテストと3件のdoc testを通過したが、
clippyが `comparison.rs` の `std::process::exit` とMLX Arrayテストのruntime guard呼出しを
既存の `absolute_paths` 規則違反として拒否した。
両呼出しをimport経由へ修正した。関数・引数・終了コードは変わらず、テストの期待値も変更していない。
推論・readback・性能判定の処理は測定時と同じである。
修正後の `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`、
`cargo fmt -- --check` と差分形式検査は成功した。これは標準check全体の再実行結果ではない。
標準check・独立評価・同じheadのCIは後続の実装手順で確認し、PRの検証欄に結果を残す。

測定時の識別値は次のとおり。sourceの版と未コミット差分を含むbinaryを識別し、最終PR commitと混同しない。

```text
start_commit: 4a029333c3b2a64c9483c410d4d722377ae93bea
tracked_diff_sha256: a1840fb3512a361835e33d04c1b435474eecf1f3666ae1fdfa367e484db45613
untracked_sha256: d0c2ea8e9b83533021c763fd23756e340641be83b9aa4ec6e834f377a03ed064
lockfile_sha256: 743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30
release_binary_sha256: ad61cf3c01349c01420b6ab72162f46fa08cc773f0f373d721863ff26a726fdc
model_sha256: 8229a6c3bbb16aa1a563eb62df60855af75b5e6e9c586f2ae571d3ab30df2804
config_sha256: 0f4eee1ac5634e11b095441246c59adc54818872e18478622338c61bc67847f8
tokenizer_sha256: 0a94ac9a0a02c067bdef25b72ae9f4ee33f48f552e55988d444f6d25eeb1d062
```

## 独立評価後の修正と証拠の適用範囲

2026-10-08の評価対象 `7997867e054d4e2ba2f8cf35594cc43d45d8c079ae0f0508be0cea04c4701ced`
に対するR1-1〜R1-4を修正した。両版ともbatch記録が欠ける比較を成功扱いしていたため、
各warm groupで相手方式の存在を両方向から確認する。
欠損テストは有効な記録から作り直し、batch欠損とsequential欠損の診断を確認する。
CLIでも両ケースが終了1・stdout空となることを確認した。

baselineのreadback再検査は削除し、`Trial::run` の各計測callの検査を共用する。
wall・forward時間・real token数の有効性検査とreadback表示は維持した。
版間遅延と方式効率の同じwall分布も、比較呼出し内で一度だけ集計する。
工程をまたぐcacheは使わない。重複処理の除去による実行時間の改善幅は未測定である。
現行のbuild条件記録と旧recordの`build_flags=null`の違いは、
[record手順](../../CONTRIBUTING.md#mlx_smoke-smoke-テスト)へ反映した。

修正後のモデル不要のsmoke単体21件と比較CLI統合1件が成功した。
単体のテスト本体は表示上0.00秒、CLI統合は0.93秒で、build時間を含まない。
smoke binary対象のclippy、fmt、差分形式検査も成功した。全体checkの結果ではない。
新しい欠損検査は純粋な合成入力を使い、実モデル測定を増やしていない。
別の欠損値で先に失敗する旧assertを置換したため、必要な検出条件は失っていない。
実機で未達診断が必ず出ることは引き続き要求せず、診断0件の成功とtier分類は合成テストで守る。

上記のホスト測定は修正前の版の履歴として保持する。
追加証拠58ログのhashとrawを再照合し、短時間36回・36,864要素、全workload464回・694,272要素を確認した。
今回、本番readback・shape検査・cleanupとcallごとの共通検証は変更していないため、
過去の実接続・追加取得・要素倍増の拒否の証拠はその経路に適用できる。
baseline集計側と比較処理は変更したので、過去の実行を現在版の全体成功とは扱わない。
モデル再実行と全体checkは今回行っていない。全baselineの既存3条件の未達、
実rev間の同条件比較と背景GPU負荷の未確認は残る。
修正後の独立評価と設定済みcheckはホストで更新する。


## wall集計の共用と再確認

評価対象 `6f72760b652c624ecadfcd2b72d70d558ee82b6ea48e987b851c35c92b461b4b` のR2-1では、
summary出力後にbaselineが同じwarm wall列を再集計していた。
集計済みの型付きsummaryを同一実行の返却結果へ含め、JSON出力とbaseline判定・表示から参照するよう修正した。
context・入力hash・workload・options・method・measuredが一致するsummaryを使い、
sequence対応とwarm-up除外を維持する。JSON再解析や工程をまたぐcacheは使わない。
wall・forward時間・real token数の有効性検査は維持し、入力値が異なるforward_eval分布も引き続き集計する。

修正後のsmoke単体21件、モデル不要のCLI統合2件、対象のclippyが成功した。
単体は表示上0.00秒、CLI統合は0.15秒だった。いずれもbuild時間を含まない。
既存のsummaryテストへ、返却する集計値のJSON形式と各条件の参照を追加した。
異なるoptions・method・計測方式の値を判定へ混入する回帰を防ぎ、warm-up除外、sub-ms精度、原recordの不変性も確認する。
既存のCLI検証は出力形式とdispatchの保証として維持し、新しい実モデルテストは追加していない。
必要な検出条件は失っていない。修正前後の実行時間、不安定さ、保守費用の改善幅は未測定である。

過去の全baselineと短時間rawを独立に再集計し、現在のbinaryによるsummary再生成ともJSON値で照合した。
全baselineの6組、短時間の8組すべてで、サンプル数・min・median・maxとsequence対応が一致した。
実readback量もそれぞれ464回・694,272要素、36回・36,864要素で変わらない。
これは過去rawに対する現在の集計処理の検証であり、現在版の実モデル測定ではない。
全baselineの既存3条件未達、成功分岐の実機未確認、実rev間の同条件比較と背景GPU負荷の未確認は残る。
設定済みの全体checkと独立評価は、この修正後の成果物でホストが更新する。

## 反復試行のcontext検査を共用した

評価対象 `1c357d5885a7686ad6543ce455eb71ed0fb13957b6c3a16fca7c1b422fd9ad67` のR3-1では、
同じwarm group内の各試行で、同一contextの20項目を再検査していた。
先頭記録で欠損を検査し、後続記録はcontext全体の一致を確認してその結果を共用する。
異なるcontextは混在として拒否し、新しいgroupでは欠損を検査する。
wall・入力hash・callsは引き続き各recordで確認する。
版間の条件比較、相手方式の対応確認、拒否時のstdout空も維持した。共用は比較呼出し内に限る。

修正後のsmoke単体22件とモデル不要のCLI統合2件、対象のclippyが成功した。
テスト本体の時間は単体が表示上0.00秒、CLI統合が0.91秒で、build時間は含まない。
追加した合成検証は、同じgroupの反復が正しく集計される対照と、後続記録のcontext混在・欠損・ゼロ時間を確認する。
先頭記録の条件拒否を守る既存検証に、検査結果を共用する後続記録の拒否条件を加えた。
必要な検出条件は失っていない。実モデルテストは追加せず、実行時間・不安定さ・保守費用の改善幅は未測定である。

過去rawのreadback量と6組・8組のsummaryを再集計し、上記の実測値との一致を確認した。
現在の比較入口へ全baselineのwarm記録を渡すと、6組の遅延と両版の効率6件が出力された。
その後続記録を個別に変更したcontext混在、build条件欠損、ゼロ時間、入力hash欠損、calls欠損は、
いずれも終了1・stdout空で拒否された。両版でのbatch欠損とsequential欠損も同様に拒否された。
これは過去rawの比較処理の検証であり、現在版のモデル再測定や実rev間の性能比較ではない。
本番readback・shape・cleanupは今回変更していないため、過去の実接続証拠はその経路に限定して使う。
baseline集計はR2-1修正後から同一で、既存summary検証が今回も成功した。
性能gateの既存3条件未達と実機成功分岐の未確認を保持する。
設定済みの全体checkと独立評価は修正後の成果物でホストが更新する。
