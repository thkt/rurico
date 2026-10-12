# Issue #309 長さ別バッチの比較手順

[Issue #309](https://github.com/thkt/rurico/issues/309)の試作・診断基準・根拠と未確認事項は
[report.md](report.md)を参照する。公開APIは従来の全pair最大長方式を使う。
試作は既存のlibtestと`smoke`内で動き、モデル・consumer・baselineを変更しない。
UI媒体は不要。以下の数値記録は、設定の`capture: null`を変更するものではない。

## ホストで比較する

Apple Siliconと対応Xcode/Metal、固定reranker310mのcacheを用意する。
[標準check](../../../CONTRIBUTING.md#テスト)の成功と実モデル比較の成功は別に記録する。
setupは`cargo fetch --locked`、全体checkは`bash scripts/check.sh`のままである。

測定前に、同じ変更版のlibtestをビルドする。出力先はcheckoutと旧runの外の新規領域とする。

```sh
cargo test --locked --release --lib --features test-mlx,smoke --no-run \
  --message-format=json > /tmp/rurico-309-build.jsonl
# compiler-artifactのtarget.name=rurico、target.kind=["lib"]、profile.test=trueの
# executableを1つだけ取り出し、以下のbinary引数へ渡す。
python3 docs/research/issue-309/run.py /absolute/path/to/rurico-libtest \
  /tmp/rurico-309-context.json /tmp/rurico-309-evidence-001
```

context JSONには実際の`machine`、`ram_bytes`、`os`、`rust`、`cargo`、`xcode`、`metal`、
`build`（release、flags、ビルド並列数等）、`load_conditions`（測定期間、負荷停止範囲、
AC電源・熱状態の取得結果または未計測、監視頻度）を記す。
`model_files`には固定snapshot
`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`内の`model.safetensors`、`config.json`、
`tokenizer.json`の絶対pathを3つ指定する。実行入口は観測したcache pathとmanifestを照合し、HF snapshotのsymlinkも扱う。
モデル内容は[#307の保存manifest](../issue-307/results/numerical-rechecked/models.json)のSHA-256と比較する。
個人pathを含むcontext、生log、モデルとbinaryは公開資料へコピーしない。

他のGPU測定やビルドを、測定開始から終了まで止める。
[#306の隔離条件](../../../CONTRIBUTING.md#options付き推論の計測issue-306)に従い、
process名・CPU使用率の定期観測もホストで保存する。
このrunnerが持つ監視は所有groupのRSS・system pressure・swapであり、競合processを監視しない。
監視欠落や負荷重複を、負荷がなかった証拠へ読み替えない。

[run.py](run.py)はsame/mixed/boundary128/boundary512/boundary2048/limit/emptyの各条件について3組を実行する。
組ごとにreference/candidateの順を交互にし、各方式を新しいprocessで動かす。
各processはモデルを1回ロードし、最初の推論をcold、続く3回をwarmとして保存する。
境界条件はpairの実長127/128/129、511/512/513、2047/2048/2049を3件ずつの独立した比較にする。
9件を混ぜた旧測定は現行側8192×9のshapeでpressure停止した。失敗履歴を残し、小規模比較へ分けた。
この分割で9件混在や5000件の完了を保証しない。
limitではraw長8191/8192/8193の3ペアを用い、truncate後8191/8192/8192を確認する。
coldはロード後の初回推論であり、OSのfile cacheを消した起動ではない。
model loadはwallの外側、pair encodingはwallの内側である。

[既存#318 runner](../issue-318/run.py)のsnapshot・reclaimを再利用し、測定全体30分、
所有group RSS 8 GiB超、pressure正常値1以外、swap増分256 MiB超では停止する。
これは[#364の測定保護条件](../../benchmarks/issue-364/README.md)を使うもので、製品hard limitではない。
途中失敗はstdout/stderr、status、monitorを残し、同じ保存先へ自動再試行しない。
0件のlibtest選択、欠損した記録、出力数の過不足、版・model・入力の不一致も未完了とする。

共有readback境界の失敗時cleanupと公開APIの配線には、新しい同等試験を増やさず既存検証を使う。
比較のビルド・GPU実行とは分け、同じ変更版の以下の対象を単独で実行する。
cleanup観測には固定embedding310mも必要である。

```sh
cargo nextest run --locked --lib --features test-mlx --run-ignored=ignored-only \
  --test-threads=1 --success-output=immediate real_model_cleanup_memory
cargo nextest run --locked --lib --features test-mlx --run-ignored=ignored-only \
  --test-threads=1 --success-output=immediate cached_model_scores_singleton_batch_and_reranks_in_input_order
```

終了コードだけでなく対象テスト名・実行件数・成功と、生stdout/stderrを保存する。
5000ペアの既存ignoredテストはこの比較runnerから起動しない。過去の未完了を消す操作でもない。

## 記録から判断する

`context.json`は開始commit、tracked差分hash、未commitを含むRust source、Cargo.lock、
入力、比較器、runner、モデル、binaryのSHA-256と診断基準を持つ。正常終了時には内容を再照合する。
`*-*.json`は入力と4回の生観測、`*.status.json`は終了値・group不在とprocess RSS標本最大、
`monitor.json`は0.1秒を目安にしたRSS/pressure/swap観測である。
`summary.json`は同じ入力のlogit/score・順位の診断、forward数・padded/real token、
cold時間、processごとのwarm中央値、Metal peakの最小・中央値・最大をまとめる。
RSSはmodel load・境界入力準備・非計測の公開API照合を含むprocess範囲の標本最大であり、
callのMetal peakと同じ範囲でも瞬間最大でもない。

測定前に合意したlogit・scoreの最大絶対差は各`2.5e-5`以下。
両方の順位と同点の元index順は完全一致を要求する。
[#307比較器](../issue-307/compare.py)の差と近接対の診断を再利用し、#309の合意値で判定する。
製品の品質許容差や新baselineとして採用しない。

warm時間は独立process3組の中央値を比べる。
中央値の差が両方式の観測範囲幅の大きい方を超えた場合だけ、今回の速度利益または退行のsignalとする。
範囲が重なる場合やcoldだけの差、paddingだけの差は、安定した高速化の根拠にしない。
3組の記述統計であり、有意差・一般的な速度上限の証明ではない。
空入力の速度signalは判定しない。利益のある条件と退行する条件を両方報告する。

`complete.json`のcollection完了は受入でも採用でもない。
診断不一致でも生出力・summaryとcollection完了を残し、終了1にする。
ホスト担当AIは公開可能な結果要約と採用/見送りの推奨をreportへ反映し、変更文書も既存の独立評価へ渡す。
実測完了後も、公開APIへの自動採用、bucket変更、cache policy変更は行わない。

## 保存済み結果

分割版の[生記録・条件・集計](results/split/environment.json)と[報告](report.md)を参照する。
個人pathはmodelの内容hashへ置き換えている。次の再集計は保存値の診断であり、新しい推論ではない。

```sh
python3 docs/research/issue-309/summarize.py docs/research/issue-309/results/split
```
