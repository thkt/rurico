# Issue #318: 上限付き逐次推論の調査

[報告](report.md)に、現行の保持量、未採用の試作、推奨する最小導入案と判断が必要な用途を示す。
要求・許可の正本は [Issue #318](https://github.com/thkt/rurico/issues/318) とその合意記録。
2026-10-12の修正前の8ケースと、修正後のcontrol/deadline/mixed・2変異・正常版復元と、embedding準備修正後のfairness/ownershipを保存した。
方式採用や5000ペアのGPU完了を示す資料ではない。
不要なreranker準備を除いた現在のfairness/ownershipは再測定済みで、
[修正後の結果](report.md#embedding専用ケースの準備修正後)と`results-revision2/`へ保存した。
旧RSSは修正後の保持条件へ流用しない。準備削除そのものの費用削減幅は未測定である。

## モデル不要の境界検証

`planner.rs`のテストは製品の通常libtestにも組み込まれる。
標準の `cargo fetch --locked` / `bash scripts/check.sh` とCIの契約は変更しない。
GPUやサーバーを起動せず、plannerだけ確認する場合は次を使う。

```sh
rustc --edition 2024 --test docs/research/issue-318/planner.rs -o /tmp/rurico-318-planner-tests
/tmp/rurico-318-planner-tests
```

## ホストで実モデル比較を実行する

Apple Silicon、対応するXcode/Metal、キャッシュ済みの固定310m embedding/rerankerモデルを用意する。
モデルrevisionは[既存#307報告](../issue-307/report.md#対象と測定条件)と同じ。
ホストの担当AIが、実際の開始commit、変更差分、Cargo.lock、モデル6ファイルを照合する。
今回の開始版は`4d8e9cc640a101b3935ea2ef3edcb43f1ad2d068`である。
checkout外の新しいbuild先・証拠保存先を指定し、他のビルド・GPU測定を止めてから実行する。

調査用crateはMLXを含む既存の製品sourceを相対symlinkで再利用する。
`probe/src/lib.rs`は製品のmodule入口を写した調査用入口で、追加の`host.rs`を接続する。
module一覧・Cargo.tomlを製品から変更した場合には、この入口との一致を再確認する。
調査用manifestの依存・feature・profileは開始版と同じで、lockは開始版のCargo.lockのコピー。
製品のCPU backend、依存分離、新しい公開APIを導入しない。

```sh
CARGO_TARGET_DIR=/tmp/rurico-318-host-build cargo clippy --offline --locked --lib --tests --features smoke --manifest-path docs/research/issue-318/probe/Cargo.toml -- -D warnings
CARGO_TARGET_DIR=/tmp/rurico-318-host-build cargo test --offline --locked --release --lib --features smoke --manifest-path docs/research/issue-318/probe/Cargo.toml --no-run --message-format=json > /tmp/rurico-318-build.jsonl
```

JSONLの`compiler-artifact`から、`target.name=rurico`、`profile.test=true`で
`executable`のあるlibtest binaryを一つ特定する。これは今回ビルドしたbinaryの絶対パスである。
`--offline`はCargo解決の指定で、初回MLXビルドのCMake取得は別である。
既存CONTRIBUTINGと同様に、必要なMLX sourceのネットワーク取得を許すホストでビルドする。
ビルドlog・実際のargv・開始版と差分hashを証拠に保持し、古いbinaryを使わない。
通常checkも同じ変更版のホストで別途実行する。

次のcontextは書式の例で、測定値ではない。ホストの担当AIが実環境を取得して埋める。
`model_files`には、固定HF snapshotのembed310mとreranker310mについて、
`model.safetensors`、`config.json`、`tokenizer.json`の実際の6パスを列挙する。
runnerはファイルhashとweight headerを取得するが、revisionやビルド由来の意味確認は担当AIが行う。

```json
{
  "machine": "実測した機種・chip",
  "ram_bytes": 25769803776,
  "os": "実際のOS版",
  "rust": "rustc --versionの結果",
  "cargo": "cargo --versionの結果",
  "xcode": "xcodebuild -versionの結果",
  "metal": "xcrun metal --versionの結果",
  "build": "今回のビルドargv・profile・targetとbinaryの対応",
  "load_conditions": "電源・熱・背景負荷・他GPU測定の有無",
  "model_files": ["embed/model.safetensors", "embed/config.json", "embed/tokenizer.json", "reranker/model.safetensors", "reranker/config.json", "reranker/tokenizer.json"]
}
```

```sh
python3 docs/research/issue-318/run.py /absolute/path/to/new/rurico-test-binary /tmp/rurico-318-context.json /tmp/rurico-318-host-evidence
```

runnerは従来の8ケースと追加deadlineを別の所有process groupで実行する。
`--cases`で今回影響したケースだけ選べる。既存証拠への上書きや自動再試行をしない。
測定開始から総30分、group RSS 8 GiB超、pressure正常以外、swap増分256 MiB超で自身の測定を止める。
これは#364の保護条件を参考にした今回の調査用の打切り条件であり、製品上限・許容deadlineではない。
モデルロードのpeakや計測自体の負荷を含む。RSSはGPU/共有メモリの全量ではない。
0.1秒を目安に観測するが、`ps`/`sysctl`の実行時間やOS schedulingにより遅れる。
停止時は所有groupを回収し、最大2秒でgroup不在を確認できなければ失敗を残す。
プロセスが自ら別groupへ離脱するケースは対象外である。

`context.json`、ケースごとのstdout/stderr・status・観測JSON、`monitor.json`を保持する。
モデル・source・binaryの前後hashと記録件数が揃った場合だけ`complete.json`を作る。
これは収集完了の記録であり、受入や改善の証明ではない。
source照合、guard、試験、記録検査のどれかが失敗したら後続ケースを実行しない。
SIGINT/SIGTERMでは部分証拠を保持する。SIGKILL・ホスト停止時の回収・最終記録は保証しない。

## 結果を判断して独立評価へ戻す

担当AIがrawから各条件の最小・中央値・最大、待ち時間とpeakを集計し、
producerのprepare区間とconsumerのforward区間の実際の重なりを照合する。
重なりの存在とwall短縮は別に判断する。3反復と順序反転だけで因果的な改善を断定しない。
比較ごとの準備・ロード・記録処理の費用と失敗も含め、同じ条件で比較する。
rerankerのscore差、入力・shape・dtype、cancel後の完了件数、公平性のbulk待ちを確認する。
生logや個人pathを一括公開せず、公開可能な数値・版・条件・停止と未確認事項を
[報告](report.md)へ追記して、変更文書を含む既存の独立評価へ戻す。

標準check、CIのtest/coverage/security/zizmor、captureなしの契約には、このrunnerと実モデル比較は含まれない。
capture媒体は不要。設定済みcheckの待機と、このホスト受入検証の待機を区別する。

## 修正後のホスト再検証手順

control/deadline/mixedの修正時に以下の手順を実行し、原資料・測定・変異・正常版復元の結果を
[報告](report.md#修正後のホスト検証と親の条件)と`results-revision1/`に保存した。
embedding専用ケースの追加修正も測定済みである。次節はその結果と再実行手順を示す。
再実行する場合は最初に親[#296](https://github.com/thkt/rurico/issues/296)の本文を原資料から取得し、URL・updatedAt・本文hashと
実行・保存に関係する条項を新しいcheckout外の証拠へ保存する。報告にはその版への参照と、開始版・lock固定、
Apple Silicon/Xcode/Metal、通常checkと追加smoke/modelの区別、保存する生log/公開要約の対応を追記する。
親の条項をこの列挙から推測しない。不足が事実確認だけなら調査を続け、要求・範囲・権限・上限の変更を要するなら実行前に人へ戻す。

今回の未commit差分で調査crateを再ビルドし、上記のClippyも確認する。
新しいcontext・binary・出力先を用意し、同じ保護条件で影響した3ケースだけ実行する。

```sh
python3 docs/research/issue-318/run.py /absolute/path/to/new/rurico-test-binary /tmp/rurico-318-new-context.json /tmp/rurico-318-new-evidence --cases control deadline mixed
```

`deadline`は逐次/queue各3反復で、最初のconsumer forward直前に測定用10 ms期限を一度だけ設定する。
producerはこのconsumer用期限を共有せず、1-slot queueのbackpressureとreceiver破棄で回収する。
実forwardの開始時の残り時間、終了時の超過時間とwall、各10件の返却、cancelなし、次forwardなし、完全Vec拒否を記録する。
開始前に期限を過ぎた場合やforwardが10 ms内に完了した場合は、必要な境界を測れなかった失敗として残す。
成功するまで期限を都合よく変える、自動再試行、sleepで境界を作ることはしない。製品deadlineやGPU即時中断の保証ではない。

検出力は今回の差分を含むcheckout外の独立コピーで確認する。正常版の成功後、`host.rs`のqueue consumerの
`stopped`判定だけを「completedが0の時のみ判定する」条件に変える。他の判定とcancel処理は保つ。
再ビルドして`--cases deadline`を新しい証拠先で実行し、2回目のforwardへの進入時に
`measurement deadline expired before forward`で失敗することを確認する。GPUへの2回目の呼出しは前提assertionが止める。
次に正常版へ戻し、逐次loopの判定だけを「startが0の時のみ判定する」条件に変え、同じ失敗を確認する。
各変異の間と最後に正常版を復元し、最後の正常版も同条件で成功することを確認する。
ビルド失敗・モデル不在・guard到達を回帰検出と数えず、binary/sourceの対応、argv、終了コード、assertion、生stdout/stderr、group回収を保存する。
変異は本checkoutと旧runへ適用しない。

新しい結果は旧`results/`を上書きせず別の保存先へ整理し、版・準備費用・wall/peak/RSS・実行上限・限界を報告へ反映する。
mixedはcontextとモデル構築が一度のreranker検証結果を共有する版で測る。
embedding検証、モデル構築時の全重み検証、source/model/binaryの前後hashは維持する。
これだけで旧測定との差を性能改善と断定しない。標準check成功は調査crateのビルドとこれらの実測を代替しない。

## embedding専用ケースの再検証手順

実行結果は[報告](report.md#embedding専用ケースの準備修正後)と`results-revision2/`に示す。
以下は再実行するときの手順である。

現在のfairness/ownershipはreranker準備の前に分岐し、各ケースで取得したembeddingのVerifiedArtifactsを
context記録とモデル構築で共有する。contextは固定embeddingのmodel/revision/configを示す。
embeddingの全検証、ロード時の全重み検証、rerankerを使うケースの検証は維持している。
runnerの測定前後の6モデルファイル・source・binary hash照合も変えない。

ホストの担当AIは現行差分で研究crateのClippyとrelease buildを上記の手順で確認する。
今回のbinaryと環境contextを新しく保存し、既存の保護条件・入力・options・3反復・順序反転を維持して二ケースだけ実行する。

```sh
python3 docs/research/issue-318/run.py /absolute/path/to/new/rurico-test-binary /tmp/rurico-318-embed-context.json /tmp/rurico-318-embed-evidence --cases fairness ownership
```

contextのembedding識別、fairness 96記録、ownership 48記録とpeak 6記録、終了コードとgroup不在を確認する。
source/model/binaryの前後hash、生stdout/stderr、監視RSS/pressure/swapと準備条件を保存し、公開可能な要約を報告へ反映する。
旧fairness/ownershipは余分なreranker tokenizerを保持した条件なので、旧RSSを修正後の証拠に使わない。
二ケースの出力へ8ケース専用のsummarize.pyを適用せず、元の記録を保って同じ単位で要約する。
時間・メモリの削減幅は未測定である。要約と文書を更新後、設定済みcheckと独立評価を受ける。

## 保存観測を再集計する

```sh
python3 docs/research/issue-318/summarize.py docs/research/issue-318/results > /tmp/rurico-318-summary.json
cmp docs/research/issue-318/results/summary.json /tmp/rurico-318-summary.json
```

集計器は8ケースの件数・条件を検査し、SHA-256を計算・記録する。期待hashとの照合は行わない。
`cmp`は別の比較で、保存済み`inputs_sha256`を含む集計全体とのbyte一致を確認する。
保存集計を同時に書き換えた場合の真正性までは保証しない。モデル・GPUは起動しない。
wall/allocator peakは各条件3試行、fairnessとownershipのtask/call統計はその3試行内の値であり、独立標本数としない。
この手順は旧8ケースの証拠の再集計である。選択した3ケースだけの新しい証拠先には使わない。
復帰後はhost.rs/run.pyも修正したため、旧集計の一致を新しい版の実モデル検証へ読み替えない。
