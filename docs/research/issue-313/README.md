# Issue #313 重複候補・集約allocationの調査

[報告](report.md)に現行契約、consumer照合、未採用案、修正版のCPU測定と残る検証を示す。
要求・採用権限の正本は [Issue #313](https://github.com/thkt/rurico/issues/313)。
既定の重複処理、公開API、JSON、モデル処理を変更する入口ではない。

現在の修正は公開head `883b4a53e25451cfffa78e5e08d7ea84a0787482` から始め、
main `01dd1c69c633a7f2eeccd188d566ec25d0f94e93` のファイルを統合した未commit版である。
GitHubでPR #380の作者thkt・同じhead・draftとmainの版を照合した。
初回のsandbox作業では版固定blobを三者統合した後、Git状態の準備をホストへ引き継いだ。
その後ホストが最新baseを取得し、既存branchの未commit mergeを準備して競合を解消した。
現在の`MERGE_HEAD`は上記mainと一致し、未解決indexはない。
保存した統合ファイルとのbyte一致、mainのlock・requirements・追加検証の保持も確認済みである。
このGit統合確認は完了しており、再実行を待つ状態ではない。
[報告のホスト確認](report.md#ホストでのgit統合確認)と
[source対応記録](results/main-integration-source-check.json)を参照する。
merge commitは未作成で、公開branchの祖先関係の確定は後続の公開工程で確認する。
今回もcommit・push・公開は行わない。

製品runtimeとprobeは旧測定版から変わらず、native release再buildは
R1測定binaryとSHA-256が一致した。[統合sourceとの対応](results/main-integration-source-check.json)を参照する。
旧測定sourceのhashを統合sourceのhashに読み替えない。現在版のCPU検証は53件とclippyが成功した。
ホストの統合版checkは成功したが、独立評価で文書の現在状態に不一致が指摘された。
この文書修正後の標準checkと変更文書を含む独立評価はホストの既存手順へ渡す。
旧accepted・check・CIは今回の版の成功として利用しない。GPU・実モデルは実行していない。

## 検証する

製品のセットアップ・全体検証はルートの [CONTRIBUTING](../../../CONTRIBUTING.md#テスト) と
設定済みの `cargo fetch --locked` / `bash scripts/check.sh` を使う。
以下のCPU調査は全体検証を置き換えず、製品のCPU backendや依存分離を導入しない。

調査用crateは製品の `src/retrieval.rs` とその既存テストを直接コンパイルする。
serde関連の依存は開始版Cargo.lockの版を使い、調査用lockにも固定した。
モデルの取得・GPU初期化は行わない。
テスト・clippy・fmtだけ確認する場合も、このcheckout専用の新しいbuild先を使う。
別checkoutや不具合注入のコピーと同じbuild先を共有すると、相対source pathのcacheを取り違え得る。

```sh
# /tmp/rurico-313-tests は未使用のパスを指定する
CARGO_TARGET_DIR=/tmp/rurico-313-tests cargo test --offline --locked --manifest-path docs/research/issue-313/probe/Cargo.toml
CARGO_TARGET_DIR=/tmp/rurico-313-tests cargo clippy --offline --locked --manifest-path docs/research/issue-313/probe/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path docs/research/issue-313/probe/Cargo.toml -- --check
```

## ホストで時間・allocation・RSSを測る

macOS Apple Siliconの権限のあるホストで、ほかの重いビルド・計測と重ならない時間を確保する。
実行前後の機種・RAM・負荷条件と、確保した期間を別途記録する。
runnerは負荷の隔離を保証しない。モデルとMetalは不要。
outputとtarget-dirはcheckout外の**新規**ディレクトリを指定する。
rootのsetupが取得する依存の一部だけを使い、測定時はoffline/lockedを維持する。

```sh
python3 docs/research/issue-313/run.py /tmp/rurico-313-host-evidence --target-dir /tmp/rurico-313-host-build
```

runnerは調査テスト、clippy、release build、大小入力での出力一致を確認してから、
2 workload × 6 variant × 3 processを実行する。2回目はprocess順を逆転する。
各processは1回warm-up後に7回測定し、計252 callのrawと36 processのpeak RSSを残す。
時間にallocator counterの費用を含む。速度・メモリの新しい合否閾値は設けない。

- `context.json`: 開始commit、source/lock hash、compiler、測定条件。
- `raw.json`: callごとのns、allocation/reallocation回数、要求byte総数、追加live heap peak、process RSS。
- `summary.json`: 同条件の最小・中央値・最大。RSSはprocess全体のpeakで、call区間の値とは区別する。
- `complete.json`: source前後一致、binary/raw/summary hash、件数。存在だけで要求充足を判断しない。

source変更、子process失敗、7回の欠落、RSS取得不能では非ゼロで停止し、部分出力を保持する。
同じ出力先へ再実行せず、失敗原因を確認して新しい保存先を使う。
記録には私的pathを含むため、生ログ・contextを一括でrepoへコピーしない。
担当AIが条件、raw、再集計値を照合し、公開可能な数値・source識別・未確認事項だけを
この報告へ追記し、既存の文書を含む独立評価へ戻す。

`scripts/check.sh`、CIのtest/coverage/security/zizmor、captureなしの契約には
このrunnerやprocess RSS測定が含まれない。標準check待ちと、この追加受入検証待ちは区別する。
今回のsandboxでは `/usr/bin/time -l` が `sysctl kern.clockrate: Operation not permitted` で失敗した。
[部分記録](results/partial-context.json)を完全なホスト測定へ読み替えない。

2026-10-08のR1修正版は36 process・252 callとRSSの取得を完了した。[修正版context](results/host-r1-context.json)、[原数値](results/host-r1-raw.json)、[再集計](results/host-r1-summary.json)、[完了記録](results/host-r1-complete.json)を参照する。外部負荷の隔離は未確認で、速度改善の因果的な証拠として使わない。修正前の原本は保持し、版を混ぜず読む。製品の全体check・独立評価・方式採用はそれぞれ別の条件である。
