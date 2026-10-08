# 分割優先の検知確認（Issue #360）

固定の期待断片と原文との連結比較を追加した。対象限定の通常テストは成功した。
一時コピーの本番関数から段落・行優先を削除すると、狙った断片比較で失敗し、復元後は成功した。
全体checkはホストで実行する。

## 対象と根拠

要求と合意範囲の正本は [Issue #360](https://github.com/thkt/rurico/issues/360)。
2026-10-08の開始commitは `24725a72be44300afc24186b82b14bcb3f5f9d3d`。
Issueの固定source
([関数](https://github.com/thkt/rurico/blob/c8f250d60a5afb9944b9008d22ad6f4dda2d7103/src/text.rs)、
[テスト](https://github.com/thkt/rurico/blob/c8f250d60a5afb9944b9008d22ad6f4dda2d7103/src/text/tests.rs))
と開始commitの両ファイルに差分はなかった。
初回記録ではローカルの `origin/main` と開始commitの一致だけを確認し、
GitHubへのDNS解決が失敗したため、着手時のリモートmainとの一致を未確認としていた。
下記の追調査で、着手直前のfetch記録と開始版の一致を確認した。
別の引き継ぎ報告は指定されていない。

[公開契約](../../../src/text.rs)は `\n\n` → `\n` → UTF-8文字境界の順。
CRは正規化せず保持する。`max_bytes < 4` は空入力も含めて入力全体を返し、
この条件ではbyte上限を要求しない。
[対象テスト](../../../src/text/tests.rs)は本番関数を直接呼び、期待値は固定の文字列で指定する。
段落より後の行が同じ枠に入る例で優先を区別し、境界前後、日本語、CR/LF、
2・3・4 byte文字、空入力と原文保持も確認する。FTSやplannerは変更していない。

### R1-1の追調査（2026-10-08）

初回評価 `review-1.json` のR1-1（対象ID
`34eb31154bcdb001176987c1de8e07958736cb33dbb68f6af3bf1629974ccdef`）は、
取得時点を伴うmainの根拠不足を指摘した。旧文書のSHA-256は
`350620d4dffeae812320d0d5a1e577b4b3d24643b4ad80534f2c8fd522aec2c4`。
原因は、現在のリモート取得失敗を記録する一方、worktreeと共有するGit管理領域の
着手直前のfetch記録を調べていなかったことだった。旧評価・実行結果は保持する。

`git rev-parse --git-common-dir` で特定した共有管理領域の
`logs/refs/remotes/origin/main` と、このcheckoutのHEAD reflogを照合した。
`origin` のURLは `https://github.com/thkt/rurico.git`。
次のコマンドで確認した最新のmain更新記録は、2026-10-08 13:13:40 JSTの
`fetch origin main: fast-forward` で、更新前は
`75ac8f20d5279da995fe5a17e8c4544f553d131c`、取得版は開始commitと同じ
`24725a72be44300afc24186b82b14bcb3f5f9d3d` だった。
このcheckoutのHEAD作成記録は同日13:14:08 JST、同じcommitだった。
開始版と、この着手直前に取得されたmainには差分がない。開始版は置き換えていない。

```sh
git remote get-url origin
git reflog show --format='%H %gD %gs' --date=iso-strict -1 refs/remotes/origin/main
git reflog show --format='%H %gD %gs' --date=iso-strict HEAD
git diff c8f250d60a5afb9944b9008d22ad6f4dda2d7103 24725a72be44300afc24186b82b14bcb3f5f9d3d -- src/text.rs src/text/tests.rs README.md CONTRIBUTING.md scripts/check.sh scripts/test.sh Cargo.lock
```

Issue固定版は開始版の祖先であり、対象関数・旧テストに差分はない。
READMEの分割優先の説明も同じ。検証方針は開始版で更新されており、
標準checkにsmoke feature、Pythonテスト、`scripts/test.sh` による実行対象の確認が加わった。
CONTRIBUTINGもこの条件と、通常checkでは実モデルのignoredテストを実行しないことを説明する。
既存の検知記録は開始版の3 featureと固定Cargo.lockを使っており、この方針に適合する。
lockの差はthiserror系の2.0.20から2.0.21への更新で、記録したlock hashは現在も一致する。
#359の性能・readbackの説明追加は分割契約を変更せず、本Issueの受入条件には加えない。

この追調査でも `git ls-remote` とscoutのGitHub API取得はDNSエラー、代替のweb取得は利用不能だった。
したがって根拠は着手直前のローカルfetch履歴であり、修正時点のリモートmainを再取得した証拠ではない。
共有の `FETCH_HEAD` は別の取得版を指していたため、着手版の証拠には使っていない。
再評価では上記reflogと版間差分を照合し、初回の根拠不足が解消したか確認する。

## 再実行手順

[CONTRIBUTINGの通常check条件](../../../CONTRIBUTING.md#テスト)と同じ
Apple Silicon、対応Xcode/Metal、locked依存、MLX-only構成を使う。
モデル取得・推論は不要。以下はcheckoutルートから実行し、作業コピーとログをcheckout外に保存する。
設定済みの `bash scripts/check.sh` は通常テストを実行するが、この一時変異は含まない。

```bash
probe_root=$(mktemp -d "${TMPDIR:-/tmp}/rurico-360.XXXXXX")
mkdir "$probe_root/checkout"
tar --exclude=.git --exclude=target -cf - . | tar -xf - -C "$probe_root/checkout"
(
  set -eu
  cd "$probe_root/checkout"
  export CARGO_TARGET_DIR="$probe_root/target"
  cargo fetch --locked
  run_text_tests() {
    cargo nextest run --locked --lib --features test-support,test-mlx,smoke \
      --profile ci --run-ignored default -E 'test(text::tests::)' \
      --no-tests fail --status-level all --final-status-level all
  }
  run_text_tests > "$probe_root/baseline.log" 2>&1
  patch -p1 < docs/research/issue-360/character-only.patch
  if run_text_tests > "$probe_root/mutant.log" 2>&1; then
    echo 'ERROR: character-only mutation was not detected' >&2
    exit 1
  fi
  patch -R -p1 < docs/research/issue-360/character-only.patch
  run_text_tests > "$probe_root/restored.log" 2>&1
)
```

非ゼロ終了だけでは検知成功としない。`mutant.log` で
`split_boundary_priority_preserves_text` の `paragraph_before_later_line: split boundary priority`
assertionが、実際の `["aa\n\nbbb\n", "cccccccc"]` と期待する
`["aa\n\n", "bbb\n", "cccccccc"]` の不一致で失敗することを確認する。
ビルドや環境の失敗はこの検知の証拠にならない。
[character-only.patch](character-only.patch) は本番関数の段落・行探索だけを外し、
UTF-8境界と4 byte未満の例外を維持する。恒常的なmutation基盤には組み込まない。

## 確認結果と限界

2026-10-08、macOS 27.0.1（26A434）、arm64、Xcode 27.0（27A266a）、
Metal 32023.921、Rust/Cargo 1.99.0、nextest 0.9.146で確認した。
Cargo.lock SHA-256は `743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。
対象テスト SHA-256は `58406078c125b096372046b9428b1f19ea0d49f7ffba658d948da70df8dc188f`。

上記と同じnextestコマンドを同じ一時checkout・targetで実行した結果:

| 版 | 終了コード | 結果 | nextestの実行区間 |
| --- | --- | --- | --- |
| 変更後の通常版 | 0 | 4件成功、401件対象外 | 0.020秒 |
| character-only変異 | 100 | 3件成功、狙ったassertionで1件失敗、401件対象外 | 0.021秒 |
| patchを戻した版 | 0 | 4件成功、401件対象外 | 0.032秒 |

変異時の実際の失敗は上記の `paragraph_before_later_line` の断片不一致だった。
本番ソースの修正は不要だった。既存のビルドcacheを一時targetへ複製して利用した。
時間はnextestのSummaryに表示された実行区間のみで、ビルド時間を含まない。
各版1回の検知確認であり、性能やテスト整理による高速化の証拠にはしない。

対象限定の `cargo test --locked --lib --features test-support,test-mlx,smoke text::tests:: -- --nocapture`
は4件成功、ignored実行なし、401件はfilter対象外だった。
全体check・同じheadのCIは未実行であり、この結果を代用しない。
smoke featureは有効だが対象filterによりsmoke suiteは実行していない。
実モデル、ignored/model検証、consumer互換性や性能は今回確認していない。
