# Contributing to rurico

## テスト

macOS Apple Silicon、Rust 1.96+、XcodeとMetal Toolchain、cargo-nextestを用意する。
`xcodebuild -version` と `xcrun metal --version` で使用する版を確認する。
Metal Toolchainがない場合は `xcodebuild -downloadComponent MetalToolchain` で導入する。
Command Line Toolsの導入だけでMetal Toolchainが揃うとは限らない。

依存関係はCargo.lockに固定する。mlx-rs 0.32 / mlx-sys 0.6の組み合わせを使い、
固定したmlx-sys 0.6.0の配布ソースはMLX v0.32.2を参照する。
MLXのビルドではCMakeがそのソースを取得するため、Cargoの依存取得後も初回ビルドにはネットワークが必要になる。
既存CIと同じcheck・nextest・doctest・clippy・fmtを省略せず実行する入口は次のとおり。

```sh
cargo fetch --locked
bash scripts/check.sh
```

`scripts/check.sh` は `test` job相当の検証を行う。`coverage`（変更行95%以上）、
`security`（`cargo deny check` / `cargo audit`）、`zizmor` は別のCI jobで確認する。
依存更新時もcoverage閾値・ignore・タイムアウトを通過目的で緩めない。
Xcode 27 / Metal Toolchain 27A266aでの検証対象と経緯は [Issue #322](https://github.com/thkt/rurico/issues/322) を参照。

compile cacheのFFIテストはモデル不要で、現在のcacheの取得・clear・handle解放と、
取得／clear／解放の失敗コードの保持を検証する。実FFIの取得時にもMetal初期化が発生し得るため、
checkはGPUを利用できるsandbox外のホストで実行する。推論の検証は既存のMLXテストを使う。
実行結果には使用したツールチェーン、実行した検証、モデル未配置などで実行しなかった検証を記す。
通常のcheck成功だけでignoredテストや全モデルの数値一致・性能改善を確認したとは扱わない。

CI で実行されるテスト一式は以下で再現できる。

```sh
cargo nextest run --workspace --features test-support,test-mlx
cargo test --doc --workspace --features test-support,test-mlx
```

`cargo nextest run` は doctest を走らせないため、`cargo test --doc` を別途実行する。nextest 未インストールの場合は `brew install cargo-nextest` または `cargo install cargo-nextest --locked`。

`test-support` / `test-mlx` feature を有効にすることで、CI の clippy step (`cargo clippy --workspace --all-targets --all-features -- -D warnings`) が見ているコードを test 側でも exercise する。CI の test step もこの組み合わせで実行される。

### CIキャッシュの検証

macOSのtest・coverageジョブは、Cargoのビルド成果物とmlx-sysが生成する
`~/.mlx/lib/<source-key>/mlx.metallib`を、rust-cacheの`cache-directories`で一緒に保存・復元する。
`key: mlx-metallib-v1`は保存キーと復元用のプレフィックスに含まれ、Metalライブラリを含まない
旧キャッシュの再利用を避ける。ジョブごとのキー分離はrust-cacheの既定設定で維持する。

キャッシュ設定を変更した場合は、同じcommitのGitHub Actionsで初回実行と再実行を確認する。
test・coverageの`Cache Cargo`と保存処理のログで`Cache Paths`に`~/.mlx/lib`が含まれること、
初回にキャッシュが保存され、別runnerでの再実行がそのキーを完全一致で復元することを確認する。
初回から既存キャッシュに一致した場合は、その実行だけで初回ビルドの証拠とは扱わない。
復元後も`mlx_cache::tests::current_cache_cleanup_succeeds_without_warnings`を含むtestジョブと、
coverage・security・zizmorが成功することを確認する。モデル追加ダウンロードは不要。
commit、run URLと実行回、保存・復元キー、対象パス、テスト結果をIssue・PRの検証記録に残す。
通常のcheck成功だけでは、別runnerへのMetalライブラリ復元を検証したことにはならない。
背景と完了条件は[Issue #327](https://github.com/thkt/rurico/issues/327)を参照。

### `#[ignore]` テストの実行

ネットワークアクセスや実モデルを要するテストは `#[ignore]` で gate されており、デフォルトでは実行されない。再有効化方法は各テストの doc comment に記載してある。例:

```sh
# 実モデルを HF Hub からダウンロードして tokenizer 動作を検証
cargo nextest run --run-ignored=ignored-only g_001_real_tokenizer_extract_prefix_tokens
```

`src/embed/tests.rs` の 3 件と、`src/reranker/tests.rs` 内 `mlx_runtime_tests` モジュールの test がこのカテゴリに該当する。

### `mlx_smoke` smoke テスト

`mlx_smoke` 統合テスト (`tests/mlx_smoke.rs`) と同名 binary (`src/bin/mlx_smoke.rs`) は `smoke` feature の背後にあり、実 ruri-v3 モデル + Apple Silicon の MLX runtime を要する。CI では走らせない（モデルダウンロードと推論で macos-latest runner の 15 分 timeout を圧迫するため）。ローカルで実行する場合は事前に対象モデルをキャッシュしてから:

```sh
# ruri-v3 系モデルがローカル HF cache にあることを前提に走らせる
cargo nextest run --workspace --features smoke --test mlx_smoke --run-ignored=ignored-only
```

binary 版を直接呼ぶ場合:

```sh
cargo run --features smoke --bin mlx_smoke
```

### `visibility` integration test (trybuild + Metal Toolchain)

`tests/visibility.rs` は trybuild で `tests/ui/*.rs` を compile_fail として exercise する。各 fixture の build にMetal Toolchainを要する。

- CI (macOS-latest runner): runner上のXcodeとMetal Toolchainを使用する
- ローカル環境で Metal Toolchain 不在: `cannot execute tool 'metal'` で trybuild が build fail する

Metal Toolchainがない場合は導入してからcheckを再実行する。
調査用にtrybuildの対象を外してもMLX依存のビルドにはMetal Toolchainが必要であり、完全な検証の代替にはならない。

```sh
xcodebuild -downloadComponent MetalToolchain
xcrun metal --version
```

## ブランチと commit

- ブランチ名は `<type>/<short-topic>` 形式（例: `fix/foo-bar`, `ci/baz-qux`）。
- commit message は Conventional Commits（`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `polish:`, `ci:` 等）。

## Lint と format

PR 提出前に以下が通ることを確認する。

```sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt -- --check
```

## `docs/issues/`

ローカル issue メモ用ディレクトリ。`.gitignore` で `docs/issues/` が指定されており、新規ファイルは git 追跡されない。

- 個人の作業メモ・調査草稿などを置く場所
- リポジトリに残したい issue / ADR / audit 結果は `docs/decisions/` (ADR) または `docs/audit/` (監査結果) に昇格させる
- gitignore 追加前から追跡されていた既存ファイル (`docs/issues/typed-fts-query-contract-migration.md` 等) は継続追跡される (`git update-index --skip-worktree` していないため、編集すると diff が出る)
