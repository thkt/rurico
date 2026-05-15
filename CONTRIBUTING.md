# Contributing to rurico

## テスト

CI で実行されるテスト一式は以下で再現できる。

```sh
cargo nextest run --workspace --features test-support,test-mlx
cargo test --doc --workspace --features test-support,test-mlx
```

`cargo nextest run` は doctest を走らせないため、`cargo test --doc` を別途実行する。nextest 未インストールの場合は `brew install cargo-nextest` または `cargo install cargo-nextest --locked`。

`test-support` / `test-mlx` feature を有効にすることで、CI の clippy step (`cargo clippy --workspace --all-targets --all-features -- -D warnings`) が見ているコードを test 側でも exercise する。CI の test step もこの組み合わせで実行される。

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

`tests/visibility.rs` は trybuild で `tests/ui/*.rs` を compile_fail として exercise する。各 fixture の build に Metal Toolchain (Xcode Command Line Tools 同梱) を要する。

- CI (macOS-latest runner): Xcode CLT 同梱、通常通り走る
- ローカル環境で Metal Toolchain 不在: `cannot execute tool 'metal'` で trybuild が build fail する

Toolchain 不在の環境で他テストだけ走らせる場合:

```sh
xcode-select --install                          # Metal Toolchain を導入する場合
cargo nextest run --workspace --lib --bins     # または trybuild test target を含めず走らせる
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
