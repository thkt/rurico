# Contributing to rurico

## テスト

CI で実行されるテスト一式は以下で再現できる。

```sh
cargo test --workspace --features test-support,test-mlx
```

`test-support` / `test-mlx` feature を有効にすることで、CI の clippy step (`cargo clippy --workspace --all-targets --all-features -- -D warnings`) が見ているコードを test 側でも exercise する。CI の test step もこの組み合わせで実行される。

### `#[ignore]` テストの実行

ネットワークアクセスや実モデルを要するテストは `#[ignore]` で gate されており、デフォルトでは実行されない。再有効化方法は各テストの doc comment に記載してある。例:

```sh
# 実モデルを HF Hub からダウンロードして tokenizer 動作を検証
cargo test -- --ignored g_001_real_tokenizer_extract_prefix_tokens
```

`src/embed/tests.rs` の 3 件と、`src/reranker/tests.rs` 内 `mlx_runtime_tests` モジュールの test がこのカテゴリに該当する。

### `mlx_smoke` smoke テスト

`mlx_smoke` 統合テスト (`tests/mlx_smoke.rs`) と同名 binary (`src/bin/mlx_smoke.rs`) は `smoke` feature の背後にあり、実 ruri-v3 モデル + Apple Silicon の MLX runtime を要する。CI では走らせない（モデルダウンロードと推論で macos-latest runner の 15 分 timeout を圧迫するため）。ローカルで実行する場合は事前に対象モデルをキャッシュしてから:

```sh
# ruri-v3 系モデルがローカル HF cache にあることを前提に走らせる
cargo test --workspace --features smoke --test mlx_smoke -- --ignored
```

binary 版を直接呼ぶ場合:

```sh
cargo run --features smoke --bin mlx_smoke
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
