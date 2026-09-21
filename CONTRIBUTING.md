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

### 推論失敗時のcleanupとメモリ観測

[Issue #303](https://github.com/thkt/rurico/issues/303) の順序回帰は通常のcheckに含まれる
`inference_drops_resources_before_one_cleanup_and_preserves_result` で確認する。
forward・pool・eval・readbackを個別に失敗させ、一時リソースの解放後にcleanupが1回走り、
元のエラーまたは成功出力を保持することを検証する。GPUメモリは測定しない。
`injection_controls_select_only_the_requested_stage_and_thread` は、実際の注入制御が
指定段階だけを失敗させ、解除後は成功し、別スレッドに注入設定・cleanup計数を漏らさないことを確認する。
cache handle取得・clear・解放と失敗通知は既存のFFI/cacheテストで引き続き確認する。

`src/mlx_cache/testing.rs` の順序検査と注入制御は通常CIのcoverage対象に含める。
cachedモデル必須の観測コードだけを `src/mlx_cache/testing/runtime.rs` に分け、
このファイルに限ってcoverageの分母から除外する。観測テストは引き続きビルドされ、
以下のホスト実行で検証する。`modernbert/model.rs` のForward注入地点は除外しない。
coverageは通常CIで実行した行を示す指標であり、実モデル観測の代わりにはしない。

実モデルの反復観測は、GPUが使えるsandbox外のホストで、embed 310mとreranker 310mの
固定revisionをHF cacheに配置してから次のテストだけを実行する。モデル未配置は失敗となる。
nextestの専用プロセスで動かし、他のGPU測定と並行させない。

```sh
cargo nextest run --locked --lib --features test-mlx --run-ignored=ignored-only --test-threads=1 --success-output=immediate real_model_cleanup_memory
```

公開APIのquery・2文書batch・2ペアrerankerを、それぞれbucket 128の固定合成文で実行する。
各経路は「成功→forward失敗→pool失敗→eval失敗→readback失敗」を6周し、最後に成功へ復帰する。
forward失敗はbackboneの最初のlayer後、pool失敗はembedのmask生成後／rerankerのCLS抽出後、
eval失敗は実eval後、readback失敗はslice取得後に、テストビルド限定の例外を注入する。
実際のOOMやMetal故障を発生させるテストではない。注入設定はthread-localで製品ビルドには含めない。
各試行でcleanup回数とエラーを確認し、成功出力は同じ経路の初回出力と最大絶対差`1e-5`以内で比較する。
従来出力との比較には既存の`smoke_verify_fixture`、rerankerの順位・入力順テストを併用する。

`cleanup_memory` 行にはモデル/tokenizerの固定revision、経路、反復番号、失敗段階、batch、bucket、
処理時間、MLXのactive/cache/peak bytesを出力する。peakは試行ごとにリセットする。
最初のquery/reranker試行はモデルをロードした直後で、batchではqueryの実行済みモデルを使う。
常駐する重みとlocal maskを含む値であり、コンパイルcache全体やプロセスRSSと同じ指標ではない。
任意のメモリ閾値による合否判定は加えていない。各経路の初回と後続、成功と失敗の推移を確認し、
継続増加があればその条件を調査する。テスト成功だけでメモリ観測の評価完了としない。

[親Issue #296](https://github.com/thkt/rurico/issues/296) に従い、対象commitと差分、lockfile、
Rust/Xcode/Metal、機種/OS、モデルrevision、実行コマンドと全サンプルを検証記録へ残す。
ホストでは同じ実行のRSSも観測し、例えば上記コマンドに`/usr/bin/time -l`を付ける場合は、
その最大RSSが実行全体の値であって試行ごとの値ではないことを明記する。
共有する結果は`docs/benchmarks/`等の証拠保存先へ置き、通常の操作説明に実測値を混在させない。
未実行・モデル不足・観測不能の指標は明記する。mockの順序確認や短い固定shapeの反復から、
全shape・全モデル・長時間運転のGPUメモリ保証やリーク削減量を推定しない。

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
