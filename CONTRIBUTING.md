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

`src/embed/tests.rs` の実tokenizerを使う2件と、`src/reranker/tests.rs` 内
`mlx_runtime_tests` モジュールのテストがこのカテゴリに該当する。

長文plannerは `src/embed/processing/tests.rs` で製品の関数を直接呼び、通常checkで検証する。
合成WordLevel tokenizerの異なるtoken IDを使い、prefix・overlap・末尾・文書順と、
上限を超える候補の縮小後もoverlapが保たれることを確認する。モデル取得は不要だが、
ビルドは引き続きMLX依存である。この検査は実tokenizer固有のprefix境界mergeや
実モデルの数値一致を保証しない。それぞれ上記のignoredテストと
[実モデルの受入手順](#重みの読込み検証)で別に確認する。

### `mlx_smoke` smoke テスト

`mlx_smoke` 統合テスト (`tests/mlx_smoke.rs`) と同名 binary (`src/bin/mlx_smoke.rs`) は `smoke` feature の背後にある。実推論モードとignoredの統合テストは、キャッシュ済みのruri-v3モデルとApple SiliconのMLX runtimeを要する。記録の単体テストと `summarize-records` はモデルをロードしない。標準CIは `smoke` featureを含めず、実モデル検証はホストで別に実行する。ignoredの統合テストを実行する場合は、事前に対象モデルをキャッシュしてから:

```sh
# ruri-v3 系モデルがローカル HF cache にあることを前提に走らせる
cargo nextest run --workspace --features smoke --test mlx_smoke --run-ignored=ignored-only
```

binary 版を直接呼ぶ場合:

```sh
cargo run --features smoke --bin mlx_smoke
```

### 公式実装との数値比較・利用側の検索評価（Issue #307）

公式Transformers/Sentence Transformersとの比較と、amiciの既存reference compositionの検索評価は、
[調査用のホスト実行手順](docs/research/issue-307/README.md)を使う。
固定revision、FP32、同一token列/shapeの比較、公開wrapper差、既存baseline差を分けて記録する。
`research::official_comparison_embedding`と`research::official_comparison_reranker`はignoredであり、
標準check成功だけでは実測済みにならない。参照出力は既存fixtureと別に保存する。
[準備時点の記録](docs/research/issue-307/report.md)から測定状況と未確認範囲を確認できる。

### options付き推論の計測（Issue #306）

`mlx_smoke measure-records` は固定revisionのキャッシュ済み310mモデルと既存W1/W2/W3の
公開可能な合成入力を使う。default optionsと `token_budget=256, forward_pause=1ms` の双方で、
通常／計測API、batch／singletonの連続呼出しを比較する。singleton側にも同じoptionsを渡す。
各組合せをwarm-up後に3回実行し、batch→sequentialとsequential→batchを交互にする。
通常／計測の順も試行ごとに反転する。3回なので完全に均等な順序ではない。
毎回、既存fixtureと文書順・chunk構成・数値を比較する（cosine ≥ 0.99999、最大絶対差 ≤ 1e-5）。
fixtureを再生成しない。shape・forward数・pause回数と、W2の非default時の分割を検査する。
pause時間はsleepの保証する要求時間以上だけを確認し、実時間の一致や上限は要求しない。
速度やタイミングの揺らぎは新たな合否閾値にしない。

短時間で計測追加のコストを比較する場合は、同じbinaryの `measure-overhead` を使える。
W2の先頭3文書とfixtureの対応する3文書を使い、同じ8組合せを各3回実行する。
workload名は `w2_first3` で、rawは33推論recordと8集計（ほかにmodel load 1件）。
非defaultでは2つのsub-batchとpauseを確認する。全workload検証の代用や長文への性能推定には使わない。
再集計手順・区間の意味・隔離条件は `measure-records` と共通。

GPUを利用できるホストで、他のGPU測定や重いビルドと並行せず実行する。
先にビルドを完了し、記録先はcheckout外の新規ディレクトリにする（計測中の記録で作業差分hashを変えない）。
測定開始前のprocess確認だけでなく、測定終了まで他作業のビルド・GPU測定が始まらない期間を
ホスト側で確保する。通常checkやCI用ビルドもこの期間と重ねない。
開始・終了時刻、負荷を止めた範囲、測定中のprocess名・CPU使用率と観測間隔を記録する。
processの定期観測は短い活動や未知のGPU利用を見逃し得るため、無検出だけを隔離の証明にしない。
負荷重複や監視欠落があった記録は条件未確認の観測として保持し、都合のよい試行だけを選別せず、
条件を確保して全組合せを再実行する。隔離条件は注記だけでは免除されない。

```sh
cargo test --locked --features smoke --bin mlx_smoke
cargo test --locked --features smoke --test mlx_smoke summarize_records
cargo test --locked --lib measured_trait_fallback
cargo test --locked --lib precise_metrics_keep
cargo build --locked --release --features smoke --bin mlx_smoke
# /tmp/rurico-306-evidence は新しい記録用ディレクトリの例
mkdir /tmp/rurico-306-evidence
RUST_LOG=off target/release/mlx_smoke measure-records > /tmp/rurico-306-evidence/raw.jsonl
# モデルをロードせず、生recordだけから同じ集計を再生成
target/release/mlx_smoke summarize-records /tmp/rurico-306-evidence/raw.jsonl > /tmp/rurico-306-evidence/summary.jsonl
```

`CARGO_TARGET_DIR`を指定している場合は、そのreleaseディレクトリを使う。
実行時の作業場所はこのcheckoutのルートとする。sourceの識別は実行時のcommit、tracked diff、
非ignoredのuntracked内容、Cargo.lock、実行ファイルのSHA-256。古いbinaryと新しいsourceを
結び付けないため、必ず上記buildの直後に実行し、使用したbuildコマンドも記録に添える。
比較する全組合せは同じbinary・build flags・入力・固定revisionを使い、実行中はsourceを編集しない。
source・fixture・Cargo.lockのファイル別SHA-256とbinaryのSHA-256を測定前後で照合して保存する。
rawと再生成summaryに加え、負荷条件・観測、model/tokenizer内容の確認結果または未確認の範囲を
同じ実行の証拠として保存する。再生成summaryとraw内のsummaryはJSON値として一致を確認し、
全組合せのサンプル数・中央値・最小最大と制約を報告する。記録は確認後に`docs/benchmarks/`へ置く。
Rust/Cargo/Xcode/Metalは実行時に取得した版で、binaryのビルドに使った版を証明するものではない。
取得不能な値・未計測のRSS/Metalメモリ・追加build flagsは`null`で、推定しない。
model/tokenizer revisionは実際のcache lookupに使う固定revisionだが、キャッシュ内容の改変までは検出しない。
入力本文・token列・個人のpath・認証情報はJSONに含めない。stderrはローカルpathを含む既存logがあり得るため、そのまま公開しない。

JSONLのschema 1は次のように読む。

- `event=model_load`は `Embedder::new` のhost wall。cache lookupとtokenizer検証は区間外。
- inference recordは一つの実行の値。`sequence`はprocess内順序、`repeat`はwarm試行番号。
  `method=batch`は1 API call、`sequential`は入力ごとにsingletonのoptions APIを呼び、
  `calls`にその順でmetricsを保持する。通常APIのmetricsは`null`。
- `state=first_inference`はprocess内最初の推論、`warmup`は後続の準備実行、`warm`だけが集計対象。
  `validation`は空入力確認用で集計対象外。
  毎forwardでbuffer/compile cacheをclearするため、warmはkernel cache保持の保証ではない。
  OSのファイルcacheは消していない。cold diskの測定とは呼ばない。
- 時間は`{"secs":整数,"nanos":整数}`。sub-msを保持する。`wall`は外側の呼出し全体、
  `calls[].wall`は各公開API内側の時間。JSON出力・fixture比較は外側wallの後。
  `preprocessing`等は内側wallの部分区間で、総和がwallになる保証はない。
  `tokenize=null`は前処理から未分離。`forwards`の長さがforward数で各要素は実際のpadded shape。
  `pause`は実sleep時間、`pause_count`は最後のforward後を含む実行回数。
- `event=summary`は同一context・入力hash・workload・options・method・measuredのwarm recordを集計する。
  `sequences`が原recordへの対応、`wall`がサンプル数・min・median・max。偶数の中央値は中央2値の中点。
  通常／計測の分布から追加計測コストを読み、各profileのbatch／sequential分布から方式差を読む。
  ばらつきが大きければ差を効果と断定しない。rawの値を集計値で上書きしない。

`measure-baseline`も同じrecord形式と交互順を使うが、defaultの計測経路だけを実行し、既存の
Phase 2閾値検査を続ける。stderrの `baseline` / `mdrow` は集計表示で、一回の推論値ではない。
通常checkだけではsmoke feature・実モデルの検証は完了しない。フルcheckとCIを同じheadで確認し、
公開可能な小さなraw record・summary・build条件・未確認事項を `docs/benchmarks/` に保存する。
今回の実測結果と制約は [Issue #306の検証記録](docs/benchmarks/issue-306-metrics.md) を参照。

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

### 重みの読込み検証

[Issue #300](https://github.com/thkt/rurico/issues/300) の CPU 検証は通常の check に含まれる。
合成 safetensors を両 loader に渡し、各必須キーの欠損、shape 違い、非対応 dtype、未知キーを
識別する。公式の [config](tests/fixtures/modernbert_configs/README.md) と
[ヘッダー](tests/fixtures/modernbert_weights/README.md)を使い、正規の parameter 集合を
誤拒否しないことも検証する。`weight_probe` は既存の probe binary を子プロセスとして起動し、
同じ欠損が公開 constructor と probe dispatcher の双方で読込み失敗になることを確認する。
CPU 検証用の合成ファイルの本体はゼロ埋めで、公式の数値出力を検証するものではない。
通常 check の `weight_load` は小さいモデルを MLX で保存・異なる seed から再ロードし、
全 parameter の値が復元されることを確認する。このテストはホストの Metal を要する。

対象を絞る場合:

```sh
cargo test --locked --lib --features test-support,test-mlx modernbert::
cargo test --locked --lib --features test-support,test-mlx reranker::mlx::tests::load_rejects
cargo test --locked --test weight_probe --features test-support
```

実モデルの受入は sandbox 外のホストで、固定 revision のモデル・tokenizer をキャッシュしてから
次を順番に実行する。通常 check はこれらの ignored 検証を実行しない。

```sh
cargo test --locked --lib --features test-mlx official_embedding_load_contract -- --ignored --nocapture --test-threads=1
cargo test --locked --lib --features test-mlx official_reranker_reload_contract -- --ignored --nocapture --test-threads=1
cargo run --locked --features smoke --bin mlx_smoke -- verify-fixture
cargo nextest run --locked --features smoke --test mlx_smoke --run-ignored=ignored-only -E 'test(probe_embed_smoke_binary) | test(probe_reranker_smoke_binary)'
```

embedding は既存 W1/W2/W3 fixture の `cosine_similarity >= 0.99999` と
`max_abs_diff <= 1e-5` を使い、fixture を再生成しない。
reranker はテストに記載した公開可能な4ペア、batch=4・seq=128・F32、seed=42/7/42で
再ロードする。比較前に定めた許容差は logit・score とも
`abs(a-b) <= 1e-6 + 1e-6 * abs(a)`、順位は完全一致とする。
head に bias がなく、最終 classifier bias が logit に加算されることも確認する。

出力ログは入力・model/tokenizer revision・seed・実行順・logit・score・順位と差分を含む。
`legacy-head` は基準 `1a53b0d` の dense bias 有効な head と初期化順を再構成した比較対象で、
旧 commit 自体の実行でも Hugging Face の公式出力でもない。旧 backbone の未使用 norm と
ゼロ bias を除いた現行 backbone を使い、再構成したスコアを同じ入力・seedの
[変更前ホスト実測](docs/benchmarks/issue-300-host-comparison.md)と上記許容差で照合する。
この条件を旧版全体との厳密一致と混同しない。変更後も同資料の公開API probeを同条件で実行し、
生logitを記録する内部テストと公開APIでの比較を区別して記録する。

load コストは各モデルで検証あり／なしを交互の順番で3回測る。どちらも同じ現行 parameter
構成・重み・seedを使い、検証なしはテスト内で mlx-rs 0.32.0 の従来 loader を直接呼ぶ。
モデルを破棄して allocator cache を clear してから次をロードし、大きなモデルを同時保持しない。
ログの wall time はモデル構築から重み eval 完了まで、Metal peak はその区間の MLX allocator
最大使用量であり、CPU ヘッダーやプロセス全体の RSS は含まない。

RSS が必要な比較では先に `cargo test --locked --lib --features test-mlx --no-run` でビルドし、表示された lib test 実行ファイルを
`/usr/bin/time -l <実行ファイル> official_reranker_reload_contract --ignored --nocapture --test-threads=1`
のように実行する。これは検証プロセス全体の peak RSS で、各 load 区間の CPU 追加量ではない。
OS のファイルキャッシュは clear しないため、初回と後続を分け、cold disk の測定と呼ばない。
各試行と範囲・ばらつき、機種・OS・Rust・Xcode/Metal・Cargo.lock・対象 commit、未実行条件を
ホストの検証記録へ残し、公開する結果は PR の `docs/benchmarks/` 等へ整理する。
CPU 検証や定義の用意だけで GPU 数値一致、実測コスト、検索品質の改善を確認済みとしない。

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
