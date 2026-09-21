# Issue #303: 推論失敗後のcleanup観測

2026-09-21に実施。[Issue #303](https://github.com/thkt/rurico/issues/303)の実モデル観測に対応する。93回の推論がすべて期待した成功または注入エラーとなり、各失敗の後も成功出力へ復帰した。6サイクルの範囲では、失敗を繰り返すほど増え続ける傾向を確認しなかった。これは有限回・固定shapeの観測であり、長時間・全shapeのGPUメモリ保証ではない。

本記録の実測と通常checkの件数は、公開版 `53c7442a41e1de193cde7e6e68dfa235f96f19bb`
に収録された測定対象manifestの版に限る。以下のcoverage修正後の再実行結果ではない。

## coverage修正との対応と再検証

[PR #329のcoverage失敗](https://github.com/thkt/rurico/actions/runs/35561524033/job/106215159012)
は上記公開版で変更行159行中101行未カバー、36%だった。通常CIで実行しない
cachedモデル観測が `src/mlx_cache/testing.rs` に混在し、注入制御も通常テストから実行されていなかった。
今回の修正では観測部分を `src/mlx_cache/testing/runtime.rs` に移し、このファイルだけを
coverageの分母から除外する。観測の入力・93試行・検査・ログ形式は維持する。
順序回帰と実際の注入制御は分母に残し、指定段階だけの失敗、解除、繰返し、
別スレッドへの設定・計数の漏出を通常テストで検査する。
`modernbert/model.rs` の部分Array生成後のForward注入地点も分母に残す。
95%基準、製品の推論・cleanup・cache policy、依存版は変更しない。

失敗ログの未カバー101行は、観測部分95行、注入制御5行、Forward注入地点1行に分かれる。
追加した通常テストは注入制御の成功・失敗分岐を実行する。
旧計測の行対応で観測部分を除き、この5行をカバーすれば63/64行（98.4%）となる。
これは修正方針の根拠であり、追加テストの行や新しいcoverage mappingを含む実測値ではない。
修正後の順序・注入制御テスト2件、対象crateのlib/testsのclippy、fmt、差分の空白検査は成功した。
実モデルの観測部分は同じrustfmtを適用した移動前後で一致することも確認した。

既存manifestと今回のソースを照合すると、記載済みファイルの変更は
`src/mlx_cache/testing.rs` のみで、分離先が新しく加わる。
元ログ・manifest・出力・ツールチェーン記録は過去の証拠として保持する。
ファイル移動を保守費用や速度の改善とは扱わず、同条件の実行時間比較は未実施。

ホストでは通常の `bash scripts/check.sh` に加えて、次の狭いcoverage実行で
順序・注入制御・cache/FFIの検査を計測し、同じ95% gateを確認する。
使用するbaseは `origin/main`。公開時にはCIのworkspace全体のcoverageも確認する。
下記の結果は本記録にはまだ含まれていない。

```sh
cargo llvm-cov --locked --lib --features test-support,test-mlx \
  --ignore-filename-regex '(test_support\.rs|/bin/|embed/(mlx|embedder)\.rs|reranker/mlx\.rs|model_io/hf_backend\.rs|/mlx_cache/testing/runtime\.rs$)' \
  --lcov --output-path /tmp/rurico-303-revision-lcov.info -- mlx_cache::
diff-cover /tmp/rurico-303-revision-lcov.info --compare-branch=origin/main --fail-under=95
```

さらに後述の同じ実モデルコマンドを新しい版で再実行し、3経路93試行の結果・全メモリサンプル・
RSSを新規ログへ保存する。モデルrevision・shape・環境・ソースハッシュを記録し、
移動後も注入と復帰を確認する。成功出力は末尾の保存済み比較プログラムを新しい版へ向けて実行し、
同じseed=42で保存済みの全出力と比較する。元資料を上書きせず、新版の結果として対応付ける。
通常check・狭いcoverage・実モデル再測定・独立評価・同じheadの必須CIはホストで確認する。

## 対象と条件

- 開始版: `1a53b0dc1859b0a432b637faf79b3287304b8875`。変更後のRustソースとCargo設定・lockのSHA-256は[測定対象manifest](issue-303-cleanup-code.json)に保存した。
- macOS 27.0 (26A428)、Apple M3、統合メモリ24 GB。Rust/Cargo 1.98.1、mlx-rs 0.32.0、mlx-sys 0.6.0。debugビルド、ホストのMetal環境で単独実行。
- Xcode 27.0（build `27A266a`）、Metal Toolchain asset `27.1.266.1`、Apple metal `32023.921`（`metalfe-32023.921.6`）。版の遡及確認方法は下記を参照。
- embed: `cl-nagoya/ruri-v3-310m`、revision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`、safetensors 1,258,462,760 bytes、全tensor F32。
- reranker: `cl-nagoya/ruri-v3-reranker-310m`、revision `bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`、safetensors 1,260,829,436 bytes、全tensor F32。
- 入力はテスト内の合成日本語。queryは「東京の人口」、文書は「東京は日本の都市です。」「京都も日本の都市です。」。query batch=1、embed/reranker batch=2、bucket=128。
- 各経路で `[成功, Forward失敗, Pool失敗, Eval失敗, Readback失敗]` を6回繰り返し、最後に成功を1回。計31回×3経路=93回。各経路の最初の成功も記録し、warmupとして除外していない。

Xcode／Metalの版は当日の測定ログに出力していなかったため、2026-09-21の文書修正時に、測定と成功出力比較に使用したCargo target directoryの保存済みビルド成果物から遡及確認した。[確認記録](issue-303-cleanup-toolchain.json)には、元ファイルのSHA-256・更新時刻・変更時刻と必要な抜粋を保存した。CMakeの記録はXcode内のコンパイラの使用とApple clang `21.0.0`（`clang-2100.3.34.2`）を示し、測定前に生成されたMetal kernel `copy.air`にも上記のMetalコンパイラ版が埋め込まれている。

Xcodeの版・build番号は、そのコンパイラを含むbundleの`version.plist`と照合した。同ファイルとclang実体の変更時刻は2026-09-15で、測定前だった。Metal assetの版とToolchain識別子も、測定前の更新・変更時刻を持つインストール済みファイルと照合した。修正時に実行した`xcodebuild -version`／`xcrun metal --version`の結果は補助確認であり、測定時に採取した出力ではない。後から再生成された共有`mlx.metallib`だけを測定時の証拠にはしていない。この追記に伴う推論・メモリ測定の再実行は行っていない。

```sh
/usr/bin/time -l cargo nextest run --locked --lib --features test-mlx \
  --run-ignored=ignored-only --test-threads=1 \
  --success-output=immediate real_model_cleanup_memory
```

実行時は独立したCargo target directoryを指定した。nextest出力・全93サンプル・time出力は[生ログ](issue-303-cleanup-memory.txt)に保存した。テスト8.116秒、コマンド全体8.78秒、timeのmaximum resident set sizeは1,643,200,512 bytes。このRSS値はコマンド全体の最大値で、サンプルごとのMLX計数とは別の指標である。

## メモリ推移

単位はbytes。各サンプルは元のcleanupが戻った後に採取する。測定のための追加clear/synchronizeは入れていない。`active`はMLXのactive memory、`cache`はMLX allocator cache、`peak`は各試行前にリセットしたMLX peakであり、compile cacheの占有量を独立測定した値ではない。

| 経路 | active最小–最大 | cache最小–最大 | peak最大 |
| --- | ---: | ---: | ---: |
| query | 1,258,676,636–1,268,124,600 | 0–9,045,024 | 1,514,531,016 |
| embed batch | 1,258,676,636–1,279,931,856 | 0–22,041,652 | 1,770,384,584 |
| reranker | 1,261,047,840–1,282,307,152 | 0–7,865,368 | 1,772,754,776 |

Forward/Pool失敗後のactiveは、全6サイクルでquery・embed batchが1,258,676,636 bytes、rerankerが1,261,047,840 bytesに戻り、cacheは0だった。成功・Eval/Readback失敗後は範囲内で変動した。各経路の最初と最後の成功のactiveは、queryが1,268,124,600→1,265,764,792、embed batchが1,258,676,636→1,258,676,636、rerankerが1,282,307,148→1,279,947,852 bytesだった。重み・mask等のモデル常駐分を含み、各試行の完了直後の計数だけから差分をリーク量とは判断しない。

## 検査と限界

テストは全試行のcleanup回数が1であること、4種の注入エラーが呼び出し元へ届くこと、成功結果が有限で同じインスタンスの最初の出力との差が全要素`1e-5`以下であることを検査した。すべて成功した。通常checkも371 passed / 28 skipped、doctest 1 passed、clippy・fmt成功。この実モデル測定は通常checkのignored対象を別途実行したものである。

Arrayの解放がcleanupに先行する順序はDropを記録する通常の回帰テストで検査する。実モデル観測はその代替ではない。Forwardは第1層の後、Poolはmask/CLS slice生成後、Eval/Readbackは実処理の成功後に人工的なエラーを返す。自然発生するGPU例外、panic、backend abort、長時間・多shape、5,000 pair、旧版とのメモリ改善率は未検証。旧版との成功出力比較は次節に示す。この反復内の比較だけで旧版との一致を判断していない。


## 修正前後の成功出力比較

同じホストで、開始版`1a53b0dc1859b0a432b637faf79b3287304b8875`と上記manifestの変更後ソースを個別のプロセスで実行した。入力・モデルrevisionは反復テストと同じ。モデル生成直前にMLXの乱数seedをそれぞれ42に固定し、debugビルドで比較した。

| 出力 | 要素数 | 最大絶対差 |
| --- | ---: | ---: |
| query embedding | 768 | 0 |
| 2文書のembedding | 1,536 | 0 |
| 2 pairのreranker score | 2 | 0 |

出力JSONは修正前後でbyte単位でも一致した。SHA-256はいずれも`fd244d1f6f7e3bbbd2603204a191663ba91761181b8364438e3167b6086ba64d`。[一致した全出力](issue-303-success-output.json)と[比較プログラム](issue-303-success-probe.rs)を保存した。reranker scoreは`[0.010769708082079887, 0.0052141048945486546]`。

再現には比較プログラムを一時Cargo packageの`src/main.rs`として配置し、次の依存を設定する。比較する各checkoutのCargo.lockをコピーして`cargo run --offline`を実行し、stdoutを保存する。両方の生成lockで既存パッケージの版が変更されていないことを確認した。

```toml
[package]
name = "cleanup-output-comparison"
version = "0.1.0"
edition = "2024"

[dependencies]
rurico = { path = "/path/to/checkout" }
mlx-rs = "0.32"
serde_json = "1"
```

seed固定は、既存rerankerが未ロードのbiasをランダム初期化する挙動（#300で整理中）の差を比較に混ぜないためでもある。#303ではモデル構造・weight loaderを変更していない。この比較は短い入力とbucket=128の成功経路を対象とし、全モデル・全shapeや公式実装との一致を主張しない。既存の大入力fixture検査・5,000 pair検査は今回は再実行していない。
