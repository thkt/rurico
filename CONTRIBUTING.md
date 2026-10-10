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

`scripts/check.sh` と `just check` はRustの`test` job相当の検証を行う。
Python参照環境の導入・整合検査・既存テストはCIの独立したstepで維持する。`coverage`（変更行95%以上）、
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
python3 -m unittest discover -b -s scripts -p 'test_*.py'
bash scripts/test.sh
cargo test --locked --doc --workspace --features test-support,test-mlx,smoke
```

`cargo nextest run` は doctest を走らせないため、`cargo test --doc` を別途実行する。nextest 未インストールの場合は `brew install cargo-nextest` または `cargo install cargo-nextest --locked`。

`test-support,test-mlx,smoke` を通常CI・標準check・`just test`で有効にする。
現行の `--all-features` はこの3 featureをすべて有効にする指定で、
MLXをCPU backendへ切り替える指定でもignoredテストを実行する指定でもない。
clippyの `--all-targets --all-features` はコンパイル検査であり、テストの実行を代替しない。
coverageも同じfeatureを使う。既存の除外regex・95%の閾値は維持し、
追加対象を通すための除外は設けない。

### download・probeの合成process検証（Issue #320）

`src/model_probe/tests.rs` の大容量stderr、孫のFD保持、短いtimeoutは、それぞれ
pipe詰まり、直接child終了後のEOF未到達、実行期限超過という異なる失敗経路を守るため維持する。
保持範囲外のACKと切り詰めた原因の分類は公開probe入口から確認する。
`src/owned_process/tests.rs` は同じ本番のcollection・終了経路を使い、両streamの
256 KiB上限、継続出力中の期限確認、子孫の継続書込み停止、process group消滅とparentのthread数を確認する。
所有groupの外にwriterが残る合成pipeでは、回収不能を2秒の猶予後にエラーとして返すことも確認する。
readerの途中I/O失敗は実pipeを包むreaderに注入し、同じ回収経路で停止することと、
別groupのwriterが動き続けることを確認する。別Mutexや合成exit statusだけを回収の証拠にしない。
このfixtureのshellがsleepを起動する際のsignal競合も、本番collectorの再試行で回収する。
失敗時はchild reap・group不在を分けて表示し、元のreaderエラー分類も確認する。

`crates/rurico-ffi/src/process.rs` はmacOSの実childを `waitid(WNOWAIT)` でzombieに保ち、
signal 0のEPERMをgroup不在と誤認せず、SIGKILLのEPERMも消さず、reap後にだけ不在となることを確認する。
[XNU xnu-12377.1.9のkillpg1](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/bsd/kern/kern_sig.c#L1579-L1624)
はzombieをgroupのsignal対象から除外し、存在するgroupにsignal対象がなければEPERMを返す。
collectorはEPERMで回収を中断せず、同じ2秒の猶予内にreap・group不在・両EOFを確認する。
確認できない場合は失敗を返し、診断にはSIGKILLの失敗と回収状態を含める。
このFFI検証だけで本番collectorの回収を証明せず、上記の反復process検証と合わせて確認する。
回収不能なpipeの検証はreader失敗なし／PermissionDeniedありの両条件を使い、
回収猶予超過時にも元のI/Oエラーの分類と原因が残ることを確認する。
同じ本番collectorで実OSのsignal操作を観測し、child reapとgroup不在を確認した後は
SIGKILL・signal 0を再送信せず、開いたpipeを猶予超過までdrainすることも確認する。

`src/model_io/download_process/tests.rs` はtest binaryをre-execし、HF I/Oだけをfakeに置き換える。
本番のdispatch/result送信・collectionを通して、path返却、途中失敗、遅い書込み、hang、
未登録時の再帰防止と不正requestの診断を確認する。timeoutを毎回別のfixtureで5回繰り返し、今回のworkerの開始とslowの書込みを要求した上で各groupが消滅し、
未完了ファイルのsizeが返却後に増えず、parentのthread数が開始時に戻ることを確認する。
孫FD保持も5回繰り返す。この2件は各テストだけを選択してtest binaryをre-execし、
隔離したprocess内で反復前後のthread数をmacOSの `ps -M -p <PID>` で等値比較する。
Cargoの並列実行でも他テストのthread増減が混ざらず、漏れの相殺や誤判定を防ぐ。
外側のテストは子の実行件数・終了結果も確認する。全体の並列実行は維持する。
既存cache・他利用者の未完了ファイルは削除しないことを合成fixtureで確認する。
同じ固定revisionのcache miss後に別consumerがpointerを公開し、HFの本番公開primitiveの
symlink直前で子processを停止する検証も標準checkに含める。正常pointerの内容・inode、
正常blob・他consumerの未完了ファイルの保全とgroup回収を確認する。
HTTP/Xet共通の公開primitiveを使うが、通信やHF finalize全体の実行検証ではない。

これらはモデル不要・非ignoredで、設定済み `bash scripts/check.sh` の標準nextest対象に含まれる。
実行対象を絞る場合は以下を使える（全体checkの代替にはしない）。

```sh
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke \
  -E 'test(model_probe::tests) | test(model_io::download_process::tests) | test(owned_process::tests) | (package(rurico-ffi) & test(process::tests))'
```

thread数観測の隔離を変更した場合は、標準checkのnextestに加えて、ホストでCargoの
並列実行を確認する。coverageもCargoのtest harnessを使うため、nextestだけでは代替できない。

```sh
cargo test --locked --lib --features test-support,test-mlx,smoke repeated_ -- --test-threads=4
```

上限とOS上の保証範囲、dispatcher登録の移行手順は[README](README.md#downloadprobeの終了管理)を参照。
合成検証は実networkの通信・HF cache全経路、実consumerの移行、推論・検索品質や
[#364](https://github.com/thkt/rurico/issues/364)の5000ペア実測を確認しない。
実行前に開始commit・変更差分・Cargo.lock、実行後にツールチェーン・コマンド・終了コード・
対象テスト名と結果を既存の検証記録へ残し、未実施の実network検証を別に明示する。

### vector byte-bindの実SQL検証（Issue #368）

`src/storage/tests.rs` は本番の `ensure_sqlite_vec` を使い、登録後に開いた2つの
in-memory connectionで、合成f32 vectorを `bytemuck::cast_slice` によりBLOBとして挿入・検索する。
明示したL2 metricで、query `[1, -2, 0.5]` に対する距離が2・3・5・10となる候補を使い、
上位3件のID・順序・距離と保存されたlittle-endian bytesを固定する。
sqlite-vecが拒否する不正BLOB長・空BLOB・次元不一致は挿入と検索の両方で確認し、
拒否後に候補が残らないことと正常な挿入・検索への復帰、空対象の空結果も確認する。
製品のvalidationやconsumerのschemaは追加しない。
既存の登録idempotence・CREATE検証をこの通し検証へ統合し、FFIのversion検証は維持する。

標準checkで実行されるモデル不要テストであり、個別の再実行は次のコマンドを使う。
ビルドは引き続きMLX-only構成で行う。

```sh
cargo test --locked --lib --features test-support,test-mlx,smoke storage::tests
```

検出力の確認はホスト上の一時コピーで行い、未commitの変更も含む今回のcheckoutをコピーする。
最初に上記コマンドで正常版が成功することを確認する。その後、`src/storage/tests.rs`に
次の変更を1つずつ適用し、毎回同ファイルを正常版から復元する。

- 挿入側で各f32の4 bytesを逆順にする。保存bytesのassertionで失敗することを確認する。
- `neighbors`の取得結果を逆順にする。ID・順序・距離のassertionで失敗することを確認する。
- `KNN_SQL`を、元の近傍取得SQLをmaterialized CTEへ入れ、外側のSELECTで
  `ORDER BY rowid`へ並べ直すSQLに変更する。近傍取得そのものは成功させ、
  上位3件の順序を比較するassertionで失敗することを確認する。
  vec0のKNN queryへ直接`ORDER BY rowid`を指定するとSQLiteが拒否するため、
  そのエラーを順位assertionの検出実績とは扱わない。

各版を上記コマンドで実行し、最後に正常版を復元して成功することを確認する。
ビルドや環境の失敗は回帰の検出と扱わない。開始commit・統合したmain・対象差分と
Cargo.lockのhash、Rust/Xcode/Metalの版、コマンド、終了コード、stdout/stderrをcheckout外に保存する。
標準checkはこれらの変異を適用しないため、その成功を検出力確認の実施済み証拠にしない。
この検査は実モデルの意味表現、consumerのCandidate pipeline、migration、検索品質を保証しない。

### 文書分割の回帰検証

`src/text/tests.rs` は本番の `split_text` を直接呼び、段落→行→UTF-8文字境界の
優先、期待断片、原文との連結一致、byte上限を標準checkで確認する。
`max_bytes < 4` の入力全体を返す例外は別に確認し、この条件に上限を要求しない。
段落・行優先を削除する一時変異の検知確認は
[再実行手順と記録](docs/research/issue-360/split-priority.md)を参照する。

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
# 登録済みbinaryでcacheを準備してから、実tokenizerの既存assertionを実行する。
cargo run --locked --features smoke --bin mlx_smoke -- prepare-cache
cargo nextest run --locked --features test-support,test-mlx,smoke --run-ignored=ignored-only \
  -E 'test(g_001_real_tokenizer_extract_prefix_tokens) | test(regression_prefix_merge_standalone_vs_full_tokenization_diverges)'
```

lib test harnessはmain登録を持たないため、実tokenizerの2件はdownloadを呼ばず
検証済みcacheを読む。`prepare-cache`は既存`mlx_smoke`のmodeで、専用helperの配布ではない。
cacheがない場合は準備コマンドを示して失敗し、合成tokenizerへの置換やassertionの緩和は行わない。

`src/embed/tests.rs` の実tokenizerを使う2件と、`src/reranker/tests.rs` 内
`mlx_runtime_tests` モジュールのテストがこのカテゴリに該当する。

長文plannerは `src/embed/processing/tests.rs` で製品の関数を直接呼び、通常checkで検証する。
合成WordLevel tokenizerの異なるtoken IDを使い、prefix・overlap・末尾・文書順と、
上限を超える候補の縮小後もoverlapが保たれることを確認する。モデル取得は不要だが、
ビルドは引き続きMLX依存である。この検査は実tokenizer固有のprefix境界mergeや
実モデルの数値一致を保証しない。それぞれ上記のignoredテストと
[実モデルの受入手順](#重みの読込み検証)で別に確認する。

前処理のclone削減、長文の不要なID/maskコピーの削減と、独立したCPU比較は
[Issue #311の検証手順](docs/benchmarks/issue-311/README.md)を参照する。
本番の組立テストは借用sliceからのpaddingも確認し、prefix境界mergeの合成検証を追加した。
CPU比較と実モデルの`measure-records`は通常checkには含まれない。

### 推論境界の回帰検証（Issue #363）

演算・製品経路・固定モデルの確認結果と隔離条件は[検証記録](docs/benchmarks/issue-363-validation.md)を参照する。

通常checkでは、製品の `execute_document_chunks` を通してbucket・sub-batchの出力を
元の文書／chunk順へ戻す。異なる入力tokenの識別値をforward境界から返し、3文書・10chunk、
複数bucket、`token_budget=383`（128で割り切れない値）、最後の端数とpause回数を確認する。
これはMLX forwardを置き換えた境界検証であり、実モデルの数値成功には数えない。
同じ組立境界で、先行文書の成功後にreadback行が欠落した場合のエラーと、
後続forwardの元のエラー保持・以後のforward抑止・失敗時のpause抑止も確認する。
LazyRerankerとoptionsのdefault fallbackはspyで内容・順序・エラー・呼出し回数を確認する。
既存implementorへoptionsの実装を強制しない。

`MockEmbedder` / `MockChunkedEmbedder` は入力内容とprefixを無視する位置依存one-hotの
fixture producerで、batchとsingletonの値が一致する保証はない。既存のmock契約は維持する。
入力委譲の検査にはspyを使い、mockの出力を本物のEmbedderの数値・意味・入力順の証拠にしない。

次のモデル不要MLX検証はignoredのため、通常checkとは別にsandbox外のApple Siliconで実行する。
同じ変更版をビルドし、終了コードと対象テスト名・PASSを保存する。

```sh
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored=ignored-only --no-tests fail --status-level all --final-status-level all \
  -E 'test(embed::pooling::tests::mlx_runtime_tests::) | test(forward_evaluates_finite_values_and_masks_padding) | test(forward_truncates_oversize_input)'
```

poolingは非対称hidden値とu32の0/1 maskで手計算したmasked mean→L2を比較し、
ゼロnorm・全zero mask・大きいshapeの既存条件も検証する。
forwardは合成重みの小さい2層モデルでglobal／local attentionをevalし、有限性と
padding ID／長さを変えても有効tokenの値が一致することを確認する。
truncate検査もeval後の有限性まで確認する。これらは公式重みとの数値比較を保証しない。
[公式比較 #307](#公式実装との数値比較利用側の検索評価issue-307)を複製せず、その固定条件と結果を別の根拠として扱う。

実際のMLX readbackから結果組立までと、通常／計測APIのoptions・端数・pauseは、
[options付き推論の計測 #306](#options付き推論の計測issue-306)の `measure-records` を再利用する。
W1/W2/W3の既存fixtureを再生成せず、同じbinaryのrawと再生成summary、source／lockfile／モデルの
識別と負荷条件を保存する。過去の[#306の結果](docs/benchmarks/issue-306-metrics.md)は当時の版の証拠であり、
今回の変更版の実行結果に読み替えない。短時間の `measure-overhead` だけでは長文・複数chunkの検証を完了しない。

### 重複検証の整理とreranker遅延の計測（Issue #364）

通常checkでは、固定budgetの4値、probeエラーの分類・message・typed source、
両kindのprobe-env/path解決とsnapshot symlink、partial deleteのNotFoundと残り2ファイルの削除を確認する。
エラー変換は`model_init`、probe-env/pathは`model_lifecycle`のテストが所有する。
rerankerの非空入力の委譲・pair組立は模擬score境界で、dispatchは本番plannerを直接呼んで確認する。
非空入力の委譲・pair順・エラー保持、ログのbatch/bucket/sub-batchの各fieldを確認する。
公開APIの空入力は、weightsをロードしない極小の未学習MLXモデルとpoisoned lockで確認する。
非空入力で同じlockのエラーになる対照を置き、空入力がlockを取得しない配線を検証する。
MLX構造体の初期化を含むが、公式モデル推論や検索品質の成功ではない。

実モデルのsingleton・batch入力順・降順rerank、公開APIの空入力とdispatchログを
1回のロードで検証するテストと、
5000ペアのsub-batch/OOM回帰検査はignoredのまま維持する。固定revision
`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`のruri-v3-reranker-310mをキャッシュしてから、
変更版のApple Siliconホストで次を実行する。既存の数値条件・timeoutを緩めない。

```sh
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored=ignored-only --no-tests fail --status-level all --final-status-level all \
  -E 'test(reranker::tests::mlx_runtime_tests::)'
```

50ペアの3 warm-up＋30回測定は`benches/reranker_latency.rs`へ移した。
`p50 > 0`を性能判定に使わず、raw wall timeとp50/p95、基準版との比を保存する。
これは固定の`("query", "doc")`を50回並べたbucket 128入力のpublic `score_batch`計測で、
モデルロード時間は別項目に記録する。毎回の50件・有限性・[0,1]も確認する。
新しいSLA・許容差は設けず、比だけで成功判定や全入力の性能改善を主張しない。

ホストで、同じ端末・電源/熱状態・背景負荷・toolchain・lockfile・release flags・model cacheを使う。
ビルドと他の重いCPU/GPU処理は測定枠外にする。端末状態が変わった結果は同条件比較に使わない。
context JSONには以下の文字列を入れる。値は実際のコマンド出力・観測で埋め、例の説明文を使わない。

```json
{
  "source": "開始commitと測定対象source snapshotのSHA-256",
  "machine": "端末識別子（比較中は同じ値）",
  "chip": "sysctl -n machdep.cpu.brand_stringの出力",
  "os": "sw_versの出力",
  "rust": "rustc -Vvの出力",
  "cargo": "cargo -Vの出力",
  "xcode": "xcodebuild -versionの出力",
  "metal": "xcrun metal --versionの出力",
  "lockfile_sha256": "shasum -a 256 Cargo.lockのhash",
  "build": "共通のビルドコマンド（実行時の入出力pathを除く）、target、RUSTFLAGSとprofile設定",
  "load_conditions": "電源、熱状態、背景負荷とmodel cacheの条件"
}
```

出力はcheckout外の未使用pathを指定する。benchmarkは既存ファイルの上書きとモデル取得を行わない。

```sh
cargo bench --locked --bench reranker_latency -- BASE_CONTEXT.json BASE.json
cargo bench --locked --bench reranker_latency -- CURRENT_CONTEXT.json CURRENT.json BASE.json
```

前者は基準source、後者は変更sourceで実行する。基準はIssue #364開始版
`16e5ca97e4079917b7f16b48c7dc003ffdd77407`の一時コピーとし、今回の同じbenchmarkファイルと
Cargoの`[[bench]]`定義だけを追加する。製品・テスト・既存fixtureは変更しない。
両側のソースと差分、Cargo.lockのhash、stdout/stderrと終了コード、
開始/終了時刻・実行順・測定中の端末/負荷の観測を保存する。
変更版は未commit差分・未追跡のbenchmarkも含むsnapshotで特定する。
比較器はsource以外のcontextとworkload/modelの一致を要求し、保存した30回のrawから比を計算する。
contextは実行担当が記録する前提で、機械的一致だけでは端末状態やsource申告の正しさを証明しない。
基準版にも同じ測定コードを入れるため、測定器の追加そのものを製品の改善に数えない。

同じホスト条件で整理前後の対象reranker runtimeテストを実行できる場合は、
ロード/score_batch/子プロセスの実観測と所要を各版の終了コード・対象名とともに残す。
コード上の呼出し数と実測を区別し、測っていない時間短縮を主張しない。
旧版の件数成功や[#311の実測](docs/benchmarks/issue-311/README.md)を変更版の成功へ読み替えない。
benchmarkと上記ignored検証は標準checkに含まれず、同じ変更版のホスト追加検証が必要である。
測定結果と5000ペアの未完了条件は[Issue #364の検証記録](docs/benchmarks/issue-364/README.md)を参照する。
検索品質はamici、公式モデル数値比較は[#307](docs/research/issue-307/README.md)の範囲に残す。
by-value API署名は所有権の検査であり、readback回数やdrop-before-clearの実測ではない。

### `mlx_smoke` smoke テスト

`mlx_smoke` 統合テスト (`tests/mlx_smoke.rs`) と同名 binary (`src/bin/mlx_smoke.rs`) は `smoke` feature の背後にある。実推論モードとignoredの統合テストは、キャッシュ済みのruri-v3モデルとApple SiliconのMLX runtimeを要する。記録の単体テストと `summarize-records` はモデルをロードしない。標準CIとcheckは `smoke` featureを含め、閾値・record/集計・mode・比較器の単体テストとモデル不要のCLI統合テストを実行する。実モデル検証はホストで別に実行する。ignoredの統合テストを実行する場合は、事前に対象モデルをキャッシュしてから:

```sh
# ローカルHF cacheにruri-v3モデルが必要
cargo nextest run --workspace --features smoke --test mlx_smoke --run-ignored=ignored-only
```

通常laneの入口は `bash scripts/test.sh`（引数によるfilterの転送なし）。
呼出しごとにlocked依存と上記3 featureでCargo metadataを取得し、そのsnapshotをビルド入口にも渡す。
テストbinaryを一度ビルドし、そのbinary metadataと同じCargo metadataを一覧取得・実行の両方に渡す。
metadataは呼出し内の一時ディレクトリに保存し、成功・失敗時に削除する。工程をまたぐ保存は行わない。
nextestのJSON一覧を実行と同じfeature・ci profile・ignored条件で検査し、
binaryと統合の各suiteが存在し、モデル不要テストが各1件以上あり、filter/ignoreで除外されていないことを確認する。
既存6件の実モデル統合テストはignoredのままかを実行前に確認する。
一覧の検査失敗・0件・nextest失敗は非ゼロ終了となる。実行ログには対象名・PASS/SKIPと総件数を表示する。
モデル不要テストの追加は件数の固定更新を要しない。実モデルテストを追加・改名する場合は、
`scripts/verify-smoke-tests.py` のモデル対象一覧とテストのignored条件を照合する。
この選択検査は推論・数値一致・性能の証拠ではない。

モデル不要laneの検出力を再確認する場合は、一時コピーで
`src/bin/mlx_smoke.rs::workload_ratio` のbatch/sequentialの除算を逆にする。
次の対象限定実行が閾値判定のassertで失敗し、元へ戻すと成功することを、終了コードとログで確認する。
通常laneと同じfeature・profile・ignored条件を使う。モデルDLやGPU実推論は不要だが、ビルドはMLX依存のままである。

```sh
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored default --bin mlx_smoke --test mlx_smoke --no-tests fail --status-level all --final-status-level all
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
Python参照環境の通常インストール・依存整合・厳密な導入版・モデル不要テストもCIの`test`ジョブで検証する。
[Python環境の検証手順](docs/research/issue-307/README.md#ホストで実行する)を使い、実モデル比較とは区別する。
[準備時点の記録](docs/research/issue-307/report.md)から測定状況と未確認範囲を確認できる。

### 検索意味とquery planの設計比較（Issue #314）

検索時のphrase・短語展開と構成識別の分担は[Issue #314の設計比較](docs/research/issue-314/README.md)を参照する。
そのSQLite研究検証はPython標準ライブラリで実行できる。製品入口との照合は通常のRust searchテストに含まれるが、
amiciの固定版parserとround-trip確認は標準checkの対象外なので、同ページのホスト手順で別途確認する。

### 構成識別の設計例（Issue #315）

embedding・FTS構成の保存と照合の未採用案は、[設計報告と実行例](docs/research/issue-315/README.md)を参照する。
記録の決定性・役割別照合・情報不足の扱いは、そこに示すPython標準ライブラリの検証で確認する。
これはモデルを実行しない設計例で、製品APIの検証や標準checkの代わりにはならない。
Rustの検証は引き続き上記のMLX構成で行う。

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
# 新しい記録用ディレクトリを使う
mkdir /tmp/rurico-306-evidence
RUST_LOG=off target/release/mlx_smoke measure-records > /tmp/rurico-306-evidence/raw.jsonl
# 生recordから集計を再生成（モデル不要）
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
取得不能な値・未計測のRSS/Metalメモリは`null`で、推定しない。
現行の`build_flags`はbuild時にprofile・flags・features・compiler・manifest/config hashを保存する。
旧recordの`build_flags=null`はbuild条件が未記録であり、現行の版間比較では拒否する。
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
通常checkはsmokeのモデル不要テストを含むが、実モデルの検証は完了しない。フルcheckとCIを同じheadで確認し、
公開可能な小さなraw record・summary・build条件・未確認事項を `docs/benchmarks/` に保存する。
今回の実測結果と制約は [Issue #306の検証記録](docs/benchmarks/issue-306-metrics.md) を参照。

### 性能判定と基準rev比較（Issue #359）

[Issue #359](https://github.com/thkt/rurico/issues/359) の性能判定は、既存の#306のrecordと交互反復を使う。
`calls[].readback_elements` は本番のhost slice取得ごとの実要素数で、配列の長さがreadback回数。
各callでforwardごとの `batch_size × hidden_size` と照合する。追加readback、未poolingの大きなslice、
欠損した観測値は検証失敗となる。空入力の実測は `[]`、旧recordの欠落は `null` として読み、ゼロ回と混同しない。
`readback_shape` は各warm batch試行の実測回数・要素数と期待値をstderrへ表示する。
実forward・readback・cleanup・pauseの順序、stdoutの1試行1record、sub-ms精度は従来どおり。

`measure-baseline` はbatch/sequential効率、padding、規模に対するforward時間のR²を検査する。
ゼロ分解能のwall/forward時間、欠損metrics、NaN/±Inf、当てはめ不能な一定系列は達成として受理せず、
理由を伴う判定不能として失敗する。tokenize/chunk_planの未分離時間は引き続き `unmeasured`。
R²は規模への当てはまりであり、速度や絶対遅延を保証しない。

既存の閾値と適用条件は変更しない。短いbucketのみのworkloadに効率のprimaryを適用し、
W1の飽和bucketとW3の混合bucketの速度はprimaryの保証対象外と明示する。
W1/W3のbatch/sequential比が100でも、paddingとR²が適用条件を満たせばprimary違反はない。
未達時だけ出るsaturated/aspirational診断が0件でも成功できる。tier振分けは合成入力で検証し、
実機テストに診断の存在を要求しない。primary成功を全workloadの性能目標達成とは読まない。
絶対SLAが必要な用途では、対象機種・負荷・許容値をユーザーが決める後続判断が必要になる。

同じ条件の基準revとの遅延変化は、効率とは別にraw recordから比較する。
比較用binaryは `bash scripts/build-smoke.sh` でbuildする。引数を転送しない固定の
`cargo build --locked --release --features smoke --bin mlx_smoke` を使い、buildscriptから見えない
inline `--config` の追加を受け付けない。環境変数でのrustc wrapperも拒否する。
直接のCargo buildは従来どおり推論・計測・テストに使えるが、比較対応入口を通していない
`build_invocation=null` のrecordは比較を拒否する。次のモード自体はモデルやMetalを初期化しない。

```sh
target/release/mlx_smoke compare-records /tmp/base-raw.jsonl /tmp/current-raw.jsonl > /tmp/comparison.jsonl
```

`revision_latency` はmethodごとに両版のwall分布、対応sequence、context、
`current_over_baseline` と中央値の増加有無を出す。`batch_sequential_efficiency` は各版の方式比を別に出す。
両方式がともに100倍遅くなれば、方式比が変わらなくても遅延増加は表示される。
中央値の増加有無は観測であり、新しい許容差や統計的な速度保証ではない。

入力hash・workload・options・通常/計測・methodのgroupを対応付け、model/tokenizer revision、
machine/chip、OS、Rust/Cargo、Xcode/Metal、lockfile、build条件、cache方針の不一致や欠落を拒否する。
比較する各版は同じ条件で再buildしてから記録する。build条件はCargo build時のprofile、opt level、
debug、target、encoded Rust flags、features/target cfg、profile override、build時のrustc版を
`build_flags` に保存し、profileのLTO/codegen設定を含むCargo.tomlと探索対象Cargo configの内容hashも含める。
recordは信頼する標準Cargoと上記の比較用build入口で生成する。手編集した記録や独自ツールによる改変を証明する仕組みではない。
commit・差分・実行ファイルhashは版の識別に使い、
異なる値でよい。group内でのcontext混在、対応する方式やgroupの欠落、ゼロ時間も拒否する。
旧recordの `build_flags=null` を同条件と仮定して比較しない。記録を補完・書換えず、必要なら両版を再測定する。
model/tokenizerのrevisionはcache lookup時の識別で、ファイル内容のhash照合や測定中の背景負荷の不在は
raw recordだけでは証明しない。#306のモデル内容照合・負荷監視手順も併用し、ばらつきと未確認条件を残す。

標準checkは以下のモデル不要のsmoke単体・CLI統合テストを含み、clippyはビルド確認に限る。
後半2つのignoredテストは別の実モデル検証で、通常checkには含めない。
同じ変更版と固定Cargo.lockでホストが実行する。後半2つのignoredテストには、
キャッシュ済みdefaultモデルと対応Metalが必要。

```sh
cargo test --locked --features smoke --bin mlx_smoke
cargo test --locked --features smoke --test mlx_smoke summarize_records
cargo test --locked --features smoke --test mlx_smoke compare_records
cargo test --locked --features smoke --test mlx_smoke smoke_measure_overhead -- --ignored
cargo test --locked --features smoke --test mlx_smoke smoke_measure_baseline -- --ignored
```

短時間の `smoke_measure_overhead` は既存W2の先頭3文書とfixtureを再利用し、
default/nondefault、batch/sequentialの本番readbackと空入力の観測を検証する。
全W1/W2/W3は `smoke_measure_baseline` で確認する。既存のprimary未達も検出するため、
失敗時はreadback・欠損等の新しい不具合と、既存閾値の未達を診断から区別する。
#306で記録されたpadding/R²の未達は[過去の実測](docs/benchmarks/issue-306-metrics.md#テストと既存報告の訂正)として残し、
今回の成功とは扱わない。fixture・baselineを再生成して失敗を消さない。
検出力のホスト確認では、一時コピーで `src/embed/mlx.rs::forward_sub_batch` のreadback呼出しを2回にした版と、
`pooled` を `mlx_rs::ops::concatenate_axis(&[&pooled, &pooled], 0)` で倍の要素数にした版をそれぞれ実行し、追加readbackは期待回数の診断、倍のsliceは製品のpooled shape検証で失敗することを確認する。
元版を戻した同じテストの結果、対象差分hash・Cargo.lock hash・ツールチェーン・終了コード・stdout/stderrを
checkout外へ保存する。算出値だけの検査やテスト側のカウンターを本番観測の代用にしない。

開始版 `4a029333c3b2a64c9483c410d4d722377ae93bea` は開始時のローカル `origin/main` と一致した。
Issueが固定参照する `c8f250d60a5afb9944b9008d22ad6f4dda2d7103` の5ファイルは開始版で変更されていなかった。
旧性能結果は旧版・旧条件の証拠として維持する。新しい速度改善や実機成功はこの説明だけでは主張しない。
今回のreadback照合・故障注入・性能gate未達は[ホスト検証記録](docs/benchmarks/issue-359-verification.md)を参照。

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

### モデル種別の拒否条件（Issue #361）

[Issue #361](https://github.com/thkt/rurico/issues/361) のkind検証は、
`src/artifacts/tests.rs`で内部helperと公開`CandidateArtifacts::verify`の両方を確認する。
有効なbackboneのprefixに禁止headを1種類ずつ加えたembedと、
classifier・head.dense・head.normのうち1種類だけを欠くrerankerを使い、
`WrongModelKind`の対象kind、禁止／欠落の理由、該当prefixを検査する。
全prefixが揃う正常例、backbone不在、header破損は既存の別テストで確認する。
ここでいう有効なbackboneはkindのprefix条件を満たすことを指し、
config由来の全必須重み・shape・F32・data rangeの保証は、次節のloader検証が担う。
小さな合成ファイルを再利用し、モデルのダウンロードや追加の実モデルロードは行わない。

通常checkはこれらのテストを実行するが、製品コードを変異させた検出力の確認は行わない。
検出力を確認する場合は、変更後のソースを含む一時コピーをホストに作り、まず正常版を実行する。
コピー元は今回のcheckoutとし、commitだけをコピーして未commitの変更を落とさない。
MLX-onlyのビルド条件は通常checkと同じにする。

```sh
rurico_kind_copy=$(mktemp -d)
rsync -a --exclude=.git --exclude=target ./ "$rurico_kind_copy/"
cd "$rurico_kind_copy"
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored default --no-tests fail -E 'test(artifacts::tests::)' \
  --status-level all --final-status-level all
```

一時コピーの`src/artifacts.rs`だけに、次の変異を1つずつ適用する。
各変異の前に同ファイルを正常版から復元し、テストやfixtureは変更しない。

- `verify_model_kind`の`else if !require_reranker_keys && found`分岐を、
  その`WrongModelKind`を返すブロックごと削除する。
  下記のembed限定実行でhelperとcandidateの2テストが拒否を期待するassertionで失敗することを確認する。
- `RERANKER_KEY_PREFIXES`から`"head.norm."`だけを削除する。
  下記のreranker限定実行でhelperとcandidateの2テストが、
  normだけを欠くケースの拒否を期待するassertionで失敗することを確認する。

```sh
# head拒否分岐を削除したコピー
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored default --no-tests fail -E 'test(embed_kind_rejects_each_head_prefix)' \
  --status-level all --final-status-level all
# 正常版を復元し、norm必須prefixだけを削除したコピー
cargo nextest run --locked --workspace --features test-support,test-mlx,smoke --profile ci \
  --run-ignored default --no-tests fail -E 'test(reranker_kind_rejects_each_missing_head_prefix)' \
  --status-level all --final-status-level all
```

終了コードだけで検出成功とせず、各テスト名・該当ケース・assertionのログを残す。
ビルドや環境の失敗は変異の検出と扱わない。最後に正常版を復元し、最初のartifact限定実行が成功することを確認する。
開始commit、作業差分、Cargo.lock、Rust/Xcode/Metalの版、実行コマンドと結果を検証記録へ残す。
標準checkや旧版の成功を、この変異確認の実施済み証拠として扱わない。

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
