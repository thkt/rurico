# Issue #308 固定bucket表3構成の比較

[Issue #308](https://github.com/thkt/rurico/issues/308)の調査用入口。
製品APIは4区分を維持する。根拠の版、consumer影響、未実測事項は[report.md](report.md)を参照する。
UI媒体は不要で、`.dotagents.json`の`capture: null`は変更しない。
数値ログの保存はUI captureではない。

## ホストで準備する

Apple Silicon、対応するXcode/Metal、Rust 1.96以上、固定embedding/reranker310mのHF cacheを使う。
[標準setup/check](../../../CONTRIBUTING.md#テスト)は`cargo fetch --locked`と
`bash scripts/check.sh`のままである。以下は標準checkが実行しない追加の受入検証である。
sandbox内ではビルド・モデル観測を実行しない。

repo rootから、checkoutと旧runの外にある新規保存先を指定する。

```sh
python3 docs/research/issue-308/run.py prepare /tmp/rurico-308-prepared-new
python3 docs/research/issue-308/run.py measure /tmp/rurico-308-prepared-new \
  /tmp/rurico-308-context.json /tmp/rurico-308-evidence-new
```

`prepare`は未commitファイルを含む現在のcheckoutを3つの独立したsourceコピーへ保存する。
候補コピーだけでbucket表、分配配列、metrics配列を5/6要素にする。
既存libtestのログfixtureには未使用要素の0を追加し、固定4要素のassertionは
期待値を保ってslice比較へ変える。元checkoutのテストは変更しない。
準備完了時はHEAD・plan・新しく列挙したcheckoutのファイル集合とhash・候補source・binaryを検査する。
保存planは一度取得したbytesで内容を照合し、同じbytesのSHA-256をprepared記録へ保存する。
検査の成功後にだけprepared記録を書き込み、新しく取得したplanと記録の整合を確認する。
この再照合が失敗すればprepareは失敗し、残った記録を準備完了として受け入れない。測定前後には別途再観測する。
3つのrelease libtestを同じ`--locked --features test-mlx,smoke`でビルドする。
対象はlibtestだけであり、候補版の既存4区分テストやsmoke binaryを成功したとは扱わない。
Cargo targetはコピーごとに分け、同名binaryの上書きを防ぐ。
`--cache-base OLD_PREPARED`は旧targetを新しいtargetへ通常コピーして依存を再利用する。
旧source・targetは変更せず、現在のcrateを新しいsourceから再コンパイルする。

context JSONには`machine`、`ram_bytes`、`os`、`rust`、`cargo`、`xcode`、`metal`、
`build`（flags、ビルド並列数を含む）、`load_conditions`を指定する。
`model_dirs`は`embedding`と`reranker`をkeyにした絶対snapshot directoryである。
embeddingのrevisionは`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`、
rerankerは`bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3`を使う。
各directoryの`model.safetensors`、`config.json`、`tokenizer.json`のSHA-256を
[#307の固定manifest](../issue-307/results/numerical-rechecked/models.json)と照合する。
観測したcache pathも照合するため、RustのHF cache設定が同じsnapshotを指すようにする。
cacheや個人pathを含むcontext・生ログを公開資料へ自動転記しない。

測定中は他のGPU測定とビルドを止め、競合process、AC電源、熱状態と観測頻度を
`load_conditions`およびホストの実行記録へ残す。runnerは所有groupのRSS、pressure、swapを
約0.1秒、process名/CPUを約1秒ごとに記録する。温度や他processのGPU使用量は取得しない。
観測欠落を負荷がなかった証拠にしない。測定保護条件としてRSS 8 GiB超、pressure 1以外、
swap増分256 MiB超、30分の期限で停止する。製品の資源上限にはしない。
`--kinds`と`--cases`で測定を分ける場合は対象一覧を記録し、部分成功を全体の成功にしない。
重複選択と、選択したkindの対象caseが0件になる組合せは、測定前に拒否する。
SIGINT/SIGTERMと通常失敗では所有process groupを回収し、部分ログを残す。
SIGKILLでrunner自体を終了した場合の回収は未確認である。

## 同じ条件で観測する

3構成の実行順を独立processの3組で循環させる。
継続観測だけを別に行う場合は`--trials 1`を指定できるが、3組の性能比較を代替しない。
各processはモデルを一度ロードし、初回をcold、残りをwarmとして同じ入力を呼ぶ。
境界・同長・空入力は4回、混在・複数chunkは31回にして連続推移を記録する。
coldはモデルload後の初回であり、OS file cacheを消した起動ではない。
`--repeats N`と`--sustained-repeats N`は4以上を指定でき、同じcaseの全構成へ同じ値を適用する。
31回が何分の観測になるかは未測定で、長時間推移の十分性は実時間とRSS/Metal系列で判断する。
不足する場合は同じ条件の新規保存先で観測を延長し、古い結果を保持する。

embeddingは製品の文書batch APIとquery APIを使う。
文書はtoken budget 8192、forward後pause 1 msを全構成で固定する。
`--budget 256000`で標準budgetも比較できるが、同じrunの全構成で固定する。
これは比較入力の指定で、主対象optionsや機種の許容メモリを製品採用として決定しない。
rerankerは既存#309の観測器をreference modeだけで使い、標準budget 256000、全pair最大長方式を維持する。
長さ別分配・cache policy・precision・modelの変更を混ぜない。

実token長127/128/129、511/512/513、1023/1024/1025、2047/2048/2049、
4095/4096/4097、8191/8192/8193を、固定tokenizerによる実encodingで作る。
embeddingのlimitは文書plannerによる分割とqueryのtruncate、rerankerはpairのtruncateを通る。
長短混在と複数chunkの文書、rerankerの同長50件・混在50件・空入力も別に観測する。
入力は公開可能な合成文字列で、私的な本文・token列は使わない。
queryは文書の時間計測後に実行し、数値の受入対象にする。query単独の性能比較ではない。

`plan.json`は開始HEAD、Cargo.lockを含むsource hash、候補source、binary hashを持つ。
測定前後に元source・HEAD・候補source・binary・modelの不変性を照合する。
生記録のbuild signatureも構成間で照合し、compiler・flags・Cargo configの差を混ぜない。
`context.json`は環境と入力計画、`*.json`は入力・生出力、`*.monitor.json`はRSS/pressure/swap系列、
`*.status.json`は終了値とgroup回収、`summary.json`は診断と時間・Metal系列を持つ。
RSSはload・入力生成・query・非計測の製品の内部score_batch経路との照合も含むprocess範囲の標本最大で、
callのMetal peakと測定範囲が異なる。瞬間RSS peakは保証しない。
rerankerの照合はRerankerInner::score_batchを呼び、公開Reranker wrapperの空入力処理やmutex取得は通らない。
各callのMetal peakはreset後の文書/reranker呼出し範囲である。
embeddingのwallはpause・cleanup・組立を含む既存metrics、rerankerのwallはpair encodingを含む既存観測値である。
GPU kernel時間へ読み替えない。

## 担当AIが実測後に確認する

比較器は文書/chunk順・ID・次元768・有限性、同じprefix/overlapを含むtoken列、
shape/forward・padding・pauseの記録を照合する。
embeddingは現行版に対してcosine ≥ 0.99999かつ最大絶対差 ≤ 1e-5で判定する。
rerankerのlogit/score差は#309の比較器、順位・同点順は#307の比較器で診断として表示し、
logit・score各2.5e-5以下と順位一致・完全同点の元index順は、[人の回答原文](comparison-agreement.md)で
#308の比較条件として承認を確認した。製品許容差や区分採用にはしない。
差の実値、順位一致、完全同点の元index順を併記し、合意済みの製品許容差とは扱わない。
数値不一致でも生出力を保持する。`complete.json`は収集完了だけで、受入や採用ではない。

時間はcoldの3組と、processごとのwarm中央値3組の最小・中央値・最大を比べる。
構成間の差と構成内の観測幅を併記し、重なる範囲を安定した利益と断定しない。
Metal/RSSの増減と時間経過、実forward数を同時に確認する。
全区分のlocal mask容量推定273.1/337.1/341.1 MiBは、
`sum(bound**2) * 4 / 1024**2`というF32の計算値で、実peakとは区別する。
保存済み256件のpadding減21.0%/27.0%を速度改善率へ読み替えない。

担当AIは数値基準の合否、利益・退行・未解決のばらつき、未使用入力の限界と
採用/見送りの推奨を[報告](report.md)へ追記し、変更文書を既存の独立評価へ渡す。
採用を推奨する場合もADR-0009の後続判断、consumer影響、主対象options、
機種の許容メモリ、reranker許容差は別に確認する。自動採用・再index・consumer変更は行わない。

モデル不要の記録検証は既存checkのPython discoveryで実行される。
個別には`python3 -m unittest discover -b -s scripts -p 'test_bucket_tables.py'`を使う。

2026-10-12の主要8/9ケースと31回の継続観測は完了した。[実測集計](results/observed.json)と
[報告](report.md)に数値条件の結果、速度・メモリの費用、見送りの推奨と未使用入力を残した。
