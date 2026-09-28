# Issue #307 の実行手順

[Issue #307](https://github.com/thkt/rurico/issues/307) の公式数値比較とamici検索評価を、別の結果として記録する。
このディレクトリのコードは調査用であり、製品の許容差や既存fixtureを書き換えない。
実測結果・確認した相違・限界は [report.md](report.md) に置く。

## Python依存の固定と更新

[requirements.txt](requirements.txt) を、導入版と検証対象の唯一の正本とする。
Python 3.12.13 / macOS arm64向けに、推論ライブラリと推移依存を含む全runtime依存を固定する。
`reference.environment()` は隔離環境・Python版・一覧の全導入版を検査し、未記載の依存も拒否する。
venvの導入ツールであるpipだけは一覧一致の対象外とし、実行環境の記録にはその版も残す。
同じ版を参照コードの別テーブルへ転記しない。

[Issue #353](https://github.com/thkt/rurico/issues/353) に従い、主要推論ライブラリを優先して
互換する最新安定版を選ぶ。Torch 2.14.0、Transformers 5.17.0、Sentence Transformers 6.1.0を
今回の組合せとし、全依存の再照合・API変更・検証状況は[更新記録](validation-353.md)に残す。
これは過去の[実測環境](results/reranker-reference-refresh/environment.json)とは別条件である。
[report.md](report.md)と`results/`の旧依存版・モデルrevision・数値を新環境の成功へ読み替えない。

最新公開版を採用できない依存は次のとおり。上位依存の更新PRで配布メタデータを再確認し、
制約が解消されたらRenovateの`allowedVersions`も見直す。恒久的な更新除外にはしない。

| 依存 | 採用版 / 採用できない最新版 | 理由と再確認条件 |
| --- | --- | --- |
| huggingface-hub | 1.33.0 / 2.0.0 | [Transformers 5.17.0](https://pypi.org/pypi/transformers/5.17.0/json)と[Sentence Transformers 6.1.0](https://pypi.org/pypi/sentence-transformers/6.1.0/json)が`<2.0`を要求。いずれかの更新時に両方の制約を再確認する |
| mpmath | 1.3.0 / 1.4.1 | [SymPy 1.14.0](https://pypi.org/pypi/sympy/1.14.0/json)が`>=1.1.0,<1.4`を要求。SymPy更新時に再確認する |

[renovate.json](../../../renovate.json) はこのrequirementsの`pip_requirements`更新を
`Python reference environment`グループにまとめる。major/minor/patchを分割せず、
`automerge: false`とする。共有presetのCargo・Actions設定と除外pathは変更しない。
依存解決と通常インストール、整合検査、Pythonテストを通したPRを人が確認する。
推論依存・wrapperが変わる場合は、下記の固定実モデル比較も確認してから採用する。

更新時はPyPIの最新安定版とRequires-Distを再照合し、checkout外の候補一覧で
主要ライブラリの目標版を明示し、他の既存依存を現在版以上として解決する。
解決結果で目標版と全既存依存の非ダウングレードを確認し、追加の推移依存も含めて
厳密な版一覧へ反映する。制約を全部外すと、hubの最新版を優先して主要ライブラリを
大幅に古くする解決があり得るため、解決成功だけで最新化成功としない。
新しい隔離環境で下記の通常インストールを行い、`--no-deps`や検査の緩和で迂回しない。
環境検証の成功だけでは実モデル数値・検索品質の再測定を意味しない。

## ホストで実行する

Apple Silicon、対応Metal Toolchain、Rust 1.96以上を用意する。
標準検証は [CONTRIBUTING](../../../CONTRIBUTING.md#テスト) の
`cargo fetch --locked` と `bash scripts/check.sh` をそのまま使う。
本調査はignored testを明示実行するため、標準checkだけでは完了しない。
CIのtest・coverage・security・zizmorも、公開担当が同じPR headで確認する。
既存の必須`test`ジョブは、SHA固定の`astral-sh/setup-uv`でuvを導入し、
uvのmanaged Python 3.12.13でcheckout外に新規venvを作る。uvのActions cacheは使用しない。
その環境で通常インストール、`pip check`、
厳密な版検査と全Pythonテストを実行してからRust検証へ進む。path filter・失敗無視は使わない。
固定実モデルのダウンロードや推論はこのCIに含まず、次のホスト手順で別に確認する。

ホストにも[uv](https://docs.astral.sh/uv/getting-started/installation/)を用意し、
Python **3.12.13** の隔離環境をcheckout外に新規作成する。
以下はリポジトリrootから実行し、既存の研究環境を再利用・上書きしない。
Python依存は参照計算専用で、ruricoのCargo依存・CPU backendを変更しない。
`--managed-python`でuv管理のPythonを指定し、未配置なら自動取得する。
`--no-project`でproject探索を避け、`--seed`で通常install用のpipを用意する。
Pythonや導入Actionの更新時は、対象OS・architecture向けの配布と新規環境での導入成功を確認する。
Actionの`python-version`入力やlint成功だけでは、Python本体の取得成功を保証しない。

```sh
(
set -eu
reference_env=$(mktemp -d /tmp/rurico-307-reference-env.XXXXXX)
uv venv --python 3.12.13 --managed-python --no-project --seed "$reference_env"
"$reference_env/bin/python" -m pip install -r docs/research/issue-307/requirements.txt
"$reference_env/bin/python" -m pip check
"$reference_env/bin/python" -c 'import sys; sys.path.insert(0, "docs/research/issue-307"); import reference; reference.environment()'
"$reference_env/bin/python" -m unittest discover -s docs/research/issue-307 -p 'test_*.py'
printf 'Reference environment: %s\n' "$reference_env"
)
```

Issue #353では、上で表示された環境pathを指定してembedding・rerankerの数値比較を実行する。
結果の保存先も新規pathを使い、比較結果・source/環境/hash・未確認範囲を[更新記録](validation-353.md)へ追記する。
このIssueはamici検索pipelineを変えないため、検索品質の再測定は要求しない。

```sh
reference_env=/tmp/rurico-307-reference-env.XXXXXX # 上で作成した実際のpathへ置換
"$reference_env/bin/python" docs/research/issue-307/run.py numerical /tmp/rurico-307-numerical
# 別途、Issue #307の検索評価全体を再実行する場合だけ:
# "$reference_env/bin/python" docs/research/issue-307/run.py amici /tmp/rurico-307-search
```

HF/GitHub/Cargoへの接続が必要。sandboxでは実行しない。サーバー・ブラウザー・画面撮影は使わない。
実行中はsourceやcacheを編集せず、他のGPU推論・ビルドを止める。
数値比較に速度の基準はない。8192 tokenのFP32 eager attentionはCPUメモリと時間を要する。
不足時は失敗記録を残し、十分なメモリを備えたホストで同じ条件を再実行する。
精度、attention方式、入力上限を通過目的で変えない。

両コマンドとも既存出力ディレクトリを拒否する。実行失敗は`incomplete.json`と残存生成物に保持し、
再実行は別の新規ディレクトリを使う。例外・ビルド・モデルログは個人pathを含み得るため非公開とする。
途中の出力や終了0だけで数値一致・Issue完了とは扱わない。

## 数値比較の記録

`run.py numerical` は指定revisionのモデルを取得し、重み・config・tokenizer・モデルカードを
HFのcommit/ETagと内容で照合する。公式Llama tokenizerの読込みに必要な`tokenizer.model`・tokenizer設定も検証し、SentencePieceとprotobufを固定版で使う。追加wrapper設定のJSONもhashを記録する。
Rustが読むcacheとPythonのsnapshotが同じ実ファイルであることをテスト内で確認する。
開始commit、実行head、変更source、Cargo.lock、実行ファイル、機種/OS、Rust/Xcode/Metal、Python依存と
生成物を記録し、終了時にsource・モデル・実行ファイルの不変性を照合する。
追加のRust flagsは指定しない。source manifestはcommit未作成の検証コードも識別する。
MLXの版はCargo.lockと固定mlx-sysの参照を併読する（この開始版はmlx-rs 0.32.0、MLX v0.32.2）。

`inputs.json` は公開用の合成入力。`text` を`repeat`回連結してからprefixを付加する。
embeddingは文書prefixに加えquery/topic/空prefix、コード、日本語、短文、境界長を含む。
rerankerは同じqueryに対する候補群、同文候補、長文ペアを含む。
両モデルでsingletonの実長・bucket長と、先頭の短文/長文混在batchを比較する。
Rustの観測は既存`smoke` feature配下の`src/research.rs`の2件のignored testから既存のtoken helper、chunk planner、モデル、pooling、公開APIを呼ぶ。
モデルはshapeごとに破棄し、診断用の実長shapeで常駐mask cacheを増やし続けない。
同じrowの実長とbucket長が等しい場合だけ、各実装で1回の観測を両条件に使う。
異なるrow、異なるshape、混在batch、公開API、Rustと公式参照の観測はそれぞれ実行する。
現行入力の新規実行はembedding 23 batch・27行、reranker 12 batch・16行になる。
公開文書batchの返却件数は、IDと出力を対応付けて記録する前に入力件数と照合し、過剰・不足の双方を拒否する。

新規記録の`schema=2`では、各batchの`conditions`が`["exact"]`、`["bucket"]`、
または`["exact","bucket"]`。最後の形式は`/exact`の1観測を共有することを示す。
比較結果の`padding[].observations`に両条件の参照先、`same_observation`に共有の有無を残す。
共有時の差0は同じ出力との比較であり、独立したpadding検証・反復推論とは扱わない。
保存済みの`schema=1`は当時の二つの観測として読み、書換えや重複除去を行わない。

記録境界とbatch選択のモデル不要Rustテストは、ホストで次を実行する。
このフィルターでは上記2件の実モデルテストはignoredのままになる。

```sh
cargo test --locked --lib --features smoke research::
```

参照側の依存版はrequirementsのとおり。CPU、FP32、eager attention、eval mode、4 threads、
seed 0、deterministic algorithmsを固定する。Transformers 5ではModernBERTの`reference_compile`設定が
削除されたため、コンパイル済みmoduleを拒否し、Torchの`force_eager`でコンパイルを無効化する。
記録の`reference_compile: false`はこの実行条件を表し、削除済みconfig属性の値ではない。
embeddingは公式wrapperのmean pooling（promptを含む）と明示L2正規化、rerankerは公式headとsigmoidを使う。
rerankerの公開wrapperは入力ごとに`predict`を1回呼び、activation callableで生logitを保存して
Torch sigmoidを返す。同じforwardのlogitとscoreを記録し、別shapeのmodel比較は別に実行する。
公式wrapper自身のtoken列・出力と、ruricoのtoken列をそのまま与えた参照model出力を分ける。
後者のchunkはruricoのplannerが生成した単位であり、公式wrapperがchunkingを行ったとは扱わない。

| ファイル | 読み方 |
| --- | --- |
| `rurico-{embedding,reranker}.json` | raw token ID/mask、truncate後、文書chunk、batch内のID/順序/padding、rurico出力、公開API出力 |
| `reference-{embedding,reranker}.json` | 独立tokenizationと公式wrapper出力、同一token/shapeを再生した公式model出力、参照条件 |
| `comparison-*.json` | token差、各vector/logit/score、padding差、wrapper差、chunk比較、候補対の順位・近接候補 |
| `models.json` / `environment.json` / `build.json` | モデル内容、source/環境、build条件とbinaryのhash |
| `complete.json` | 実行完了と生成物hash。数値一致を意味しない |

embeddingのcosineと最大絶対差、rerankerのlogitとscoreの双方を評価する。
閾値はIssueの測定前診断値のまま。`same_token_numerics_within_criteria`は同一token比較だけを示し、
wrapperや検索品質の総合合格ではない。`ranking`では2e-4以下の公式logit差を近接候補として残し、
それより離れた候補対の逆転を不一致とする。公開`rerank()`は実返却順位を評価する。
形状・件数・順序・非有限出力が不正なら比較を中止し、数値の閾値超過なら生値と診断結果を保存する。

比較器だけを再実行する場合（新しいoutput名を指定）:

```sh
python3 docs/research/issue-307/compare.py /tmp/rurico-307-numerical/rurico-embedding.json /tmp/rurico-307-numerical/reference-embedding.json /tmp/embedding-recomparison.json
```

原因を読む順序は、raw token差、wrapperのstrip/truncation、ruricoのchunk列、
同一tokenの実長/bucket差、公開API対直接演算、hidden probe対pool後の差とする。
embeddingのhidden probeは有効tokenの先頭・中央・末尾の3位置だけを保存する。
ここが一致してpool後だけ違っても、全hidden tensorやpoolingだけに原因を断定できない。
FP32同士で差が出た場合も、演算順序やkernel実装は仮説であり、BF16/量子化・全layerの検証済みとはしない。

## amiciの検索品質

`run.py amici` は公開commit `547f9ee2ed734a2eab316fdbd62f194849875ee4` のarchiveを新規取得する。
既存のamici checkoutを参照・変更しない。依存の通常用/dev用2箇所のrurico revisionだけを開始版
`d0639cc815d03f44dc7c80abebe40e9157deaca1`へ置換し、`cargo update -p rurico --precise <開始版>`で解決する。
変更前後のmanifest差分・lockfile・解決済みpackage一覧を保存する。一括依存更新はしない。

既存`eval_harness`をdev buildし、次の既存modeを使う。
`capture-baseline`の保存先は**新しい観測ファイル**で、fixtureのbaselineを置き換えない。
aggregation、RRF/source weights、normalizationは既存baselineの値を明示して揃える。

```text
cargo build --locked --features eval-harness --bin eval_harness
eval_harness capture-baseline output=<新しいamici-measured.json> <baselineと同じ設定>
eval_harness verify-baseline baseline=tests/fixtures/eval/baseline.json <同じ設定>
```

`amici-measured.json`が既存metrics/95% bootstrap CIとcategory別結果の正本。
`search-comparison.json`は件数・query分類、設定、既存baselineとの差とverify終了値をまとめる。
CIは既存の1000 resamples/seed 42を再利用し、新しいCI計算器を作らない。
verifyの終了1はbaselineとの不一致として保持する。その他の失敗は未完了であり、既存baselineの再生成や
amiciコード修正で通過させない。再開には失敗した工程の依存・ツールチェーン・runtime条件を解消する。
製品修正が必要なら別repoの合意範囲確認へ戻る。

amiciのraw snapshotには`mlx_rs_version=0.25`というコード内定数が残る。
値を修正せず、実際に使った版は`resolved-packages.json`と`amici-resolved.lock`から読む。
この結果はamiciのreference compositionに限る。recall等の全アプリ品質や速度改善へ外挿しない。

## 結果の引き継ぎ

実測後は両比較の件数・範囲・超過条件・近接順位、検索metrics/CI/baseline差、確認できた原因、
未確認箇所と後続候補を`report.md`へ追記する。製品修正や基準変更は別途扱う。
公開する生出力、manifest、差分、lockfileを確認してこのディレクトリ内の実行別保存先へ選択して置く。
`*.private.log`、実行用checkout、venv、archive、巨大な中間tensorを一括追加しない。
モデルカードの固定版・wrapper設定と内容の意味も確認する。hash一致だけで指示や意味の一致を証明しない。
同じheadの標準check・独立評価・CIと、追加調査の実測完了を区別してPRへ引き継ぐ。

## 保存済みの生出力を再集計する

大きな生JSONはgzipで保存した。モデルを取得せずに次の操作で再集計できる。
`complete.json`のhashは展開後のJSONを指し、`results/packaging.json`は圧縮前後を対応付ける。

```sh
mkdir /tmp/rurico-307-offline
gzip -dc docs/research/issue-307/results/numerical-rechecked/rurico-embedding.json.gz > /tmp/rurico-307-offline/rurico-embedding.json
gzip -dc docs/research/issue-307/results/numerical-rechecked/reference-embedding.json.gz > /tmp/rurico-307-offline/reference-embedding.json
python3 docs/research/issue-307/compare.py /tmp/rurico-307-offline/rurico-embedding.json /tmp/rurico-307-offline/reference-embedding.json /tmp/rurico-307-offline/comparison-embedding.json
```

rerankerも同様にファイル名の`embedding`を`reranker`へ変える。再集計したJSONは
`results/numerical-rechecked/comparison-*.json`と同じ内容になる。

公式reranker参照を1回のpredictへ統合した後の追加実測は
`results/reranker-reference-refresh`に保存した。これは同じRust sourceで取得した
`results/numerical-rechecked/rurico-reranker.json.gz`を再生した公式参照の再実行である。
新しい`reference-reranker.json.gz`とそのRust出力を展開して比較でき、
全体の再実行には上記の`run.py numerical`を使う。結果と失敗時の経緯は[report.md](report.md)に記載した。
