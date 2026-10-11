# Issue #307 の実行手順

[Issue #307](https://github.com/thkt/rurico/issues/307) の公式数値比較とamici検索評価を、別の結果として記録する。
このディレクトリのコードは調査用であり、製品の許容差や既存fixtureを書き換えない。
実測結果・確認した相違・限界は [report.md](report.md) に置く。

## 候補commitの継続検証（Issue #366）

[Issue #366](https://github.com/thkt/rurico/issues/366)の候補検証は
[`candidate.py`](candidate.py)から明示的に起動する。常時モデルを取得するworkflowや
self-hosted runnerへのPRコード自動投入は追加せず、対応ホストで必要な候補を指定する。
通常CI・標準checkは軽量laneを維持する。実モデルlaneはそれらの成功とは別の受入証拠である。
ここでいうモデル出力のcaptureは数値の記録であり、UI媒体や画面撮影は不要。

embedding/rerankerの演算、tokenizer・prefix・truncation・chunk planner、bucket/padding/mask、
model loader、MLXやCargo依存、公式参照環境、FTS/vector/RRFの意味に影響する変更では、
採用判断の前に同じ候補の実モデルlane全体を実行する。文書だけの変更やモデル不要の入口修正では、
軽量laneが守る条件と残る実測範囲を説明する。#366の入口自体は少なくとも一度、数値と検索を
通して対応ホストで確認する。skip、モデル・Metal不在、中断、0件、旧記録の再集計はその代替にならない。

### 起動と候補の固定

下記のPython環境準備とホスト条件を満たし、リポジトリrootから実行する。
候補にはローカルに存在し、amiciが公開originから取得できる**完全なcommit SHA**を指定する。
branch名・短縮SHAは拒否する。入口は候補のcommitted sourceだけを新規隔離コピーへ取り出す。
元checkoutの未commit変更はモデル推論へ混ぜず、検証入口側のdirty状態・source hashとして別に記録する。
未commitの製品変更を検証したことにはならない。結果に対応するSHAをPRのheadと照合する。
検証入口だけを未commitで変更した場合は、そのsource hashも結果とPR説明へ引き継ぐ。

```sh
reference_env=/tmp/rurico-307-reference-env.XXXXXX # 下記で準備した実際のpath
candidate_sha=266fc41ff6bdf7ca5110e55a7315f8cbb5022cc6 # 例。評価対象の完全SHAへ置換
"$reference_env/bin/python" docs/research/issue-307/candidate.py \
  --candidate "$candidate_sha" --lane light --output /tmp/rurico-366-light-001
"$reference_env/bin/python" docs/research/issue-307/candidate.py \
  --candidate "$candidate_sha" --lane model --output /tmp/rurico-366-model-001
```

`light`は既存のPython環境・比較器・wrapperテストと候補照合のモデル不要テストを実行する。
新しい制御テストは標準checkの`scripts/test_candidate_quality.py`にも含まれる。
参照テストにはTorch等の導入が必要。固定310mモデルをdownload・推論せず、
小さなローカルcheckpointによるCPU検証を含む。
`model`は既存`run.py`の数値測定と`search_quality.py`のamici評価を順に呼ぶ。
数値参照はnative入力とbatchごとに新しい子processで実行し、終了してから次へ進む。
混在batchを含む元のshape・FP32・eager・4 threadsを維持する。各processはモデルを読み直すため、
測定時間の改善は目的にしない。`reference-events.jsonl`に入力ID・推論shape・開始／完了を残し、
全数値参照が完了してからamiciのビルド・評価へ進む。
数値・検索ハーネスは複製しない。固定入力hashが候補と異なる場合は不一致を保持して調査する。

HF/GitHub/Cargoへの接続と、固定した両310mモデルのcache、対応Xcode/Metalが必要。
モデルcacheは既存のHF snapshotを再利用し、revision/ETag・重み/config/tokenizerの内容hashを
開始時と終了時に照合する。amiciの通常/dev依存2箇所のみ候補SHAへ変更し、lockを局所解決する。
元consumer checkout・fixture・評価コード・baselineは変更しない。
出力先ごとに空の`amici/build-target`を作り、Cargoの依存解決・metadata・ビルドへ同じ
`CARGO_TARGET_DIR`を渡す。既存targetは再利用せず、metadataのtargetとの一致も検査する。
固定amiciのcapture/verifyモードは`fixture_dir`引数を読まないため、その引数では参照先を変更できない。
新規ビルドで今回の隔離コピーの`CARGO_MANIFEST_DIR`を埋め込み、両モードのprivate logにある
実際のfixture pathが今回のコピーと一致しなければ未完了として停止する。
HFとCargoの取得cacheは再利用できるが、amiciのbuild targetは実行ごとに増える。
成果物の確認・保管後、不要な出力先のbuild targetは利用者が削除できる。
Cargo metadataとlockの双方が唯一のrurico packageを同じ候補へ解決し、
取得済みrurico sourceが隔離候補と一致することを推論前後に検査する。
旧rev、別source、未取得候補では失敗し、baseline再生成やconsumer修正で通過させない。

### 結果の読み方と再実行

| 成果物 | 確認する内容 |
| --- | --- |
| `started.json` / `candidate.json` | 候補SHA、入口と候補のhead/dirty/source hash、候補Cargo.lock、入力・requirements hash、lane |
| `numerical/` | 既存の両モデル出力と独立した公式参照、全件数・順序・有限性・固定診断値、Python/model/tokenizer/環境・binary hash |
| `amici/`の直下の成果物 | 固定consumer commit、manifest/lock差分、解決済みpackage/source、既存metrics・分類別CI・baseline verify終了値、モデル・環境 |
| `complete.json` | 全工程の完了、候補SHAとlane、成果物hash、実測有無、診断・既存baselineの一致を別に記録 |
| `incomplete.json` / `failure.private.log` | 失敗・中断工程と原因。新規出力先で再実行するための手掛かり |

`complete`は実行完了であり、製品採用の承認ではない。model laneはembedding 23 batch/27行、
reranker 12 batch/16行、amici固定fixture 60文書/168クエリを確認する。
数値の診断は#307の基準をそのまま使い、hidden probeには正規化embeddingの基準を適用しない。
token列・padding・公開wrapper/API・順位を含む既存診断と、amici自身のbaseline許容差を別に記録する。
実行が完了して診断またはbaselineに不一致があれば、成果物を保持して終了1となる。
構造不正・工程失敗では`incomplete`となる。強制終了で`incomplete.json`さえ保存できなくても、
全体の`complete.json`がなければ未完了である。lightの完了は`model_measurement: false`で、実測成功ではない。
数値工程と検索の各外部commandは専用process groupで起動し、SIGTERM/SIGINTや工程例外時には同じgroupの子孫を停止し、
直接childの回収とgroup不在を確認してから`incomplete`を保存する。
2秒の回収猶予でgroup不在を確認できなければ回収失敗を記録し、成功とは扱わない。
lightのunittest runnerにもSIGTERMを例外として伝える。外側はまずrunnerへ停止を通知し、
最大10秒、テストの後片付けを待ってからrunnerのgroupを回収する。
制御テストの後片付けは検証用入口へSIGTERMを送り、その入口が所有する別groupの回収を待つ。
後片付けが応答しない場合はrunnerのgroupを強制停止するが、任意の別sessionの回収は保証しない。
SIGKILL等で入口自体が終了した場合の回収は保証しない。

再実行は同じ候補SHA・固定入力・参照環境で、未使用の出力ディレクトリを指定する。
停止したrunの再開・出力上書き・結果後の閾値緩和・baseline上書きは行わない。
保存済みデータの比較器再実行は下記の手順でできるが、新候補の実測と区別する。
PR/採用判断には候補SHA、入口source hash、lane、完了状態、診断超過、amici verify結果、
実行日時・環境と選択した小さな成果物への参照を残す。同じ候補の結果へ辿れない旧実測は流用しない。
公開前に成果物を確認し、private log・隔離checkout・archive・モデル・認証情報・巨大な生JSONは公開しない。

数値比較は固定入力の独立したモデル診断、検索は
[accepted ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)に従うamiciのreference compositionである。
検索metrics/許容差の所有者はamiciのままで、recall等を含む実アプリ全体の品質は未確認。
新しい製品品質閾値、性能/SLA、consumer migration、別backendの根拠にしない。

### 引き継ぎ根拠と検証の限界

開始版は`266fc41ff6bdf7ca5110e55a7315f8cbb5022cc6`（照合したlocal `origin/main`と一致）。
引き継ぎ[report.md](report.md)のblob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`は現物と一致した。
Issueの参照main `c8f250d60a5afb9944b9008d22ad6f4dda2d7103`以後、この研究ディレクトリでは
requirementsの6 pinが更新されている。#307の旧実測、#353の[更新記録](validation-353.md)は
その版と条件の証拠として保ち、現在の候補・現行requirementsの実測成功へ読み替えない。
遠隔mainの最新性と新しい候補の実測は、このローカル照合だけでは確認していない。
公開されたIssueの要求が正本であり、報告内の未採用仮説は今回の追加要求ではない。

2026-10-11の[候補検証記録](validation-366.md)に、候補入口の通し測定の結果、
memory pressureで中断した通し測定を区別して記載した。

標準checkは新しいモデル不要の候補照合・未完了制御テストを実行するが、実モデルlaneは起動しない。
CIのPython環境導入と既存テストも実モデルlaneの代替にならない。
対応ホストの担当AIが上記のmodel入口全体を実行し、同じsource hashの結果を確認してから
変更文書を含めた既存の独立評価へ戻す。撮影は不要。通常check・同じheadのCIは別に確認する。

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
