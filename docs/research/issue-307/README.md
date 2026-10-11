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

## 4 embeddingモデルの実出力検証（Issue #367）

[#367](https://github.com/thkt/rurico/issues/367)の入口は `run.py variants`。
既存の `numerical` と同じRust観測・公式参照・比較器を共用し、310mの既存入力を
このlane内で一度だけ観測する。#367のために `numerical`、reranker、amiciを追加実行しない。
`candidate.py` の#366入口はcommit済み候補の数値・検索品質用であり、今回の未commit差分の
4モデル実出力検証には使わない。過去の310m実測は[報告](report.md)の対象版の証拠であり、
今回の版や他3モデルの成功へ投影しない。

要求の照合版 `c8f250d60a5afb9944b9008d22ad6f4dda2d7103` から開始版
`fa31e0256b6d437d2fb1009fdb2b1eea0f97da45` への関連差分を確認した。
`src/embed.rs` はprefix encodeの共通化、`reference.py` は観測eventの追加があり、
ModelIdのrevision・次元の説明、config fixture、weights検証、報告本文は変わっていない。
引き継ぎREADME blob `e82e8f3ecc7cc18dbf20bbe911b566cab37c1603` と
報告blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` は開始版と一致した。
開始時のローカル `origin/main` もこの開始版だが、sandboxのDNS制限でremoteの最新mainは
再照合できていない。ホストでremoteと対象差分を照合する。
Cargo.lockの開始時SHA-256は
`96418da46475476d6f3c09f26d12160ceeed8c47da016250cdd344616a2d422c`。
過去の報告のPython版と現行requirementsの版は区別し、再実行は現行の厳密な環境検査を使う。

### 事前条件と判定

固定revisionは製品の[ModelId](../../../src/embed.rs)と[variants.py](variants.py)で照合する。

| model | revision | 次元 | 入力数 |
| --- | --- | ---: | ---: |
| 30m | `24899e5de370b56d179604a007c0d727bf144504` | 256 | 11 |
| 70m | `07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc` | 384 | 11 |
| 130m | `e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6` | 512 | 11 |
| 310m | `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` | 768 | 13 |

入力の正本は既存[inputs.json](inputs.json)と `variants.cases`。
30m/70m/130mはshort、日本語、コード、local-after、limit-after、query/topic/空prefixの
8条件にdocument/query/空prefixの空文字3条件を加える。310mは既存10条件に同じ空文字3条件を加える。
空の文書batchも別に確認する。local-afterはpadding混在、limit-afterは複数chunkの順序を観測するために保持する。
各modelのdoc数・chunk数とtoken長は実tokenizationから検査し、異なるtokenizerへ310mの数値を仮定しない。
文書は入力のdocument prefix条件を元の順で抽出し、chunkはplanner順で返す。
各batchのキー・件数・token/mask・shapeを独立した数値比較の前に検査する。

公開文書の全入力は4モデル共通の既存 `EmbedOptions { token_budget: Some(8192), ..Default::default() }`
で観測する。長文の各forwardは1 chunkとし、短文・空文書の混在batchは
`&dyn Embed` の既定 `embed_documents_batch` でも別に確認する。
製品の既定token budgetは変更しない。既定設定で複数の長文を同時に処理する条件は、
310mでmemory pressure停止を再現したため、今回の成功範囲に含めない。
`plan.json`とRust生出力にoption・実shapeを残し、旧default条件の成功へ読み替えない。

事前の診断基準は既存#307のcosine ≥ 0.99999かつ最大絶対差 ≤ 1e-5。
同一tokenの正規化vector、padding差、公開入口と直接演算、batch/single、queryとquery prefix付きtextへ適用する。
vectorの次元・件数・入力/doc/chunk順と有限性は必須。
具象 `embedding_dims()` と実返却vector、`&dyn Embed` のquery/document入口を同じinstanceで確認する。
公式wrapperの独立tokenizationとwrapper出力は別の診断として残す。
末尾空白・空prefix本文などのwrapper前処理差を、製品の前処理変更や新しい許容差の根拠へ自動昇格させない。
hidden probeは3位置の最大絶対差だけを記録し、embeddingの基準をhidden stateへ適用しない。
基準を超えた場合は比較結果を保存して調査し、通過目的で緩めない。

HF metadataのrevision/ETagと内容hash、F32 header、config次元、Rust/Pythonが使う3ファイルの
実path、具象次元、記録されたrepo/revisionを照合する。終了時にsource・実行ファイル・モデル内容を再照合する。
model identityやdimの不一致は拒否し、cache/headerの確認だけでは実推論成功にならない。
このlaneは索引へvectorを保存せず、モデルごとにprocess・結果directoryを分ける。
consumerで索引を用意する場合はrepo/revision/dimを索引の識別条件として保ち、異なるモデルの結果を同じ索引へ追加しない。
現行sqlite-vecの固定次元・不正BLOB拒否は[既存SQL検証](../../../CONTRIBUTING.md#vector-byte-bindの実sql検証issue-368)を再利用する。
同じ次元の構成互換性を判定する公開APIやconsumer migrationは導入しない（#315の別判断）。

### 実行と証拠保存

通常checkはモデル不要で、上のignored観測を実行しない。Apple Silicon、対応Xcode/Metal、
Rust 1.96以上、cargo-nextest、uvとネットワークを用意し、前節のPython 3.12.13新規venv・
requirements通常install・pip check・`reference.environment()`を行う。
`cargo fetch --locked` と設定済み `bash scripts/check.sh` はそのまま維持する。
Python依存は独立したCPU参照用で、製品は引き続きMLX-only。

以下は前節で作成した `reference_env` を使い、sandbox外・repo rootから実行する。
新規outputはcheckout外へ置き、既存の結果を上書きしない。

```sh
"$reference_env/bin/python" -m unittest discover -s docs/research/issue-307 -p 'test_*.py'
python3 -m unittest discover -b -s scripts -p 'test_*.py'
"$reference_env/bin/python" docs/research/issue-307/run.py variants /tmp/rurico-367-output-new
```

cacheはHF標準の `HF_HUB_CACHE`（未指定時 `HF_HOME/hub`、HF_HOMEも未指定時
`~/.cache/huggingface/hub`）をRust/Pythonで共用する。各モデルの
`models--cl-nagoya--ruri-v3-{model}/snapshots/{revision}` に固定weights/config/tokenizerと
公式wrapper用JSON等を取得する。4モデルは順にloadし、同時に保持しない。
必要容量・実行時間はこの変更では未測定であり、SLA・資源上限を新設しない。

`plan.json` は実測前に入力・全4モデル・基準・開始版・HEAD・未commit差分を含むsource hashと
Cargo.lock hashを保存する。モデルごとのdirectoryにenvironment、model manifest、build、
Rust生出力、公式参照生出力、comparison、completeを残す。private logは公開資料へ自動転記しない。
[既存#366の中断所見](validation-366.md#過去の指摘と証拠の適用性)を踏まえ、同じprocess group所有runnerを再利用する。
SIGTERM/SIGINTでは現モデルのRust・参照workerを回収してからincompleteを記録し、後続モデルを開始しない。
既存の実process中断テストへvariants入口の条件を統合した。SIGKILLで入口自体を終了した場合の回収は未確認。
`variants.json` は4モデルそれぞれの判定をまとめ、終了0には全モデルの事前基準内を要求する。
`complete.json` は実行完了であって基準内の保証ではない。失敗時の `incomplete.json` と
完了していないモデルdirectoryは未確認として保持する。

準備時にはモデル不要の比較器・manifest・新しい誤記録の反例を確認した。
2026-10-11〜12の[ホスト検証記録](validation-367.md)に、4モデルの現版実出力と公式数値比較を記録した。
310mは保存できたRust出力を照合し、メモリ確保後に公式CPU参照だけを分離実行して完了した。
[結果JSON](results/model-variants-20261012.json)から条件・件数・sourceと結果hashを辿れる。
単一runの全体入口の終了0は確認していない。既定設定の大きな長文batchも今回の成功範囲に含めない。
標準check・固定Python参照17件の成功と、変更文書を含む独立評価・同じheadのCIは区別する。
過去の結果を再生成して不一致を消したり、他モデルの成功で未実行モデルを埋めたりしない。
