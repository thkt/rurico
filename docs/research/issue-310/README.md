# Issue #310 mask・cacheの独立比較

[Issue #310](https://github.com/thkt/rurico/issues/310)の調査用入口。
製品のbucket・precision・cleanup policyを維持し、隔離コピーで候補を一つずつ比較する。
API調査、実測結果と根拠の適用条件は[報告](report.md)を参照する。
UI媒体は不要で、`capture: null`と[標準check](../../../CONTRIBUTING.md#テスト)は変更しない。
数値記録はUI captureではない。

## ホストで準備・検証する

担当AIはsandbox外のApple Silicon、対応するXcode/Metal、Rust 1.96以上で実行する。
開始HEAD `4cd349961e5d656dac0e272ec8d9b6573a854da3`と未commit差分、Cargo.lockを固定する。
最新mainと[親Issue #296](https://github.com/thkt/rurico/issues/296)の検証・成果物条件を取得して照合する。
初回担当のsandboxでは取得できなかったが、担当AIはホストで開始版と最新mainの一致、親Issueの条件を確認した。
再実行時にmainとの差があれば関連箇所を再評価し、
古いplanへsourceを上書きしない。要求・権限・実行上限の変更が必要なら実行せず人へ戻す。

[既存#308 runnerのホスト準備条件](../issue-308/README.md#ホストで準備する)を使い、
固定310m embedding/rerankerのcache、環境context JSON、他のGPU処理との隔離を確認する。
model revisionと3ファイルのhashは[#307 manifest](../issue-307/results/numerical-rechecked/models.json)へ照合する。
モデルの取得が必要な場合も既存の登録済みdownload入口を使う。私的入力は取得しない。
setupは`cargo fetch --locked`、全体検証は`bash scripts/check.sh`のままである。

repo rootから、checkoutと旧runの外の新しい保存先を指定する。

```sh
python3 docs/research/issue-310/run.py prepare /tmp/rurico-310-prepared-new
python3 docs/research/issue-310/run.py verify /tmp/rurico-310-prepared-new /tmp/rurico-310-context.json /tmp/rurico-310-verify-new
python3 docs/research/issue-310/run.py measure /tmp/rurico-310-prepared-new /tmp/rurico-310-context.json \
  /tmp/rurico-310-evidence-new --kinds embedding --cases boundary128
```

`prepare`は未commitファイルも含む5つの独立コピーを保存し、同じlock・release・
`test-mlx,smoke`のlibtestをビルドする。`--cache-base OLD_PREPARED`は対応する5つのtargetを通常コピーし、
現在sourceを再コンパイルする。旧source・targetは変更しない。
元checkoutのファイル集合、HEAD、候補source・binaryのhashを準備完了時と実行前後に照合する。
各`verify_plan`境界でplanを新しく一度読み、そのbytesからhashとJSONをそれぞれ一度算出し、準備完了検査・内容照合・返却へ共用する。
呼出元が使うplanとcontextへ記録するhashも、この観測から返す。実行後は新しく再照合する。
#308のprepareを共用し、その既存CLIのbucket比較と既定動作を保つ。
measureとcollectは所有groupの回収とstderr閉鎖後、一度の読取り・行分割で数値記録とmaskイベントを抽出する。
両入口は解析結果を受け取り、raw文字列はファイルごとの抽出後に保持しない。保存完了時のhash検査は別に行う。

`verify`は現行コピーでCPU/GPU両maskの全要素を照合する。長さ1/8/9/17と4 bucket、
half window 0/3/64、対称性・window両端・右padding・内部のmasked keyを含む。
8192のreadbackは意図的に大きく、製品のreadback量や性能へ読み替えない。
全5コピーで既存`real_model_cleanup_memory`を実行する。各経路31試行、計93試行について
成功→Forward/Pool/Eval/Readback失敗からの復帰とcleanup一回、成功出力の有限性・初回からの差を
既存assertionで確認し、72回の注入失敗では両FFI clearを呼んだことも追加照合する。
`/usr/bin/time -l`、stdout/stderr、終了結果、JSONイベントを新規保存先へ残す。
各検証の回収とログ閉鎖後にstderrのmask310イベントを解析し、cleanup・モデルidentity・mask設定数を照合する。
全検証後にplanとモデルhashを再照合してから`complete.json`を保存する。解析や照合の失敗も`incomplete.json`へ残す。
verify全体で30分の期限を共有し、各検証の所有process groupのRSS、pressure、swapを
約0.1秒ごとに標本観測する。RSS 8 GiB超、pressure 1以外、開始時からのswap増分256 MiB超、
期限到達で停止する。開始時pressureが1以外なら検証を起動しない。
`baseline.json`、各検証の`*.monitor.json`と`*.status.json`に基準値、観測、終了値とgroup回収を残す。
停止時もstdout/stderrと部分観測を保ち、所有groupの回収後に`incomplete.json`を記録する。
この入口に監視を接続したため、外側の独自wrapperは不要である。
取得／clear／freeの失敗コードとArray寿命は標準checkの既存検証も維持する。
backend abort・panic・複数threadの競合・無期限リーク保証は対象外である。

変更版の既存fixture検証もホストで別に実行し、終了コードとログを外部保存する。

```sh
cargo run --locked --release --features smoke --bin mlx_smoke -- verify-fixture
```

## 同じ入力・shape・F32で比較する

| コピー | 単独の変更 | 維持する条件 |
| --- | --- | --- |
| current | 観測のみ | local層ごとに合成、CPU local mask生成、毎forward両cache clear |
| shared | forward内で合成maskを一度共有 | CPU local mask生成、毎forward両cache clear |
| gpu | `arange`・距離比較・`select`でlocal mask生成 | local層ごとの合成、毎forward両cache clear |
| buffer | allocator cache limit 64 MiB、成功時だけ保持 | CPU mask、層ごとの合成、毎forward compile clear、失敗時両clear |
| compile | 成功8 forwardごとにcompile clear | CPU mask、層ごとの合成、毎forward buffer clear、失敗時両clear |

buffer limitは専用process内で設定し、cleanup境界でcache観測値が64 MiBを超えた場合もclearする。
allocatorへの返却・非同期解放の瞬間を含む厳密なpeak上限ではない。
compile案は保持期間をforward数で制限する仮説で、entry数・bytesの上限APIは使っていない。
無制限保持を安全な改善として推薦しない。設定は製品hard limitではなく、consumerへ公開しない。

計測は#308の公開文書batch/query入力と#309のreranker reference観測器を共用する。
文書budget 8192・pause 1 ms、reranker budget 256000・全pair最大長方式を固定する。
#308の新bucket、#309の長さ別分配、#321のprecision変更を混ぜない。
各caseについて3組の独立processを使い、5コピーの開始順を0/2/4だけ循環させる。
これは全順列を均等に試す設計ではない。load後の初回をcold、残りをwarmとして分ける。
OS file cacheを消したcoldではない。
通常9回、mixed/multichunkは31回で、compile clearの8 forward周期と継続系列を観測する。
shape数はlocal mask cacheのkey数と実forward形状を別に記録し、compile entry数へ読み替えない。

`--kinds embedding`または`reranker`、`--cases CASE...`で分割して新規保存先へ実行する。
embeddingの8 case、rerankerの9 caseは[#308の入力一覧](../issue-308/README.md#同じ条件で観測する)と同じである。
全caseの比較は省略せず、部分成功を全体の成功として扱わない。
長いcaseは`--trials 1 --trial-start 0`、`1`、`2`を各々新規保存先で実行し、
3組のsummaryを合わせて判断する。単独の1組を3組のばらつきの代用にしない。
この分割でも同じprepared/context、全5コピー、9回以上の反復を維持し、実行上限を増やさない。

mixed/multichunkで5コピーを一度に測ると上限に達する場合は、候補ごとに31回を測る。
次の例をtrial 0/1/2と全5候補へ繰り返し、計15の異なる保存先を使う。

```sh
python3 docs/research/issue-310/partition.py --repo "$PWD" collect \
  /tmp/rurico-310-prepared-new /tmp/rurico-310-context.json \
  /tmp/rurico-310-phase-0-current --kind reranker --case mixed --variant current --trial 0 --repeats 31
python3 docs/research/issue-310/partition.py --repo "$PWD" combine \
  /tmp/rurico-310-combined-new /tmp/rurico-310-phase-*
```

`combine`には同じcaseの15保存先だけを渡す。source/model/collectorのhashと条件、
全候補・3組、終了コードとprocess回収を照合し、欠落・重複・保存hashとの不一致を拒否する。
必要な各ファイルを一度読み、そのbytesのhash確認後に同じ内容を解析・集計する。
stderrのイベント解析、`phase-complete.json`の解析と記録hashも、それぞれ同じ観測を使う。
生bytesはファイルごとに処理し、全phaseの全ファイルを常駐させない。
これは確認したbytesと集計内容の対応であり、保存先全体の原子的snapshotや改変者の認証ではない。
`phase-complete.json`は1候補の終了で、比較全体の終了ではない。停止した保存先は再利用しない。
初回追加時の入口は実測中の外部collectorと同じbytesだった。現在はplanの解析・hash共有、回収後stderrの共用、結合時の読取りを修正した。
推論sourceは変更せず、保存済み実測と現在runnerへの適用範囲は[最新の評価記録](report.md#assessments-r5指摘に対する修正2026-10-13)へ記す。

1回の分割でも既存上限に達する場合は部分ログを保ち、上限を無断で延長しない。
長時間推移が不足する場合は`--sustained-repeats N`（9以上）を増やして新規実行する。
回数を時間へ換算して推測しない。
#308の既存監視・停止条件（30分、RSS 8 GiB、pressure=1、swap増分256 MiB）と
process group回収を共用する。上位ホストの実行上限も維持し、超過は新しい条件を無断採用せず記録する。
SIGINT/SIGTERM・通常失敗では部分ログを残す。verifyとmeasure/collectは、所有groupの回収と
monitor/status保存中のSIGINT/SIGTERMを一時的に無視し、終了時に元のhandlerへ戻す。
回収失敗は伝播し、`complete=False`のまま保存する。`group_absent=True`は回収の成功後だけ記録する。
runner自体のSIGKILL後の回収は未確認である。

## 実測後に解釈する

`summary.json`はcold/warm・Metal peak・RSS/pressure/swapの時間系列、入力と出力の差、
mask graph構築・層ごとの合成構築、cache clear実呼出しとcleanup後active/cacheを持つ。
embeddingの既存metricsのforward/eval・CPU readback・cleanup・pauseと実readback要素数も利用する。
実出力は文書/chunk ID・768次元・有限性・全要素の差・queryを照合する。
同一token列・shape・options・model条件を変えた比較を拒否する。
rerankerはlogit/score実差、順位・完全同点順を診断として保存する。
#308の各2.5e-5という比較承認を#310の許容差に移さない。
既存観測器内の同一コピーの内部製品経路照合はその既存基準を維持する。

maskの単体graph/eval/readback時間は補助的な観測で、CPU→GPU転送だけの時間ではない。
CPU/GPUの単体oracleは同じprocess内でCPUから実行し、順序の影響も含む。
`local_readback_ns`は`as_slice`取得時間で、全要素走査・コピーの時間ではない。
モデル内graph時間はlazy構築であり、個々のMetal kernel時間やlazy重複除去の証明ではない。
全候補へ同じログ観測を入れるためwallには観測負担が含まれる。
RSSはprocess範囲の約0.1秒ごとの標本最大、Metalはcall範囲のallocator peakで、範囲が異なる。
GPU生成後もF32 dense maskは長さ²で、8192では計算上256 MiBである。中間Arrayとbatch合成も残る。

担当AIは原本の生値を保持し、同じ条件の時間・peak・shape・系列・数値を合わせて
[報告](report.md)へボトルネック、候補ごとの利益／退行、ばらつき、採用候補と保留理由を追記する。
`complete.json`は収集完了で、受入や採用ではない。
長文attentionがなお支配的である根拠が出た場合だけ既存backendの双方向windowを追加調査する。
causalへの置換や独自kernel開発、許容差変更は行わない。
文書を含む同じ差分を既存check・独立評価へ戻し、draft PRと同じheadのCIを確認する。製品policyの採用とマージはこの調査に含めない。

モデル不要検証は標準Python discoveryに含まれ、個別には次を使う。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -b -s scripts -p 'test_mask_cache_spike.py'
```
