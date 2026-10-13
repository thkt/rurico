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
prepareの内容照合と準備完了hashも、一度取得したplan bytesへ対応させる。
prepared記録の書込み後にはplanを新しく取得して再照合し、検査後の差替えを拒否する。
既存prepareテストは実ファイルとhashを使い、この差替えを確認する。git・ビルド・候補patchは模擬する。
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
推論sourceは変更せず、保存済み実測と現在runnerへの適用範囲は[測定版と現在版の適用範囲](report.md#coverage接続修正の確認2026-10-13)へ記す。

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
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -b -s scripts -p 'test_bucket_tables.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -b -s scripts -p 'test_mask_cache_spike.py'
```

入口検証ではclockとguardをmockにし、固定した開始値と30分後の期限が全検証へ渡ることを確認する。
実時刻の浮動小数点差分の厳密比較は使わない。資源停止・部分ログ保持・再割込み時の回収は、
既存のprocess検証で実child groupと模擬sensorを使って確認する。固定clockは実時間の停止精度を保証しない。

## coverage接続修正をホストで確認する

全体採用の見送り、512境界のMetal peak +320 MiB、mixed rerankerの利益不足と
長文共有の限定候補は[報告](report.md)のままである。保存実測は固定310m/F32と有限系列に限られる。
今回の変更は検証の接続であり、旧測定を新sourceの実行結果として扱わない。

通常CI・checkの`test-support,test-mlx,smoke`で`small_mask_elements`を実行する。
長さ1/3/5/8/9/17、half window 0/1/3/64のCPU/GPUについて、local全要素と
2行のpadding合成を独立した距離・key maskの期待値へ照合する。
window内のmasked keyは[ADR-0010](../../decisions/0010-adopt-maskfill-constant-to-avoid-metal-kernel-nan.md)の
固定値`-1e9_f32`を期待し、製品の`MASK_FILL`から期待値を取得しない。
既存の長さ5/window 1と長さ3/window 0のignoredテストをこの入口へ統合した。
単独で`test-mlx`だけを指定した従来の実行では統合先を選択できないため、`smoke`も有効にする。
negative引数はGPU生成の拒否を通常CIで確認し、既存CPUの拒否テストは維持する。
モデルは不要だが、MLXのCPU生成案もMetalを使う。CPU backendへの切替ではない。

担当AIはGPUを利用できるsandbox外のホストで、checkout・旧runの外の新しい保存先を使う。
公開head `6a69fee9bafea825ba341fe8f4a3d80694f1f816`＋今回の未commit差分、
開始commit `4cd349961e5d656dac0e272ec8d9b6573a854da3`からのPR全差分、Cargo.lockのhash、
Rust/Cargo/Xcode/Metalの版、生成sourceのhash、実行時間・終了コード・stdout/stderrを保存する。
CIと同じfeature・除外regexで次を実行する。実行対象だけを今回のoracleへ絞る。
通常の全テストは標準checkと公開CIで別に確認する。95%閾値・ignore・期限は変更しない。

```sh
cargo llvm-cov --workspace --features test-support,test-mlx,smoke \
  --ignore-filename-regex '(test_support\.rs|/bin/|embed/(mlx|embedder)\.rs|reranker/mlx\.rs|model_io/hf_backend\.rs|/mlx_cache/testing/runtime\.rs$)' \
  --lcov --output-path "$MASK310_EVIDENCE/lcov.info" \
  -- modernbert::model::spike310 --test-threads=1
diff-cover "$MASK310_EVIDENCE/lcov.info" \
  --compare-branch=4cd349961e5d656dac0e272ec8d9b6573a854da3 --fail-under=95
```

`MASK310_EVIDENCE`は新規外部保存先である。環境の既存diff-cover（必要ならCIと同じvenv）を使う。
未測定行とそこで守る失敗条件を記録し、95%以上の実測を確認するまで成功としない。
設定済みcheckはlcov/diff-coverを実行しない。公開CIは同じheadをorigin/mainと比較するため、
上の開始commitからの局所計測も別に必要である。

padding期待値の修正後は、ホストの一時コピーで製品側の`MASK_FILL`だけを
`0.0`と`1.0`へそれぞれ変更し、同じfeatureの`small_mask_elements`を限定実行する。
各試行で正常sourceから復元し、両変異がpadding合成の数値assertionで失敗することを確認する。
ビルド失敗・GPU初期化失敗は検出の証拠にしない。変異を混ぜない正常版の成功も保存する。
変異用の一時コピー・target・coverage出力は通常検証から分離し、旧run・保存比較・checkoutを変更しない。
生ログ・終了コード・正常版と変異版のsource/hash・失敗したassertionを外部保存する。
標準checkは変異を適用しないため、その成功だけではこの検出力を確認できない。

共用oracleを変えたため、同じsourceの大容量入口もcurrentの1回だけ限定再検証する。
モデル再ダウンロード・255processの性能再測定・全5候補の実モデル反復は追加しない。
`cargo test --locked --lib --features test-support,test-mlx,smoke --no-run`で新しいlibtestをビルドし、
表示されたbinaryを`MASK310_BINARY`へ指定する。次は既存の監視・回収関数を使い、
30分・RSS 8 GiB・pressure=1・swap増分256 MiBを維持して実行する。
ビルド・coverage・他のGPU処理とは重ねず、上位ホストの実行上限も維持する。

```sh
python3 - "$MASK310_BINARY" "$MASK310_EVIDENCE/full-mask" <<'PY'
import importlib.util
from pathlib import Path
import signal
import sys
import time
def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}; retain partial evidence")
signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)
spec = importlib.util.spec_from_file_location("mask310_host", "docs/research/issue-310/run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
output = Path(sys.argv[2]).resolve()
m.compare.oracle.require(not output.is_relative_to(m.ROOT), "use external evidence directory")
output.mkdir(parents=True, exist_ok=False)
baseline = m.runner.host.snapshot(-1)
m.runner.save(output / "baseline.json", baseline)
m.compare.oracle.require(baseline["pressure"] == 1, "pressure abnormal before verification")
started = time.monotonic()
m.guarded_verification(["/usr/bin/time", "-l", str(Path(sys.argv[1]).resolve()),
    "modernbert::model::spike310::all_mask_elements_and_costs", "--exact", "--ignored",
    "--nocapture", "--test-threads=1"], m.ROOT, output, "mask", baseline, started, started + 1800)
m.compare.oracle.require("1 passed; 0 failed; 0 ignored" in (output / "mask.stdout").read_text(),
    "missing selected test execution")
events = m.parse_observations((output / "mask.stderr").read_text())
expected = {(n, h, g) for n in [1, 8, 9, 17, 128, 512, 2048, 8192]
            for h in [0, 3, 64] for g in [False, True]}
m.compare.oracle.require(len(events) == 48 and
    {(e["seq_len"], e["half_window"], e["gpu"]) for e in events} == expected and
    all(e["event"] == "mask_elements" and e["elements"] == 2 * e["seq_len"]**2 for e in events),
    "incomplete full-mask verification")
m.runner.save(output / "mask.json", events)
PY
```

生ログ、48設定とイベント形式、Array解放後cleanupのsource照合を新しい検証要約へ残す。
失敗時は部分ログとstatus/monitorを保ち、旧runへ上書きしない。
必要な要約・文書更新後、同じcheckout・元の差分基準で通常入口のcheckと独立評価へ戻る。
この限定検証はモデル出力・失敗時cleanupの再測定ではなく、その既存検証は維持している。
