# Issue #367のホスト検証記録

2026-10-11、開始commit `fa31e0256b6d437d2fb1009fdb2b1eea0f97da45` と今回の未commit差分を測定した。
開始時の遠隔mainも同じcommitで、Cargo.lockのSHA-256は
`96418da46475476d6f3c09f26d12160ceeed8c47da016250cdd344616a2d422c`。
各runのplan/environmentに測定対象の全ファイルhashを保存した。

## 現在の結果

4モデルそれぞれの出力契約と独立した公式数値比較が完了した。
[小さな結果記録](results/model-variants-20261012.json)に条件・件数・sourceと結果hashを保存した。
30m・70m・130mは、既存の文書batch optionに
`token_budget=8192`を指定した今回の条件で完了し、出力契約を満たした。
各11入力・25 batchの公式参照、公開APIの次元・件数・順序・有限性、空入力、chunk、
batch/single、query/text、短文の既定trait batchを確認した。
次表は同一tokenの正規化vectorと公式参照の比較である。

| モデル | 次元 | 公式数値・公開契約 | 最小cosine | 最大絶対差 |
| --- | --- | --- | --- | --- |
| 30m | 256 | 成功 | 0.999999999997 | 4.991889e-07 |
| 70m | 384 | 成功 | 0.999999999991 | 9.83476639e-07 |
| 130m | 512 | 成功 | 0.999999999981 | 1.13248825e-06 |
| 310m | 768 | 成功 | 0.999999999987 | 9.53674316e-07 |

310mのRust観測は13入力・29 batchを完了した。公開文書8件・短文の既定batch6件、
query/textとbatch/singleの自己整合は事前基準内だった。2026-10-12の公式CPU参照も全42単位が完了した。
今回の再測定で、同じRust出力と独立した参照の比較も事前基準内となった。
実件数とshapeは生出力を正本とする。

cosine下限0.99999・最大絶対差1e-5を実行前に固定した。hidden probeとwrapperのtoken差は
別の診断であり、製品の許容差や前処理を変更していない。
参照環境はPython 3.12.13、Torch 2.14.1、Transformers 5.18.0とSentence Transformers 6.1.0。
演算条件はFP32/eager/4 threadsで固定した。ホストはApple M3・24GiBである。

## 中断と検証方法の修正

2026-10-11の承認測定枠は19:54:11〜20:54:11 JST。
既定設定の最初のrunは20:04:01、再試行は20:15:50にmemory pressureが正常値1から警戒値2となり停止した。
両方とも310mのRust処理中だった。最初のログには入力別の進捗が含まれておらず、停止箇所の特定は次の診断で行った。
進捗ログを追加した310m単独の診断では、直接モデル比較と公開textは完了し、
公開文書batchの開始後に20:21:18に同じ停止を再現した。

研究laneだけで既存の `EmbedOptions::token_budget` を8192へ明示した。
長文の各forwardは1 chunkとし、短文・空文書の混在batchは既定 `&dyn Embed` 入口でも確認する。
入力、公式参照のshape・演算条件、数値基準は維持した。製品の既定値・資源hard limitは変更していない。
この条件で4モデルのRust観測を通過したが、310mの公式Python参照の8190-token native入力で
20:35:23に停止した。既定設定の大きな長文batchは今回の成功範囲に含めない。

GPU終了後、保存できた310m Rust出力のsource・binary・モデル・入力hashを照合し、
新しい出力先で公式CPU参照だけを実行した。8190-token native入力は完了したが、
次の8192-token入力で20:39:11にpressureが2となり停止した。
保存Rust出力の再利用と新しいCPU推論を区別し、全体入口の新しい完了には読み替えない。
当日の全体variants.jsonと310mの独立したcomparison/completeは未作成だった。
その中断runは未完了のまま保持した。

各中断の生ログと所有process groupの不在を保存し、停止後にpressureが1へ戻ることを確認した。
CPU分離実行の最大観測RSSは約7.58GiBだが、5秒間隔のRSSでありpeakや原因を証明しない。
比較入力・数値基準の削減、停止条件の緩和、旧310m結果の流用で未完了を埋めない。
長文の参照を実行する際は、別作業を停止し十分なメモリを確保したホスト状態が必要である。

## 2026-10-12の310m公式参照

ユーザーがメモリ確保と回答後30分の測定枠を承認した。01:23:30〜01:53:30 JSTの枠で
公式CPU参照だけを実行し、01:29:17に終了0で完了した。native13入力と29 batchの計42単位を、
各単位ごとに新しいprocessで実行した。全監視点のpressureは正常値1で、最大観測RSSは約9.63GiB。
終了後、所有groupの不在と完了artifact hashを確認した。

Rust出力は2026-10-11に完了した観測を再利用した。再測定前にbinary・モデル内容・入力のhashを照合し、
source差分が研究READMEとこの検証記録だけであることを確認した。推論・観測・比較コードと依存は同じである。
参照のmodel manifestは再取得したmetadataと実ファイル内容を照合し、旧Rust観測と一致した。
旧参照JSONを再集計した結果ではなく、新しい公式CPU推論との比較である。

4モデルの結果は同じ演算コードと明示したoptionに対するモデル別証拠として揃った。
単一runによる全体入口の終了0や新しいvariants.jsonは確認していない。
保存済みRust出力と新しいCPU参照の分離実行は、全体入口が中断なく完了した証拠とは区別する。
既定設定で複数の長文を同時処理する条件、SIGKILLによる入口終了、長時間安定性も未確認である。

## 通常検証と残作業

初回標準checkはClippyのimport違反1件を修正後に成功し、固定Python参照テスト17件も成功した。
測定後の追加修正は進捗ログ、研究用optionの明示、既定の短文trait batchの観測・検査である。
option取り違えと既定batchの欠損は既存の反例テストへ統合した。
修正後の標準checkは成功した。310mの公式比較は上記の再測定で完了した。変更文書を含む独立評価と同じheadのCIは別に確認する。
検索品質、モデル・量子化の製品採用、長時間安定性は今回の検証範囲に含めない。

旧成果物を保持し、生ログ・モデル・個人path・巨大な数値JSONは公開しない。

## assessments: ホスト復帰後のR1-1修正（2026-10-12）

評価対象 `a5027c72372119603c471a4ec9714d452d2d78bfc11a7ed0df74e82cdf3ef5d0` の
未解決必須指摘R1-1を再現して修正した。`run.py models` は40桁Git blob ETagの
検査で同じファイルをSHA-256用とGit blob SHA-1用に二度読んでいた。
現在は一度取得したbytesから両hashを計算する。LFSの逐次読取り、revision/ETag照合、
独立したheader/config検査、推論後の全モデル内容再検証は維持した。
評価対象からの変更はこの処理、既存manifestテストと本追記だけである。

既存 `test_capture.py` のfixtureへ、実モデルと同じ5ファイルのGit blob ETagと
LFS ETagを混在させた。hash関数の呼出し数だけを見る従来のassertionを、実際の
ファイルopenの観測へ置き換え、header/configの追加読取りは明示的に保持した。
同じfixtureで修正前は15回のopenとなり失敗、修正後は10回で成功した。
期待SHA-256・ETag・保存manifestを照合し、Git/LFS双方の不正ETagではmanifestを
保存せず拒否することも同じテストに統合した。既存のLFS・追加JSONの検出条件は失わない。
二重読取りの再発と内容照合を省く退行を防ぐ保証は、合成vectorの比較テストにはないため維持する。
ネットワーク・推論・巨大fixtureを追加せず、Python 3.15.0で今回のmanifestテスト1件は0.004秒で成功した。
実モデルのI/O時間・長期不安定さ・保守費用の改善は測定していない。
固定Python 3.12.13参照環境全体の再検証は今回行っておらず、過去の17件成功とは区別する。
同じ環境のモデル契約反例3件は0.055秒、既存比較器7件は0.011秒で成功した。
テストの削除や別ファイルへの移動、製品基準の緩和は行っていない。

元runの停止記録・生応答・成果物とホスト復帰の評価履歴を読み直した。
停止記録に列挙された12記録と追加ホスト証拠の39記録はSHA-256が一致した。
証拠のpassedは実行の申告として扱い、4モデルのRust/公式生出力を現行
`variants.evaluate`で再集計し、各保存comparisonとの完全一致と小結果JSONの
出力hashを確認した。各manifestの固定revisionと5個の40桁ETagも照合した。
測定時sourceとの相違は研究README、本記録、今回のmanifest処理とテストであり、
推論コード・入力・依存・比較基準は同じである。manifest生成の修正は推論や
数値条件を変えず、局所fixtureで内容照合を確認したため、保存数値は記載条件での
モデル別証拠として適用できる。今回の再集計は新たな推論成功の証拠ではない。

文書の根拠は [CONTRIBUTING](../../../CONTRIBUTING.md) の通常/実モデル検証の分担、
[現行手順](README.md)、[固定参照依存](requirements.txt)と上記生記録である。
開始commit `fa31e0256b6d437d2fb1009fdb2b1eea0f97da45` の
[旧報告](report.md) は指定blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` と一致し、
本文も保持した。READMEの開始blob `e82e8f3ecc7cc18dbf20bbe911b566cab37c1603` からの
変更は#367のモデル別入口・事前基準・cache・option・結果保存の案内である。
旧報告の旧依存による末尾空白差を今回の結果へ投影しない。
[ADR-0011](../../decisions/0011-adopt-uniform-vector-length-contract-for-the-embed-trait.md) と
[ADR-0014](../../decisions/0014-adopt-mlx-only-embedding-backend-and-retire-the-candle-fallback.md) の
acceptedな契約は維持し、[同次元の構成互換案](../issue-315/report.md) は未採用のままとする。

### handoff

R1-1は局所再現と修正後の内容照合で解消した。上記モデル別証拠は明示した
token_budget=8192と短文の既定trait batchに適用する。全体variants入口の連続完了、
既定budgetの複数長文同時処理、入口自体のSIGKILL回収、長時間安定性には適用しない。
これらの追加主張には新しい実測が必要であり、今回の局所修正の完了条件には加えない。
実モデルの再推論、通常Rust検証と同じheadのCIはこの修正では実行していない。
設定済み `bash scripts/check.sh` はホストが変更後に実行する。captureは未設定でUI媒体も不要。
manifestテストは既存の固定Python参照テスト入口の対象で、標準checkとは別の検証である。
変更文書を含む独立評価は今回の差分とこの証拠の適用性を確認する必要がある。
旧run・生結果・歴史的source hashを変更せず、commit・push・公開は行っていない。

## assessments: 初期planのR2-1修正（2026-10-12）

評価対象 `0dfca48fe993acf5b0b6a5390d72a73ed0c820c0e4216f30719fa75192d15012` の
必須指摘R2-1を修正した。初期planは `source()` でCargo.lockを読み終えた直後に、
同じ開始時点を記録するために再び全文を読んでいた。`cargo_lock_sha256` は取得済みの
`lane_source["Cargo.lock"]` を使用する。モデル開始前・モデル間・推論後のsource検査と
ビルド後のlock記録は維持した。工程をまたぐキャッシュや新しい分岐は追加していない。

Python 3.15.0の一時fixtureで変更前後の `main()` を同じlock・HEAD・入力条件で比較し、
初期planが完全一致し、plan保存までのlockのopenが2回から1回になることを確認した。
推論はstubで停止し、実モデル・ビルド・ネットワークは使っていない。
lock欠落時は変更前がFileNotFoundError、変更後がKeyErrorとなり、いずれもplanを保存せず
推論前に停止した。今回のコード差分はこの記録値1箇所だけで、後続検査は同一である。
局所比較の生記録はcheckout外の `/tmp/rurico-367-r2-local-check.log` に保存した。
実I/O時間の改善量は測定していない。

過去のR1-1も自動的に解消扱いせず、現行manifestテストを再実行した。
Git/LFS混在、open回数、保存hash、双方の不正ETag拒否の1件は0.005秒で成功した。
この既存検証は、同一内容の二重読取り再発と内容照合の省略を防ぎ、vector比較器とは
検出条件が異なるため維持する。R2-1は内部の記録値再利用で振る舞いを追加しないため、
局所比較で確認し、恒久テストは追加しなかった。既存テストの削除・統合・移動はなく、
失う検出条件もない。実モデル、process取消し、固定Python参照環境全体は再実行していない。
長期不安定さや保守費用の改善も未測定である。

停止記録の12記録と追加ホスト証拠の39記録、停止記録・result・追加証拠自体のhashを
再照合した。review-1のR1-1とreview-2のR1-1/R2-1を現在のコードへ適用して確認した。
証拠のpassedを受入として引き継がず、4モデルの生出力を現行比較器で再集計し、
保存comparisonとの完全一致と、小結果JSONの件数・モデル識別・出力hashを確認した。
事前planの数値基準も小結果JSONと一致する。測定時sourceからの相違は研究README、
本記録、manifest/初期planの処理とmanifestテストであり、推論・入力・依存・比較基準は同じである。
したがって上記条件でのモデル別数値証拠は適用できるが、新しい推論成功とは扱わない。

指定された開始commitとreport/READMEのblobを再確認した。旧reportは同一bytesであり、
READMEの差分はモデル別の現行手順である。根拠と合意状態は前節の参照を維持し、
旧310m観測の他variantへの投影や、未採用の構成互換案の採用は行わない。
追記は固定版natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151` の
手動観点で通読した。lintはuv cacheへの書込み制限があり、一時cacheへ切り替えても
offlineでSudachiPyを取得できず未完了だった。意味・条件・数値の照合は上記のとおり行った。

### handoff

R1-1は現行fixtureで、R2-1は同条件の局所比較で解消を確認した。変更後の標準checkは
ホストが設定済み `bash scripts/check.sh` で実行する。captureはnullで、UI媒体は不要である。
変更文書を含む独立評価ではこの追記と現行差分を確認する必要がある。
単一のvariants入口の連続完了、既定budgetの複数長文同時処理、入口のSIGKILL回収、
長時間安定性と同じheadのCIは未確認のまま残す。分離した310m参照の終了0を、
中断した全体runの成功に読み替えない。これらの追加保証を今回の完了条件には加えない。
旧run・生結果・歴史的source hashは保持し、commit・push・公開は行っていない。


## CIの差分coverage失敗への修正（2026-10-12）

公開head `b22c575efbe04d6ad3b7188f06707f5091966f96` から修正した。
[CI run 38158845188](https://github.com/thkt/rurico/actions/runs/38158845188) の
coverage job 114527335722は、集計後の変更行95% gateで77行すべてが未実行となり失敗した。
test・security・zizmorの成功と、同じheadの全CI成功は区別する。
原因は、モデル選択と公開APIの観測記録がignoredの実モデル入口だけから呼ばれていたことである。
過去の標準check成功や独立評価を、今回のCI失敗の解消へ読み替えない。

公開入口の観測を既存 `Embed` 境界の記録関数へ分け、モデル不要で実行する。
新しいbackendや製品APIは追加しない。cache選択、configの期待次元検査、
具象 `embedding_dims()` の照合は実モデル入口で行う。
次元照合は観測した具象値と期待値を返し、JSONには両方を記録する。
具象値の取得と照合は公開text/documentの実行前になったが、値を返すだけのaccessorであり、
公開APIの入力・prefix・順序・option・呼出回数は変わらない。
旧numerical入口の310m/default条件と、variantsのtoken_budget=8192・短文の既定trait batchを保つ。

shapeの準備と開始ログはembedding/reranker共通の既存 `shape()` が担当する。
embeddingではモデルload後、forward直前に開始を記録する。
各batchの完了ログは次のbatch開始または公開text開始でも進行を確認できるため整理した。
公開text、文書batch、既定短文batch、singleton、queryの進捗ログは保つ。
このログ整理による速度改善は測定していない。
演算に渡すtoken/mask/shape、モデルload・forward・pool・eval・readback・解放の順序は同じである。

### 通常テストの保証と費用

既存の件数拒否テストは、zipで不足・過剰出力が消える不具合を防ぐため維持した。
公開記録の新しいテストは、入力のrepeat/prefixを失う、doc/queryを誤選択する、
文書や複数chunkを並べ替える、8192 optionを落とす、既定短文batchや空batchの実呼出しを
省く、観測shapeやモデル識別を誤記する退行を検出する。
512-token chunkを含め、後続に513-token chunkがある文書を除外する対照を使う。
入力とは異なる順のtoken記録を照合し、文書の選別と元の順序を別々に確認する。
合成vectorは一要素の識別値であり、公式一致・実モデルの次元/有限性の証拠にはしない。
モデル選択と次元照合のテストは、未知指定の既定modelへのfallbackと、
768次元を256次元として記録する構成を拒否する。

既存batchテストには、flattenしたID/maskとshapeの対照を統合した。
7件の短文から先頭5行だけを選ぶ条件も確認し、混在診断の行数増加・選択順の変更を防ぐ。
既存の2短文・長文の対照だけでは、この先頭5行の上限が失われても検出できなかった。
件数・次元・有限性・数値逸脱を検出する既存Python比較器と反例は重ねて作らず維持する。
一時sourceコピーで短文選別を「先頭chunkだけ」に改変すると、新しい公開記録テストは
短文結果にlong文書が混ざったassertionで失敗した（終了101）。
同じ本体sourceの通常実行は成功しており、ビルド失敗を検出成功とは扱っていない。
この反例のログはcheckout外の `/tmp/rurico-367-recorder-counterexample/test.log` に保存した。
恒久テストの削除や除外pathへの移動はなく、失う検出条件もない。
追加したspyの保守は公開記録の契約に限定する。モデル取得・巨大fixture・network待機は追加しない。

Python 3.15.0で既存比較器7件は0.011秒、モデル契約反例3件は0.060秒、
manifest検証1件は0.006秒で成功した。
MLX-onlyの既存build資産を一時領域へ複製し、固定lock・既存featuresでRustの対象5件が成功した。
追加3件と既存2件を含み、実行部分は0.01秒、実モデル3入口はignoredのままである。
標準check全体、固定Python 3.12.13参照環境全体、新しい実モデル推論は実行していない。
実行時間・不安定さ・保守費用の改善量は未測定であり、テスト数や行数を改善の指標にはしない。

Rust 1.99.0の同じlib testを追加でinstrument-coverage付きでコンパイルし、
対象5件だけを実行したLCOVと開始版のローカル `origin/main` の変更行を突き合わせた。
変更された実行行267行中254行を通り、局所診断は95.13108614%だった。
未実行13行はcache/具象modelの配線とignored入口に残る。
これはworkspace全体を実行するCIのcargo-llvm-cov成功ではない。
ローカルdiff-cover launcherはPython moduleがなく起動できず、LCOVのDA行とGitの差分を照合した値である。
ホストの既定coverage jobは元のコマンドで実行し、最新mainとの同じheadのgateを確認する。
局所Rustログ・LCOV・差分診断はcheckout外の `/tmp/rurico-367-diagnostic-coverage` に保存した。

固定版natural-japanese `9a78a42964096da509b8f3e011f0085a5f080151` を照合して追記を通読した。
lintは一時cacheとoffline指定で試したがSudachiPyがなく未完了だった。
手動観点で文の主旨・条件・数量・断定・出典を確認し、lint成功とは扱わない。

### 保存実測の適用性と引継ぎ

旧host-returnの14記録と追加証拠本体、追加証拠の39記録のhashを照合した。
4モデルの生Rust出力と独立した公式参照を現行 `variants.evaluate` で再集計し、
保存comparisonとの完全一致を確認した。
[小結果JSON](results/model-variants-20261012.json)のRust出力/comparison hash、件数、model/revision、
開始commitとCargo.lock、各environmentの入力hashを照合した。事前planの基準も維持する。
310mは保存Rust観測とfresh公式CPU参照のmodel manifestが同一であり、分離実行のまま扱う。
推論演算、入力、参照依存、比較器と数値基準は変更していない。
今回のRust差分は選択・次元照合・観測記録・ログの責務とモデル不要テストであり、
保存数値は上記optionと固定依存に対する証拠として適用できる。再集計は新しい推論成功ではない。

開始版と引継ぎreport/README blobは前節の値を再照合した。
旧reportは同一bytesで、現行READMEはモデル別の実行と通常laneの説明を更新した版である。
acceptedなADR-0011/0014、未採用の#315案と、既存PRの根拠リンクは前節の参照を保つ。
既存PR本文と保存本文にアップロード済み媒体リンクはなく、captureは未設定でUI媒体も不要である。
次の独立評価ではPR全体と今回の修正・変更文書を照合し、以前の受入や指摘IDを再利用しない。
新しいPR説明には4モデルの結果、開始版と固定依存、310mの分離実行、token_budget=8192と
短文の既定batch、全体入口の連続完了・既定複数長文・SIGKILL・長時間安定性の未確認を残す。
旧310m報告・#366記録・requirements・README/CONTRIBUTING・ADR・#315の公開リンクも保持する。

設定済み `bash scripts/check.sh` とtest/coverage/security/zizmorはホストが変更後の同じheadで確認する。
今回のcheck/capture契約、coverage閾値・除外regex・feature・timeoutは変更していない。
commit・push・公開と新しい実モデル測定は行っていない。
