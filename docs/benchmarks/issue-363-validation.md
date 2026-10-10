# #363 演算・入力順・委譲の検証

2026-10-10、固定310mモデルとW1/W2/W3の既存fixtureによる全組合せの測定が正常終了した。
これは下記の測定時版の記録であり、末尾の局所修正後に実モデルを再実行した結果ではない。
97回の推論、24グループそれぞれ3回のwarm反復、24件の集計が揃った。
[raw](issue-363/isolated-20261010/raw.jsonl)内の集計と、同じbinaryで再生成した
[summary](issue-363/isolated-20261010/summary.jsonl)はJSON値として完全一致した。
[検証結果](issue-363/isolated-20261010/validated.json)と
[版・隔離条件](issue-363/isolated-20261010/provenance.json)も保存した。

## 対象と確認した契約

測定時版は開始commit`24725a72be44300afc24186b82b14bcb3f5f9d3d`と測定時の未commit差分。
ファイルごとの内容hashは[provenance](issue-363/isolated-20261010/provenance.json)の`source_files`を参照する。
最新mainは`a769df9005cddd329911b5c909624530a2c654b9`で、今回の測定対象とは区別する。
Cargo.lockのSHA-256は`743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。
使用したrelease binaryのSHA-256は`9a05055ce74be432164340968357dff1536b6fe1f1c9f138c341de6c1dd3138c`。
固定embedding revisionは`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`。

default optionsと`token_budget=256, forward_pause=1ms`で、batch/singleton、通常/計測APIを比較した。
各呼出しで既存fixtureの文書順・chunk構成・数値条件を検査し、fixtureを再生成していない。
計測APIのreadback量は各forwardの`batch_size * 768`と一致し、pause回数は設定時のforward数と一致した。
数値条件・テストのtimeout・coverage除外・既定model・optionsの契約は変更していない。

同じ変更版で実行済みのignored MLX検証5件も、対象sourceの内容hashが変わっていないことを再確認した。
手計算のmasked mean→L2、zero norm、全zero mask、大shape、小さい2層modelのeval後の有限性・padding不変性・truncateを確認する。
入力順・floor budget・端数の製品経路と、内容・順序・エラー・単回委譲のspyを組み合わせる。
plannerコピー、単独Mutexのテスト、#307の公式比較は複製していない。

## 測定枠と版の照合

ユーザーが回答後15分間、他のビルド・GPU作業を止めて測定する枠を承認した。
回答確認時刻から00:55:54〜01:10:54 JSTを記録し、担当AIのbuild/check/actor/GPU作業を重ねなかった。
事前監視と測定は00:57:08〜01:09:21 JST、約12分13秒で終了した。事前監視10秒を含む時間である。
0.25秒間隔を目安とするprocess監視は2,424回で、既知の競合検出は0回。
競合・監視失敗・枠終了時は自分の測定を停止する条件を用意し、今回の測定では発動しなかった。

source・fixture・Cargo.lock・binaryは測定前後で同一。固定modelの全内容hashも前後で一致した。
保存済みのrelease binaryを使い、測定後に同じcheckoutで標準release buildを再実行した。
生成binaryが測定したbinaryとbyte単位で一致することを確認し、古いbinaryと異なるsourceを結び付けていない。
再buildは承認された測定枠の終了後であり、測定中には行っていない。
この測定記録を最初に保存した時点では、測定後の更新は説明と記録の追加だけだった。
その後の局所修正は末尾に記し、測定時版と区別する。

## テスト整理と観測の限界

shape/unit normだけの検査と単独u32検査を手計算検査へ統合し、固定scoreの委譲検査をspyへ置換した。
大shape、全zero mask、emptyで初期化しない条件、OnceLockの並行初期化は維持した。
旧人工shuffleのsort自体の能動性と768次元の合成forward検査は失う。
製品の入力順復元、小さい実演算、大shape pooling、今回の固定768次元model/fixture検証を根拠にする。
テスト件数やcoverageを成果の代用にせず、実行時間・安定性・保守費用の改善量は主張しない。

以前の全測定には外部buildの重複があり、機能観測として保存したまま、今回の隔離結果へ読み替えない。
前回の自動承認レビュー拒否では測定processを起動していない。
今回もprocess名の定期観測だけでは、未知のGPU利用や短い負荷の不存在を証明できない。
ユーザーが確保した枠と観測を合わせた条件であり、OS cacheを消したcold測定ではない。
RSS・真のGPU物理メモリpeak・OOM不存在・全入力での速度や品質を保証しない。
これは#363の実モデル処理経路の確認で、#307の公式数値比較やamici検索品質の新しい測定ではない。

再実行は[CONTRIBUTING](../../CONTRIBUTING.md#推論境界の回帰検証issue-363)の手順を使い、
source/model/binaryのhashと隔離条件を新規保存先へ記録する。

## 局所修正の再評価（assessments）

この節は公開commit `74b9ff2e3bc5af10bd4212b6fb8a8856419d381a` に至る局所修正時点の記録である。
ホストの独立評価対象`d682cdb8345ee49af2ed6e49efde24666e5bb2764491e215c5a0f46dc61b1797`に対する
必須指摘R1-1とR1-2を、測定時版と現在のソースを比較して修正した。
評価記録はホストの`/private/tmp/rurico-363-host-review-20261010/verification/review-1.json`に保持されている。
過去の成功や証拠の`passed`を受入判定として引き継いでいない。

R1-1ではbucket内sortとその専用計測、sortキーだけに使う文書・chunk位置のフィールドを削除した。
`build_indexed_chunks`は文書・chunkの順に生成し、`distribute_into_buckets`はその順でpushする。
そのため現在の唯一の製品経路では、削除前後のsub-batch入力とforward回数は同じになる。
元順への格納に必要な`global_idx`、bucket値、readback検査、cleanupとpauseは保持した。
位置フィールドの検査は、元のtoken内容と`global_idx`の対応を確認する検査へ更新した。

R1-2では旧`t_bkt_009_empty_input_zero_subbatches`とproxy説明を削除した。
同じ空tokens・countsから同じhelperを呼ぶ製品経路テストが、空結果・forward抑止・pauseなしも確認するため、
旧テストに固有の現実的な検出条件はない。失うのは包含済みのhelper単独観測である。
複数bucket・sub-batch・端数から異なるrowを復元する期待値と、失敗forwardでpauseしない検査は保持した。
公開計測APIの実モデル空入力検査も変更していない。追加テストは設けていない。

修正後のモデル不要のprocessing検査23件は成功し、fmtと差分の空白検査も成功した。
これは製品の組立境界の確認であり、MLX演算を実行した結果ではない。
日本語は固定版`9a78a42964096da509b8f3e011f0085a5f080151`の手動チェックリストで確認した。
同版のlintは依存取得時のDNSエラーで実行できず、クイック手順の機械検査は未完了である。
旧テストの単独表示24msはprocessなどを含む過去の値で、check全体の短縮量ではない。
実装・テスト・文書を同条件で測定した整理前後の総時間、安定性、保守費用の改善量は未測定である。

追加証拠の生ログと保存hashを照合し、24集計と各3反復、readback量・pause回数を確認した。
局所修正前のRust・fixture・Cargo.lockは測定manifestと一致していた。
修正後はprocessing本体・そのテスト・metricsの説明が測定manifestと異なるため、同じbinaryの証拠とは扱わない。
pooling・MLX forward・readback・cleanup・モデル・fixtureは変更しておらず、
生成と振分けの順序保存、および修正後の製品経路検査を根拠に、過去の演算・fixture観測をこの局所修正に利用する。
時間の内訳ではsort専用区間を削除したため、保存されたpreprocessing時間を現在版の時間として引用しない。

## 公開後のCI不備への修正

公開commit `74b9ff2e3bc5af10bd4212b6fb8a8856419d381a` の
[coverage job](https://github.com/thkt/rurico/actions/runs/37960680950/job/113922437326)は、
processingの変更59行のうち121・130・133行が未計測で94.9%、変更行95%の条件を満たさなかった。
121行はpause計測分岐の末尾、130・133行は欠落slotのエラー生成経路である。
通常test・security・zizmorは成功した。GitHub merge commitのRust419件とローカルPR版408件は別の結果である。
これらの成功を修正後のcheck・coverage成功として扱わない。

新しい境界ケースは、1文書分の行を返した後、次のsub-batchの行が欠落する条件を作る。
製品の組立から欠落位置を含むエラーを返し、先行文書だけの部分結果を公開しないことを確認する。
MLXの`split_pooled`にはbuffer長の既存検査があるが、検証済み行の返却・格納を変えた際の
行落ちを組立境界で検出する保証はその単体検査に含まれない。欠落slotの防御を維持し、重複した検査は追加しない。
このケースは非計測経路と0秒pauseを使うため、実GPU・モデル取得・待機時間を必要としない。
全体の実行時間、不安定さ、保守費用の改善量は未測定である。

既存の先頭forward失敗ケースは、1文書成功後のforward失敗へ更新した。
元の`NonFiniteOutput`を保持し、残る文書のforwardを呼ばず、成功したforwardだけpauseすることを確認する。
先頭で失敗する単独観測は失うが、同じエラー伝播と失敗時pause抑止を途中失敗で確認する。
正常時の文書/chunk/row復元、budget=383のfloor分割と端数、空入力のforward・pause抑止は維持した。
実装の状態や公開APIは追加せず、不要なhelperのテスト向け説明とテストの重複コメントを削除した。

演算・forward入力・readback・cleanup・モデル・fixture・Cargo.lockは公開版から変更していない。
測定済みの演算と固定入力の観測は引き続き過去の証拠として利用できるが、修正後の実モデル測定ではない。
追加GPU測定は開始せず、15分の測定枠と測定後の再buildという記録も変更していない。
公開前にlintを実施できなかった事実は、前節に当時の記録として残す。
採用済み修正要求では、その後ホストが修正後報告に固定版lintを実施し、情報TTRのみ、自然度欠陥なしと報告している。
CI不備への初回修正でも固定版のlintを実行し、情報TTRのみで自然度欠陥はなかった。
読解負荷の候補は、助詞の連続を修正し、検証条件・限界をまとめた既存の列挙は文脈上必要として残した。
原資料と意味を照合した更新文書を、checkと既存の独立評価へ渡す。

### 独立評価で確認したテストの重複

CI修正の評価対象`f86e41f2044cd745b6a72021f32de9eaa46418ea17ec47bb4c7c3f9250e50cda`に対する
採用済み指摘R1-1は、LazyRerankerの成功初期化後の混合呼出しを二重に検査している点である。
記録はホストの`/private/tmp/rurico-363-ci-repair-20261010/verification/review-1.json`に保持されている。
これは前節の公開前評価とは別の評価であり、その受入や指摘IDを引き継がない。

入力委譲をspyへ強化した際に、初期化回数だけを確認する旧`first_call_initialises_then_caches`を残したことが原因だった。
現在の各入口は同じ`inner`から`OnceLock::get_or_init`を通り、初期化成功後にwrapped methodへ委譲する。
既存spyも成功初期化後に三つの入口を反復し、wrapped methodのエラー後も初期化が1回であることを確認する。
旧テストを削除し、spyの初期化回数・入力内容・順序・返却値・エラー・呼出し回数の検査を維持した。
失うのは包含済みの成功キャッシュ単独観測であり、固有の故障検出条件は失わない。
構築時未初期化、初期化失敗の保持、並行初期化、空入力抑止の検査も維持する。

同様の重複は検証を強化する際に再発し得るため、追加assertだけでなく同じ初期化経路かを比較した。
新しいhelper・状態・テスト・lint規則は不要と判断した。現行の検証入口や設計契約は変わらない。
小さい既存テストの削除であり、実行時間・不安定さ・保守費用の改善量は測定していない。
測定manifestとの差には今回の`src/reranker/lazy/tests.rs`の削除も加わるが、製品の演算やforward入力は変更していない。
今回更新した文書は固定版の手動チェックリストで原資料と意味を確認した。
同版のlintはSudachiPyの依存取得時にDNSエラーで終了し、機械検査は未完了である。
これは過去のlint成功とは別の結果である。fmtと差分の空白検査は成功した。
今回の全体check、独立評価、変更行coverageの結果はまだ確認していない。

## 次の評価への引き継ぎ（handoff）

合意範囲は[Issue #363](https://github.com/thkt/rurico/issues/363)のままで、差分基準は開始commitから変更しない。
[CONTRIBUTING](../../CONTRIBUTING.md#推論境界の回帰検証issue-363)の通常checkとignored検証の分担も変更していない。
修正は公開commit `74b9ff2e3bc5af10bd4212b6fb8a8856419d381a` から開始した。
設定済み`bash scripts/check.sh`は修正後にホストが実行する。
旧評価の受入や指摘IDを再利用せず、Issue全体・今回採用されたCI修正・変更文書を独立評価する。
媒体は不要でcaptureは未設定のまま。今回の全体check・独立評価・同じheadのCIは未確認である。
旧runは再開せず、別runで同じ[PR #383](https://github.com/thkt/rurico/pull/383)を修正する。
accepted評価のassessmentsとhandoffには、現在も適用する責任分担、六つの完了条件との対応、
未確認事項、上記raw・summary・validated・provenanceの公開済みリンクを明示的に残す。
文書一覧の出典だけで証拠リンクを置き換えない。古いcheck・CI・独立評価の成功を現在版へコピーしない。

上記の順序保存と変更箇所の照合により、今回の局所修正に追加の実モデル再測定は必要と判断していない。
過去のignored演算検証は変更のない対象への観測、固定モデル測定は同じforward入力への観測として扱い、
修正後の実モデル実行や性能改善とは主張しない。
#306の報告は開始版blob`fe49cc333e50bfa40273b9ca5ca3425d74547b30`、
#307の報告は開始版blob`7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`と現在の内容が一致する。
[#306](issue-306-metrics.md)は既存fixture・options検証の再利用根拠、
[#307](../research/issue-307/report.md)は公式比較の過去観測として参照し、新しい成功へ読み替えない。
最新mainとして保存された版のソース比較、RSS・真のGPUメモリpeak、未知・短時間GPU活動の排除、
全入力の性能・品質は未確認のままである。公開・consumer変更・許容差変更は行っていない。
