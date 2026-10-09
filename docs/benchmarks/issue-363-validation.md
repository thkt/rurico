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

## 次の評価への引き継ぎ（handoff）

合意範囲は[Issue #363](https://github.com/thkt/rurico/issues/363)のままで、差分基準は開始commitから変更しない。
[CONTRIBUTING](../../CONTRIBUTING.md#推論境界の回帰検証issue-363)の通常checkとignored検証の分担も変更していない。
設定済み`bash scripts/check.sh`は修正後にホストが実行し、R1-1・R1-2とこの文書を独立評価し直す。
媒体は不要でcaptureは未設定のまま。修正後の全体check・独立評価・同じheadのCIはこの局所修正の時点では未確認である。

上記の順序保存と変更箇所の照合により、今回の局所修正に追加の実モデル再測定は必要と判断していない。
過去のignored演算検証は変更のない対象への観測、固定モデル測定は同じforward入力への観測として扱い、
修正後の実モデル実行や性能改善とは主張しない。
#306の報告は開始版blob`fe49cc333e50bfa40273b9ca5ca3425d74547b30`、
#307の報告は開始版blob`7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`と現在の内容が一致する。
[#306](issue-306-metrics.md)は既存fixture・options検証の再利用根拠、
[#307](../research/issue-307/report.md)は公式比較の過去観測として参照し、新しい成功へ読み替えない。
最新mainとして保存された版のソース比較、RSS・真のGPUメモリpeak、未知・短時間GPU活動の排除、
全入力の性能・品質は未確認のままである。公開・consumer変更・許容差変更は行っていない。
