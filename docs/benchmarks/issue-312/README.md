# Issue #312 正規化とquery内vocab照会の比較

正規化済み入力のallocationと、重複短語のvocab照会を削減した。検索意味・公開String API・identifier/schemaのエラー契約は保持する。すべての入力の高速化を示すものではなく、変換が必要な正規化入力には時間増加も観測した。

## 要求と参照版

要求と合意の正本は [Issue #312](https://github.com/thkt/rurico/issues/312)（依頼時の更新時刻2026-10-07T17:05:19Z）。[親Issue #296](https://github.com/thkt/rurico/issues/296)（確認した更新時刻2026-10-07T17:08:32Z）のセットアップ・証拠保存方針を適用する。#314のphrase/fallback設計、DB状態をまたぐcache、consumer移行は今回の範囲外。

開始commitと比較基準は `24725a72be44300afc24186b82b14bcb3f5f9d3d`。開始時のcheckoutとローカル`origin/main`は同じcommitだった。初回のGitHub main追加取得は通信失敗だったため、指定された開始版を基準にした。その後ホストが取得したmainは同じ `24725a72be44300afc24186b82b14bcb3f5f9d3d` と確認された（今回の固定入力による引き継ぎ）。評価担当自身による取得や性能検証を意味しない。Issueの現行根拠`c8f250d60a5afb9944b9008d22ad6f4dda2d7103`から、対象のsearch/query_normalize本体・テストと[共有済み#307報告](../../research/issue-307/report.md)には差分がない。報告の開始版blobは `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`。その2026-09-27の固定入力・モデル・amici構成の数値は過去の証拠として維持し、今回の検索品質・性能の実測へ読み替えない。

[ADR-0001](../../decisions/0001-typed-fts-query-contract.md)（開始版blob `063be37733cd61524044fa9b6bd726b05589a5ab`、accepted）の型付きMATCHと、[ADR-0012](../../decisions/0012-adopt-symmetric-phase-5-query-normalization.md)（blob `27f2d15c5279aa5fcc0b81c23074571482577c2e`、accepted）のindex/query対称性・旧baseline全OFFを維持する。#297の引用修正とamici wire-formatは変更しない。親Issueに引用された旧checkコマンドは当時のfeature構成であり、現行の[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)・`scripts/check.sh`にはsmokeとモデル不要laneの選択検査が追加されている。今回の契約にはその現行入口をそのまま使う。

## 採用した処理と比較条件

内部CowはNFKC結果が同じ入力を借用し、ASCII lowercaseが必要な場合だけ所有化する。NFKC quick-checkがYesなら借用し、Noなら変換する。Maybeの場合は1本のNFKC iteratorで比較を進め、最初の差で一致済みのbyte prefixをコピーし、そのiteratorの残りを出力へ引き継ぐ。空白処理は単語を1回走査し、元入力内の期待位置と単語の開始位置を比較する。非canonicalな区切りまたは末尾空白がある場合だけStringを作る。公開`normalize_for_fts`は所有Stringを返し、query経路だけが借用を利用する。

短語は呼出し内のHashMapで再利用し、出力の重複語とANDは残す。値は同じ呼出しの出力partsのindexで、余分な結果Stringコピーを保存しない。短語がある場合はcached statementを実行し、SQLiteのschema再検査を受ける。長語のみの場合はcached statementを使わず、`SELECT term, cnt FROM vocab WHERE 0`相当の検査statementをprepareしてstepする。行を取得せずterm/cntとschema cookieを検査するため、別接続で呼出し前に確定したschema変更も観測する。短語経路にはこの検査を重ねない。最初の短語の位置を保持し、それより前は引用、その位置は短語処理へ進める。引用ループの文字数/operator判定は境界より後のtokenにだけ行い、短語なしの場合も再判定しない。検査を省く遅延prepare、cache全体の破棄、全vocab SELECTは採用しない。線形探索案は、短語の種類が多い入力に二乗の探索費用を増やすため採用しなかった。HashMapの割当・hash計算費用は残る。

2026-10-08、arm64 macOS 27.0.1 (26A434)、Rust/Cargo 1.99.0、Xcode 27.0 (27A266a)、Metal 32023.921を確認した。機種・メモリ容量と背景processはsandboxから取得できなかった。計測はCPUのみで、GPU・モデル・ブラウザー・サーバーを起動していない。

[probe.rs](probe.rs)は製品sourceをbyte単位でコピーした変更前後のmoduleを同じbinaryへコンパイルする。Rust側は`-O -C lto`、両版に同じ既存release dependency rlibを使った。既存buildのCargo.lockは今回とSHA-256が一致し、版は変更していない。初回測定のsource・lock・dependency・binary hashは[manifest.json](manifest.json)、R1修正版は[開始版比較manifest](repair-1/start/manifest.json)と[初回評価対象比較manifest](repair-1/prior/manifest.json)、R2修正版は[開始版比較manifest](repair-2/start/manifest.json)と[前回評価対象比較manifest](repair-2/prior/manifest.json)に保存した。R3修正版は[開始版比較manifest](repair-3/start/manifest.json)と[前回評価対象比較manifest](repair-3/prior/manifest.json)で特定する。コメント整理前のR4測定版は[開始版比較manifest](repair-4/start/manifest.json)と[前回評価対象比較manifest](repair-4/prior/manifest.json)で特定する。これはworkspace全体のrelease buildやMLX測定ではない。最初のprobe linkはLTO指定不足によるApple linkerのbitcode版差で失敗し、`-C lto`を加えて新規出力先で再実行した。失敗した実行を成功として数えていない。

SQLiteのin-memory unicode61 FTS5へ1000文書（`audit0000 authentication0000 login`等）を投入し、更新後に`autumn`を追加した同一DBを使う。vocabは2002種類。変更前後のserialized queryを更新前後で照合する。NFKC/ASCII/空白は、空・Unicode・ASCII・記号等8入力×全8設定で変更前後を照合し、固定literal期待値と冪等性は通常テストでも確認する。

各組合せで20回warm-upし、正規化は10000回、queryは1000回の区間平均を7試行取得した。版順は試行ごとに反転し、7回なので完全均等ではない。DB生成・print・照会traceは時間区間外。allocationは別の1呼出しでRust GlobalAllocのalloc/reallocを数え、SQLite内部のC allocation、bytes、RSS、Metalメモリは測っていない。背景負荷の隔離は未確認で、細かな速度差は効果と断定しない。cold diskや初回prepare性能も保証しない。

## 初回版の観測結果と限界

この節の[raw.jsonl](raw.jsonl)・[manifest.json](manifest.json)・[table.md](table.md)は初回評価対象`f3ef9e8435ad20db1ed1730b1e418c01befc6a8801fa7b47164458d21ffbf6d2`の過去の観測であり、現在のsource hashとは一致しない。schema検査の不足とMaybe経路の二重正規化が残っていた版で、要求充足の証拠にはしない。初回probeは[initial-probe.rs](initial-probe.rs)に保存し、そのhashは当時のmanifestと一致する。現在のprobeへ黙って置き換えない。修正後の根拠は「初回評価後の修正と再測定」に示す。

[raw.jsonl](raw.jsonl)が正本で、[table.md](table.md)はその7試行の中央値・最小最大とallocation回数を表示する。正規化のasciiは`react hooks`、unicodeは`日本語 カタカナ`、changedは`ＡＢＣ　Foo  Bar`、spacesは空白・tab・改行・全角空白。queryのlongは`authentication login`、uniqueは`au zz`、repeatedは`au au au au`、repeated_missは`zz zz zz zz`。

SQLite traceの実vocab SELECT回数はlong 0→0、unique 2→2、repeatedとrepeated_missは4→1。prepareの検査を省略して回数を減らした結果ではない。TEMP B-TREE等のEXPLAIN表示から全vocab走査とは断定していない。

ascii正規化のallocationは4→1、unicodeは5→1、long queryは26→21。重複短語queryの中央値は928077→231622 ns/call、重複missは767919→193823 ns/callだった。一方changed正規化は213.02→235.18 ns/call、spacesは67.59→78.80 ns/callで時間が増えた。unique queryは426409→432708 ns/callで分布が重なり、速度改善とは判断しない。入力の出現率は未測定で、一般的な検索遅延やアプリ全体の改善量は推定しない。

## 検証の価値と残る確認

既存の#297 literal引用・日本語展開・実MATCH、正規化の設定/冪等性、missing vocab・identifier/schema検査を再利用した。既存`expand_special_chars_escaped`の%負例を残し、独立したliteral vocab一覧とtrigramの実MATCHを同じテストへ統合した。`_`がwildcardに化けて別文書を含む不具合、backslashがescapeとして消えて候補を失う不具合を防ぐ。両escapeを別々に削除した一時コピーでは、このassertionが終了101で失敗した。%について合意記録にある既存検出を重複テストとして追加しない。

既存のidentifier/schemaテストに長語のみと重複短語を加え、検査を省く遅延化を防ぐ。追加したDB更新テストは、呼出しをまたぐ古い展開結果を返すcacheを防ぎ、更新後の実MATCH・DROP後fallback・壊schema拒否を確認する。全8設定の混合Unicode/記号テストは、NFKCの結合文字とASCII限定lowercase、Unicode空白の境界で意味がずれる失敗を防ぐ。期待値は固定literalで、製品変換から作っていない。

初回版の対象moduleの既存検証を再利用したCPU-only test harnessは[53件成功](tests.txt)（表示時間0.01秒）。温まったcacheの長語条件は当時検証しておらず、下記の修正後検証とは区別する。ネットワーク・モデル・時間閾値を持たず、追加保証に対する実行費用と不安定さは小さい。テストを削除せず、%検査を拡張したため失った検出条件はない。件数・coverageだけを根拠に維持したものではない。通常suiteの速度改善は測っておらず、対象harnessの時間を全体checkの短縮量とは扱わない。

対象sourceのClippy静的検査とfmtを確認した。isolated harnessだけで使われない公開APIのdead_codeはClippy時にallowしたが、repositoryのlint・check・coverage条件は変更していない。標準checkはホストが`bash scripts/check.sh`で実行する。通常CIのtest/coverage/security/zizmorも別の契約として維持する。全workspace・FFI/MLX・同じheadのCIはこのsandboxで未実行。今回の性能probeはcheckの代替ではなく、実モデル数値やamici検索品質の再測定も主張しない。今回の内部FTS最適化にはUI媒体は不要でcapture=nullを維持する。

## 初回評価後の修正と再測定

この節の「修正後」「現在版」はR1修正時の評価対象`9fed7e3b57063f54c88ccb08905114b1ca2d5c5bc2cec4089f1f1cd73328e519`を指す過去の記録である。新規prepareだけでは別接続のschema変更を検査できないことがR2で判明したため、現行のschema保証とsource hashは次節へ更新した。R1の測定・probe・失敗/成功ログは保持する。

初回評価R1-1/R1-2の指摘を、上記manifestの4 source hashと一致する修正前コピー、および固定依存版rusqlite 0.40.2の`StatementCache::get`とunicode-normalization 0.1.25の`is_nfkc`へ照合した。cache hitは再prepareせず、Maybe判定は内部でNFKC iteratorを消費する。前者は長語でstatementを実行しないためschema変更を見逃し、後者は不一致時に同じ入力を再正規化していた。合意済みのschema検査と呼出し内の費用削減に限定して修正した。

既存schemaテストへ、有効vocabの短語呼出しでcacheを温める→cntのないschemaへ変更→長語呼出しが`VocabLookupFailed`となる条件と、その後のDROP時のliteral fallbackを統合した。最初の再現fixtureは既存helperのvocab名と不一致で準備に失敗し、Redとは数えていない。fixture訂正後の修正前実行は、[エラー拒否のassertで終了101](repair-1/schema-red.txt)。修正後は同じassertを含む[対象53件が成功](repair-1/start/tests.txt)（0.02秒）し、[対象Clippy](repair-1/clippy.txt)とfmtも成功した。NFKC-onlyの既存テストに、多byte prefixの結合、文字途中の結合、結合文字の並べ替え、Maybeでも変更不要な入力の固定期待値とBorrowed確認を統合した。出力保証に加え、prefixの切断・suffixの欠落・不要な所有化を検出する。二重正規化の費用は時間閾値テストではなく、iteratorの処理経路と以下の実測で確認する。

同じprobe・入力・rlib・flagsで、[開始版→修正後](repair-1/start/table.md)と[初回評価対象→修正後](repair-1/prior/table.md)を別々に実行した。各実行のraw・manifest・対象テスト結果は同じディレクトリへ保存した。Cargo.lockと5 dependency hashは初回測定と一致する。初回評価対象はGit commitではなく成果物hashで、[prior-source.patch](repair-1/prior-source.patch)は開始commitからその4 sourceを復元する差分である。現在版は両manifestの`current/` source hashで特定する。

追加入力はMaybeかつ変更が必要な`e`＋U+0301、`日本語 prefix `を128回繰り返した末尾の`e`＋U+0301、変更不要な`x`＋U+0301。期待出力はそれぞれ`é`、同じprefix＋`é`、入力そのままという固定値で検査する。試行数・warm-up・区間回数・交互順・allocationの定義は初回と同じ。ビルドと実行を直列に行ったが、機種・背景負荷の隔離は引き続き未確認で、速度保証にはしない。

初回評価対象との比較で、短いMaybe変更入力の中央値は97.10→70.22 ns/call、長いMaybe変更入力は38164.19→24224.76 ns/call、後者のRust allocationは10→1だった。一方、長語queryは676.38→1952.29 ns/callで増加した。schemaの鮮度を検査する費用を省いていた修正前の速度を、受入可能な改善とは扱わない。開始版との比較でも長語queryは937.71→1980.08 ns/call、長いMaybe変更入力は17468.56→28066.96 ns/callで増加し、後者のallocationは19→1。他の入力の分布も表に残し、都合のよい入力だけを性能保証へ使わない。

修正後も開始版との実vocab SELECT回数はlong 0→0、unique 2→2、repeated/repeated_miss 4→1。初回評価対象との回数はすべて維持した。DB更新前後のserialized query照合と、既存の実vocab/literal MATCH・DB更新テストを保持する。テスト削除はなく、失う検出条件はない。既存fixture・テストへ条件を統合し、ネットワーク・時間閾値を加えず、通常suite全体の費用や将来の保守削減量は未測定。過去の53件成功では温まったcacheの長語条件を守れなかったことを訂正し、全体checkと独立再評価は変更後の成果物に対してホストが行う。


## 二回目評価後の修正と再測定

この節はR2修正時の評価対象`23c720e2591c6367b914758e2014e67d7d0bf2f3b8c523b728057746afeb6865`の過去の記録である。schema保証は維持するが、混合queryの判定済みprefix再走査はR3で修正した。現行sourceと測定根拠は次節を参照する。

ホストの初回評価`review-1.json`と二回目評価`review-2.json`（保存先: `/private/tmp/rurico-implement-312-20261008/verification/`）、attempt 1の修正結果、check-1/check-2を比較した。評価ログはホスト内であり公開repoからの参照ではない。attempt 1のstdout SHA-256は`f32bacdae1dcb834aa23334de44535b4dcc519b19a3701bdaccb2b993e38c8e2`。前回sourceの4 hashはR1の両manifestに一致した。R1-1は同一接続のschema変更を直したが、長語経路でstepしないため接続内schemaの鮮度を確認できなかった。新規prepareへの変更だけでは保証が足りず、R2-1では行を取得しない検査statementの実行へ変更した。R2-2は不変のtoken群に対する既存の`has_short_term=false`を引用分岐で利用し、重複分類を省いた。公開API、短語のcached lookup、呼出し内memo、missing vocab fallback、引用順序は維持する。NFKCのR1-2修正には変更を加えていない。

新しい回帰条件はfile-backed DBを通常の2接続で開き、Aの実fts5vocab照会→BのDROP/cnt欠落テーブル作成確定→Aの長語呼出しという順序である。間にAのschemaを更新するSQLは挟まない。正常な展開と壊schema拒否、別接続でDROP後のliteral fallbackを同じテストにまとめた。固定依存版rusqlite 0.40.2 / SQLite 3.53.2を使うRust APIで、[修正前](repair-2/schema-red.txt)は狙った拒否assertで終了101、[修正後](repair-2/tests.txt)は54件成功（表示0.02秒）。R1の同一接続テストは異なるcache境界を守るため残す。R2-2の引用結果は既存の長語・混合語・operator検証で守れ、追加テストは作らない。新規回帰テストは一時ディレクトリでDBを隔離・削除し、ネットワーク・待機・時間閾値を追加しない。ファイルI/Oとfixtureの保守費用は増えるが、既存in-memoryの同一接続検証が見逃す確定済みDDLの不具合を検出する価値がある。通常suite全体の追加実行費用は未測定。削除したテストや失った検出条件はなく、既存のliteral vocab・実MATCH・DB更新・全8設定を保持した。[対象Clippy](repair-2/clippy.txt)と変更したRust sourceのfmtも成功した。standaloneでだけ未使用となる公開APIのdead_code allowは、repoのlint条件へ適用していない。

[今回のprobe](repair-2/probe.rs)は変更したquery経路だけを再測定する。正規化の設定同値性・Maybeの固定期待値はprobeでも照合するが、正規化工程単独の時間を重ねて測らない。入力は従来4 queryに、`日本語`を1024回繰り返した長語を追加した。両版のserialized MATCHはDB更新前後で一致した。1000文書・2002種類のvocab、20回warm-up、1000回の区間平均×7試行、版順反転、Rust allocationの別観測、同じlocked dependency rlibと`-O -C lto`の条件はR1と同じ。DB作成・trace・printは時間区間外。背景負荷の隔離は引き続き未確認で、測定中の外れ値も削除しない。現行4 source・lock・probe・binaryは[前回版manifest](repair-2/prior/manifest.json)と[開始版manifest](repair-2/start/manifest.json)、テストbinaryと追加のtempfile依存は[test-manifest](repair-2/test-manifest.json)で特定する。前回評価対象のコピーは[差分](repair-2/prior-source.patch)で開始版から復元でき、R1の`current/` hashへ照合した。

[前回版比較](repair-2/prior/table.md)では長語queryの中央値2141.58→1590.83 ns/call、allocationは20→20。ただし現行側に26836.46 ns/callの試行があり、一般的な速度改善とは判断しない。長いUnicode語は67693.04→73418.54 ns/callで増加し、分布が重なる。schema検査と分類省略を合わせた変更の観測であり、R2-2単独の時間効果は測っていない。[開始版比較](repair-2/start/table.md)では長語925.25→1430.38 ns/call、長いUnicode語42622.88→67445.75 ns/callで増加し、allocationはそれぞれ26→20、29→14。重複短語は1001572.79→240044.33 ns/call、576→162 allocations、重複missは812511.83→214346.17 ns/call、48→30 allocationsだった。uniqueと他の全試行はrawと表に残す。語彙・入力分布・全モデル・利用側品質への推定は行わない。

traceで観測したprefix SELECT回数は、開始版との比較でlong/long_unicode 0→0、unique 2→2、repeated/repeated_miss 4→1。行を取得しないschema検査のSELECTはlong/long_unicodeで各呼出し1回、短語のあるqueryでは0回で、prefix lookupとは別にrawの`schema_validations`へ記録した。SQL実行が一切ないという意味ではない。全vocab走査やcache全体の破棄は追加していない。通常check・独立再評価はこの変更後の成果物をホストが実行する。過去checkの成功は現行版の成功とせず、契約`bash scripts/check.sh` / capture=nullは変更しない。

## 三回目評価後の修正と再測定

この節はR3修正時の評価対象`a77b057b0f2e33bfb648b73b012fcf67d8da677bfea61e21e4d4b514ccc7d930`の過去の観測である。空白処理の内容再比較が残っていたため、現行版のsourceと測定根拠は次節へ更新した。保存済みR3のprobe・raw・manifestは維持する。

R3-1は、R2の`has_short_term=false`だけを再利用する修正では、混合queryの既判定prefixが再走査されることを示した。ホスト内の`review-1.json`〜`review-3.json`、attempt 2のfindings・sourceBefore/sourceAfter・stdoutHash、check-3の保存ログを照合した。attempt 2のsourceBeforeは`987fd42f9b44a22ff1da6e9c654b42805e27d24c782f24d239eaebcca3069f01`、sourceAfterは`e1c36305a7764164e163797ac38ad57044bdf1e2f99474d04b1f1d2f1c73167b`、stdout SHA-256は`2ffd24b062117d944a7ce8900af845b8dc47ed02ce728da05c23a6260817de36`。check-3はそのsourceAfterに対して終了0で、保存stdout/stderrのhashも評価対象記録に一致する。修正開始時の4 sourceはR2 manifestの`current/`と一致した。R1-1/R1-2/R2-1/R2-2が解消済みという過去の判定を維持し、check成功をR3-1不存在の証拠にはしない。

最初の短語を探す処理を`any`から`position`へ変更し、該当なしはtoken数を境界とする。境界前のtokenは引用対象、境界のtokenは短語と確定しているので再分類しない。境界より後だけ文字数/operatorを調べる。分類配列・追加helper引数・工程間cacheは導入していない。identifier検査→statement prepare/必要なstep→出力構築の順序、長語のschema検査、短語cached lookup、呼出し内memo、引用とmissing vocab fallbackは維持する。NFKC処理と公開APIには変更がない。

[比較probe](repair-3/probe.rs)はR2の全query条件を残し、`日本語`を1024回繰り返した長語に` au`を続ける`long_unicode_then_short`を追加した。[前回版との差分](repair-3/prior-source.patch)は開始commitからR3評価対象の4 sourceを復元するためのもので、そのsource hashはR2 manifestへ一致する。開始版と前回評価対象の双方を、同じprobe・locked dependency rlib・flags・1000文書/2002語のDB・warm-up・試行数・区間回数・版順反転・allocation観測で比較した。serialized MATCHは両版でDB更新前後に一致し、全8正規化設定とMaybeの固定期待値も照合した。生recordは各比較のrawが正本で、表はその全7試行から生成した。現行source/lock/probe/dependency/binaryと対象テストbinaryは各manifestで特定する。

R3の最初の測定はfmt修正前のsourceで実行したため、[そのmanifest・raw・表・テスト結果](repair-3/before-fmt/manifest.json)も残した。fmt指摘を反映後、採用sourceとhashが一致する版で両比較を実行した。fmtによる行の変更を性能改善には数えない。背景負荷の隔離は引き続き未確認で、全試行・増加条件も保持する。分類の再走査を除いたことは制御経路で確認できるが、時間観測だけから一般的な高速化や利用側検索品質の改善を断定しない。

[前回版比較](repair-3/prior/table.md)の混合queryは中央値441768.25→381912.88 ns/call、範囲335067.25–611735.54→325120.79–680832.96、Rust allocationは157→157だった。[開始版比較](repair-3/start/table.md)では332208.08→372319.08 ns/call、allocationは170→157で、時間は増加した。前回版比較でもlong_unicodeは75300.17→83605.17 ns/call、repeatedは269940.21→277339.00、repeated_missは230408.12→278829.42で増加し、全条件の分布は表へ残す。分布の重なりと背景負荷の未確認から、境界利用の一般的な時間改善とは判断しない。照会traceは混合queryが両比較の両版でprefix lookup 1回・schema検査0回、long/long_unicodeはprefix lookup 0回で現行版のschema検査は1回、uniqueは2回、repeated/repeated_missは開始版4回→現行版1回を維持した。DB更新後の展開結果の一致も維持した。

既存の長語・混合語・operator・重複短語・literal vocab/実MATCH・DB更新・schema検査・正規化検証を再利用する。境界の扱いで引用や展開が変わる不具合を既存の固定期待値と実SQLiteで検出し、長いprefixの再走査費用は同条件probeで観測する。時間閾値の回帰テストは背景負荷による不安定さと保守費用に見合う保証を加えないため追加しない。既存テストの追加・削除・移動はなく、失う検出条件もない。同一接続と二接続のschema検証は異なるcache境界での誤成功を防ぐので保持する。通常suite全体の時間・保守費用は未測定で、対象harnessの表示時間を全体checkの短縮量へ読み替えない。 [前回版比較の対象テスト](repair-3/prior/tests.txt)と[開始版比較の対象テスト](repair-3/start/tests.txt)はいずれも54件成功（表示0.02秒）。[対象Clippy](repair-3/clippy.txt)、変更した製品Rust sourceのfmt、差分検査も成功した。Clippyのdead_code allowはstandaloneでのみ未使用となる公開APIのためで、repoのlint・checkには適用しない。全workspace・FFI/MLX・実モデル・最新remote main・CIは今回未確認である。

要求の正本は提供されたIssue #312と合意記録で、#307報告は開始版から変更されていない過去の観測である。親Issue #296の更新時刻と固有条件は今回独立確認できておらず、この節では提供Issueと現行CONTRIBUTINGで確認できる範囲だけを適用した。未確認の条件を新たな完了条件にしない。次の独立評価ではR3-1の境界利用、既存4修正の維持、現行manifestとsourceの一致、rawと本文の数値・条件・限界を確認する。変更後の全体checkと独立評価はホストが行う。過去の成功を現行の成功へ流用せず、`bash scripts/check.sh` / capture=nullを維持する。

## 四回目評価後の修正と再測定

この節の「現在」「current source」や成功結果は、コメント整理前の公開head `b24ea787dcd59d33e04aa9b8255f3310f2070aa1` の歴史的byte版を指す。整理後のsource hashとは一致しない。原資料は保持し、現在版への限定適用は[コメント整理後の版照合](comment-source-check.md)を参照する。

R4-1は、単語の走査を1回にした空白処理でも、`split_whitespace`で元入力から切り出した単語を`starts_with`で再比較していたことを示した。R1〜R4の評価・assessmentsと空のhandoff、attempt 3のfindingsと変更版、ホストcheck-4を現在のsourceへ照合した。attempt 3のsourceBeforeは`e1c36305a7764164e163797ac38ad57044bdf1e2f99474d04b1f1d2f1c73167b`、sourceAfterは`d7cd4fa88c31f004d371e988812add519e4b053a0a83ae5c648dcbd421d490bd`、stdout SHA-256は`d8e30e79790608c2b70ea137e2d86d7d6e9a3d3d5dc70c2fb1e1164375789770`。check-4はそのsourceAfterで終了0、stdout/stderr SHA-256はそれぞれ`b3a989a689efb8bf6899fd3785d4aa8e18a5be3f4cb83264486c93624a649cd3` / `c5b6a2f47eb29bde9ee04d1e1bae1404af8d3dfd9255075ed832b7baa7ac39ef`で評価対象記録と一致した。修正開始時の4 source・lockはR3 manifestと一致した。過去5件の修正は維持し、check成功を冗長処理の不存在へ読み替えない。

`output=None`の間はoffset以前がcanonicalで、単語は同じ不変のtextを借用する。区切りを確認した後のtailとwordの開始アドレスを比較すれば、必要な位置の一致を判断できる。内容の再比較だけをこの位置比較へ置き換え、追加の状態・helper引数・工程間cacheは設けていない。先頭・末尾・Unicode空白と空入力での所有化、変更不要入力のBorrowed、NFKC→ASCII→空白の順序、公開String APIは維持する。search本体・テストは変更せず、schema検査・query内memo・引用・literal wildcard・DB更新後の観測も維持する。

[比較probe](repair-4/probe.rs)はR3の6 queryを残し、公開String APIの正規化に`日本語`×1024、`canonical`×1024、`react hooks`、`ＡＢＣ　Foo  Bar`、空白・tab・改行・全角空白を加えた。長いcanonical語の正規化と、内部Cowを経るqueryを同じ版・入力・依存・flagsで比較する。公開APIの時間には所有String化も含まれる。全8設定とMaybeの固定期待値、DB更新前後のserialized MATCHを照合した。20回warm-up、正規化10000回/query1000回の区間平均×7試行、版順反転、別呼出しのRust allocation、時間区間外のDB生成・trace・printは従来と同じ。背景負荷の隔離は未確認で、外れ値や時間増加条件を除かない。実モデル・GPU・利用側の検索品質の測定ではない。

[前回版manifest](repair-4/prior/manifest.json)のbaseline sourceはR3のcurrent hashに一致し、[差分](repair-4/prior-source.patch)で開始commitから復元できる。[開始版manifest](repair-4/start/manifest.json)と両比較のcurrent source・lock・probe・dependencyは一致する。R3の測定と違い、今回は正規化の時間も測っているため、R3のrawと数値だけを直接比較しない。前回版比較の[raw](repair-4/prior/raw.jsonl)・[表](repair-4/prior/table.md)、開始版比較の[raw](repair-4/start/raw.jsonl)・[表](repair-4/start/table.md)を原資料とする。

前回版との比較では、長いUnicode語の公開正規化の中央値は111802.24→77334.59 ns/call、queryは103949.71→74668.29 ns/callだった。一方、長いASCII語の正規化は16677.15→17019.35、重複短語queryは348795.92→440227.08、重複missは297129.25→443249.92で増加した。全条件のallocationとprefix lookup/schema検査回数は前回版から変わらない。試行範囲は重なり、背景負荷も未確認なので、位置比較単独の時間効果や一般的な高速化とは断定しない。

開始版との比較では長いUnicode語の正規化は41635.40→67559.06 ns/callで増加し、長いASCII語は107386.10→14578.62だった。両方の公開正規化のallocationは14→1。長いUnicode語queryは58324.00→75447.54、長語prefix＋短語は336099.25→375217.46で増加し、allocationは29→14、170→157だった。重複短語と重複missは1192181.29→288769.50、939156.79→239045.04 ns/callで、allocationは576→162、48→30、prefix lookupは4→1だった。長語のみのprefix lookupは0→0、行を取得しないschema検査は0→1。短語があるqueryには検査を重ねず、uniqueの照会2回、長語prefix＋短語の照会1回も保持する。開始版比較は今回だけでなく過去5修正を含む観測である。

[前回版比較の対象検証](repair-4/prior/tests.txt)と[開始版比較の対象検証](repair-4/start/tests.txt)はともに54件成功（表示0.02秒）。R3の保存結果も同じ54件・0.02秒であるが、表示精度と異なる背景負荷のため追加assertの費用は分離できず、実行時間の改善とは扱わない。[対象Clippy](repair-4/clippy.txt)、変更Rust sourceのfmtと差分検査も成功した。standaloneで未使用になる公開APIのdead_codeだけをClippyでallowし、repoのlint・check条件は変更しない。全workspace・FFI/MLX・実モデル・CIはこのsandboxで未実行。

既存の空白onlyテストに固定literal期待値とBorrowed確認を統合した。空入力、canonicalなASCII/Unicode、同じ内容の語が異なる位置にある入力、Unicode区切り、先頭・末尾空白で、誤った位置判断による空白の残存・単語欠落や不要な所有化を防ぐ。元の`React App`の期待値は保持し、新しいテスト関数・fixture・待機・時間閾値は加えない。既存の空白出力検証へ内部Cowの所有/借用保証を加える小さな固定入力なので、追加の実行費用・不安定さ・保守範囲は限定されるが、通常suite全体の追加費用は未測定で、費用低減は主張しない。削除・移動したテストと失う検出条件はない。同一接続/二接続schema、literal vocab/実MATCH、DB更新、全8設定、Maybe検証は引き続き現実的な誤成功・誤hit・失hitを防ぐので維持する。再比較の費用はprobeと処理経路で確認し、不安定な時間閾値テストは追加しない。

要求の正本は提供Issue #312と合意記録で、CONTRIBUTINGとaccepted ADR-0001の#297訂正、ADR-0012を従来と同じ適用条件で使った。#307報告は開始版blobから変わらない過去の観測で、新しい品質許容差や方式採用の許可ではない。最新remote mainと親Issue #296固有条件は独立確認できていないという限界を保持し、未確認条件を今回の完了条件へ加えない。次の独立評価では位置比較の不変条件、過去5修正の維持、固定期待値とBorrowedの保証、現行manifest/source・raw/表/本文の一致を確認する。変更後の`bash scripts/check.sh`と独立評価はホストが実行する。capture=null・CI条件は変更せず、過去checkを現行版の成功としない。

## 再実行

先に通常のホスト検証でlocked依存をビルドし、release dependencyを必要なら`cargo build --locked --release --features test-support,test-mlx,smoke`で用意する。`run.py`は新規のcheckout外ディレクトリだけを使い、Cargo構成・依存graph・制御スクリプトを編集しない。使う`deps`は、そのCargo.lockと対応するbuildで生成したものを指定する。

```sh
python3 docs/benchmarks/issue-312/run.py --deps "$CARGO_TARGET_DIR/release/deps" --out /tmp/rurico-312-new-evidence \
  --probe docs/benchmarks/issue-312/repair-4/probe.rs
```

`CARGO_TARGET_DIR`を設定していない場合は`--deps target/release/deps`を指定する。同じcrate名のrlibが複数ある場合は曖昧な選択を拒否する。実行中に測定対象sourceやCargo.lockが変われば失敗し、新しい版の成功記録として保存しない。生record・manifest・対象テスト結果を新規出力先へ残す。通常check・独立評価・CIと、性能測定の結果は分けて扱う。

このコマンドは実行するcheckoutの現在のbyte版を測る。保存済みR4の測定対象を再現する場合は、公開head `b24ea787dcd59d33e04aa9b8255f3310f2070aa1` の未変更の別checkoutで同じ手順を使い、4 source・lock・probeをR4両manifestへ照合する。整理後のcheckoutを使った新しい結果をR4原本へ上書きしない。コメントだけの今回の変更に再測定は要求せず、[実行部分の同一性照合](comment-source-check.md)と新しい通常check・独立評価へ引き継ぐ。

修正前の評価対象との比較を再現する場合は、開始commitの4 sourceとCargo.lockをcheckout外へ取得し、`repair-1/prior-source.patch`をそのコピーへ適用してから、`--baseline-dir /tmp/rurico-312-prior-source --baseline-id f3ef9e8435ad20db1ed1730b1e418c01befc6a8801fa7b47164458d21ffbf6d2`を追加する。比較用sourceのhashを初回manifestの`current/`へ照合する。追加引数なしの場合は引き続き開始commitと比較する。保存済み過去rawとmanifestを上書きしない。

R2直前の評価対象を比較元にする場合は、同じ開始版コピーへ`repair-2/prior-source.patch`を適用し、`--baseline-dir`と`--baseline-id 9fed7e3b57063f54c88ccb08905114b1ca2d5c5bc2cec4089f1f1cd73328e519`を指定する。source hashはR1 manifestの`current/`へ照合する。初回/R1 probeは過去条件の再現用として保持する。

R3直前の評価対象を比較元にする場合は、同じ開始版コピーへ`repair-3/prior-source.patch`を適用し、`--baseline-dir`と`--baseline-id 23c720e2591c6367b914758e2014e67d7d0bf2f3b8c523b728057746afeb6865`を指定する。source hashはR2 manifestの`current/`へ照合する。R2のprobe・測定結果も過去の条件として保持する。

R4直前の評価対象を比較元にする場合は、同じ開始版コピーへ`repair-4/prior-source.patch`を適用し、`--baseline-dir`と`--baseline-id a77b057b0f2e33bfb648b73b012fcf67d8da677bfea61e21e4d4b514ccc7d930`を指定する。source hashはR3 manifestの`current/`へ照合する。過去のprobe・raw・manifestは上書きしない。

## コメント整理後の扱いと引き継ぎ

今回の固定入力で採用されたコメント整理に従い、4 Rust sourceの古いテスト番号、関数名やassertionの反復、末尾の重複wire-format説明を整理した。公開契約、対称正規化とlegacy default、SQLiteのschema/implicit AND、fixtureが区別する失敗条件、同じ元入力を借用する位置比較の不変条件は保持した。[版照合](comment-source-check.md)で、R4測定byte版と整理後の非コメント行が一致することを確認した。コメントの削減・行数・ファイル配置を性能や保守費用の改善量へ数えず、時間比較は再実行していない。測定結果の原本・数値・版は書き換えない。

親Issue [#296](https://github.com/thkt/rurico/issues/296)は過去の独立評価では取得できなかったが、ホストの追加read-only取得は成功している。提供ファイルの更新時刻は `2026-10-07T17:08:32Z`（初回記録と同じ）、本文のUTF-8 SHA-256は `5918980b98047e009322bf25711d01b7aab98802c826719ef3cc0bcab7d00ee6`。その本文にあるlocked依存、Apple Silicon/Xcode/Metal、既存check、証拠の公開先、方式・consumer・品質許容差の未採用条件を合意済みIssueと現行CONTRIBUTINGへ照合した。取得不能だった過去の実行事実は各節に残し、現在のホストで未取得とは扱わない。この確認は評価担当自身の取得成功や実モデル/検索品質の実測ではない。#307報告とaccepted ADRの対象版・適用範囲は「要求と参照版」のままで、方式採用や品質改善の許可には読み替えない。

コメントはコードを言い直すだけのものが再び増え得るが、今回の整理基準と必要な理由は残した。新しいルール・専用lint・テストの追加では、今回の実行契約に検出保証を加えられない。既存のliteral vocab/実MATCHはwildcardの誤hitとescape消失の失hit、呼出し間DB更新は古い結果、同一接続/二接続のschema検証は異なるschema cache境界での誤成功を防ぐため維持する。全8設定・Maybe・空白位置とBorrowedの検証も、正規化順序・prefix/suffix欠落・不要な所有化を防ぐ。assertion・数値条件・テスト構成は変更せず、追加・削除・移動・統合したテストも失う検出条件もない。二接続のfile I/Oには同一接続で守れない保証があり、時間閾値テストは背景負荷による不安定さと保守費用に見合う追加保証がないため加えない。通常suiteの費用は未測定で、過去の対象harnessの0.02秒を現在版の所要や改善量へ流用しない。

設定済み `bash scripts/check.sh` は通常の対象テスト・doctest・Clippy・fmtを含み、capture=nullはUI媒体不要というIssueに一致する。コメント整理後の全体check、変更文書を含む新しい独立評価、同じ公開headのCI（test・coverage・security・zizmor）はホストの後続工程で確認する。古いacceptedやcheck成功は流用しない。新しい評価のassessmentsとhandoffには上記限界と[既存公開添付リンク](comment-source-check.md)を明示し、PR本文へ引き継ぐ。ここではcommit・push・公開を行わず、draftを維持する。
