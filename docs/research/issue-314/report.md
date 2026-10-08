# Issue #314 phrase・短語・query planの設計比較

既存の文字列APIを保ち、明示的な入力型からliteral・phrase・展開groupを組み立てる追加APIを推奨する。
入力の引用符やORを自動で検索言語へ昇格させない。今回の調査は製品採用ではない。
amici固定版のparser本文と既存round-trip、合成入力26観測はホスト証拠で確認した。
これは旧wireの観測した範囲に限る。内部引用符のnative実測でもliteral欠落を確認した。新APIや全consumerの互換性は保証しない。

合成phrase例は、現行経路でunicode61が2件、trigramが0件、明示phraseなら両方1件だった。
25件上限は両tokenizerで61文書中50件を返し、同頻度の語と稀な語に11件の取りこぼしが残った。
順序と打切りを明示する案はこれを可視化するが、品質改善にはならない。

## 根拠の版と適用範囲

要求と合意の正本は[Issue #314](https://github.com/thkt/rurico/issues/314)（提供本文の更新時刻2026-10-07T17:05:41Z）。
開始commitは`24725a72be44300afc24186b82b14bcb3f5f9d3d`で、ローカル`origin/main`も同じだった。
ネットワーク経由の最新main取得はできていない。ホストが準備した開始版を対象にし、別版へ黙って切り替えない。
Cargo.lockのGit blobは`ead843554d0d5f2019f6f15b29a12c02100fdbab`、
SHA-256は`743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。変更していない。

| 出典と版 | 状態・適用する判断 |
| --- | --- |
| [search.rs](../../../src/storage/search.rs)、[query_normalize.rs](../../../src/storage/query_normalize.rs)、開始版 | 現行実装の事実。#297の引用修正後のliteral、短語prefix展開、正規化対称性を確認 |
| [ADR-0001](../../decisions/0001-typed-fts-query-contract.md)、開始版 | acceptedの公開境界。2026-09-20の訂正を適用し、古い「ORは演算子」の注記を最終MATCHの説明へ流用しない |
| [#315報告](../issue-315/report.md)、blob `1e32f409ade44c73fa6d47719310c5861c1dc817` | 未採用の構成識別案。FtsIndexSpecとFtsSearchPolicyを分離する設計根拠。APIやstripの採用許可ではない |
| [#307報告](../issue-307/report.md)、blob `7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5` | 過去観測。固定amici/d063の60文書・168クエリのreference compositionに限定。今回のphrase/短語やconsumer全体の品質保証ではない |
| [ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md)、開始版 | accepted。実モデル付き検索品質の評価はamiciが所有する。今回の小さなSQLite意味比較は評価harnessの再移植ではない |
| [SQLite FTS5](https://www.sqlite.org/fts5.html#the_trigram_tokenizer)、2026-10-08取得の公式説明 | 引用内の`*`はprefix構文ではない、trigramの3文字未満はMATCHしない、phraseはtokenizerの位置列で定義される。実測の説明に使用 |

指定の二報告は開始commitのblobと現在ファイルが一致した。Issue見直し版
`c8f250d60a5afb9944b9008d22ad6f4dda2d7103`から開始版まで、上記二報告・search.rs・query_normalize.rsに差分はない。
初回調査の`32513da690653a8baf30d2af4ae77d2129838f55`は、#297以前の二重引用の根拠として保持する。
その旧版の文字列出力を現行出力とはしない。#312の意味を保つ最適化や#315の構成識別を今回の検索意味の採用根拠にしない。

READMEの[FTS説明](../../../README.md#fts-クエリパイプライン)、CONTRIBUTINGの
[テスト](../../../CONTRIBUTING.md#テスト)・#307/#315の手順、decisionsの入口と関連ADRを参照した。
リポジトリにはdocs/wikiの開発方針はないため、追加の製品方針を作らない。
既存の文書を現行説明として保ち、今回の未採用比較はresearchに置く。

## 実MATCHと品質

測定日は2026-10-08。Python 3.14.8、SQLite 3.53.1、macOS 27.0.1 arm64。
正規化は索引とqueryの両側で全OFF、FTS5のunicode61/trigramはdefault options。
[cases.json](cases.json)が入力・文書順・評価意図、[results.json](results.json)がwire・hit ID・品質・source hashの原本である。
評価意図は合成例のsubstringまたはphrase一致であり、新しい製品受入基準ではない。
precisionはTP/(TP+FP)、recallはTP/(TP+FN)。分母0はnullにし、空結果をprecision=1としない。
順位や実モデルのRecall@kは測っていない。

以下の「現行」は現在のSQLとfixtureのlegacy tokenを参照例で実行した結果。
参照例のvocab不在modeはlookupを省いてliteral fallbackを実MATCHする。SQLiteのmissing-table例外自体は
追加Rust群が実際に存在しない表名で確認し、既存のSQLite障害テストも維持する。
製品の公開入口との照合は追加したRustテストが標準checkで実行する。
評価対象`360237b39dcf937e30d75c5b522cdd0504372e4c8b316a385d6744f1e465875d`のホストcheckログで、
`research_query_plan_legacy_matches_shared_observations`の成功（0.032秒）を確認した。
この事実を修正後の全体check成功へ引き継がない。

| 入力・評価意図 | unicode61現行hit | trigram現行hit | vocab不在時／意味の差 |
| --- | --- | --- | --- |
| `"foo bar"`、連続したphraseだけ関連、2文書 | [1,2]、precision 1/2・recall 1 | []、recall 0 | 明示phrase `"foo bar"` は両方[1]、precision/recall 1。現行wireは `"""foo" AND "bar"""` |
| `foo OR bar`、literal OR、5文書 | [1] | [] | どちらも `"foo" AND "OR" AND "bar"`。ORをBooleanへ変えると別の検索になる |
| `abc*def%_`、記号込みliteral、3文書 | [1,2] | [1] | wildcard構文を発動していない。unicode61は記号をtoken境界として扱うため文字列の完全一致とは異なる |
| `日`、含む4文書 | [1,2,4]、recall 3/4 | [1,2]、recall 1/2 | 不在時は[4]／[]。prefix展開では`休日`を拾えず、trigramでは単独`日`も索引にtokenがない |
| `日本`、関連3/4文書 | [1,2,3]、recall 1 | [1,2]、recall 2/3 | 不在時は[3]／[]。3文字未満だけの文書はtrigram展開でも回復できない |
| `珍`、稀な`珍語彙`1/2文書 | [1] | [1] | 不在時は両方[]。稀でも候補数が上限以内なら回復する |
| `無`、候補なし | [] | [] | literal fallback。vocab不在と候補なしの結果は同じでも診断原因を分ける |
| `日%`／`日_`／`日\`、LIKE記号を含む短語 | [2,3]／[2,3]／[2] | [2]／[3]／[2] | `%`・`_`・backslashをLIKEでescapeする。無関係な`日本語`を候補へ混ぜない。不在trigramは全て[] |
| `日`、30同頻度語×2文書＋低頻度1語、61関連文書 | 50件、recall 50/61 | 50件、recall 50/61 | 10件の同頻度文書と稀な1件を失う。不在は両方0/61 |

現行の`ORDER BY cnt DESC LIMIT 25`は同頻度の第二キーを指定しない。
今回のSQLiteではvocab走査順が字順となり、第二キー付き案と同じ25語だった。
これは将来版や別vocabに対する決定的順序の契約ではない。新案は
`ORDER BY cnt DESC, term COLLATE BINARY ASC LIMIT 26`で、先頭25語と26語目の有無を返す。
別の挿入順を持つ同頻度vocabを検証に使い、25候補で打切りfalse、26候補でtrueを確認した。
現行順序の変更自体も上限境界で結果を変え得るため、#312の意味不変修正には含めない。

## 小さな文字列修正と型付きplanの比較

| 案 | phrase・短語・診断 | 互換性・保守費用 | 判断 |
| --- | --- | --- | --- |
| 現行API＋第二sortキーだけ | phrase境界を保持しない。打切り/原因は見えない | 公開型は保てるが、同点上限のhit集合を変え得る | 問題全体を解決しない |
| `prepare_phrase`等の別文字列入口＋diagnostics返却 | 単一phraseを引用できる。複合group/phraseを追加するたびに文字列処理が増える | 小さな単一phrase用途には十分。旧入力の引用符解釈を変える案は互換性を破る | 比較候補。複合planをconsumerが再parseする課題が残る |
| literal/phrase/展開groupを保つ追加API＋legacy adapter | 入力意図とfallback/打切りを保持し、SQLite化を最後に一度だけ行う | node/constructor/policyを保守する費用は増える。consumerとversionの調整が必要 | 設計として推奨。製品採用・互換性確認は未完了 |

具体的な追加面は以下の案とする。現在のcrateからimportできるAPIではない。
既存`prepare_match_query`と`MatchFtsQuery`はそのまま残す。

```rust,ignore
pub enum SearchRequest { LegacyText(String), Structured(Vec<InputNode>) }
pub enum InputNode { Literal(String), Phrase(String) }
pub struct FtsSearchPolicy { /* expansion limit, tie order, fallback, semantics version */ }
pub struct PreparedPlan { /* private nodes + diagnostics; validated constructors */ }
pub enum FallbackReason { MissingVocab, NoCandidates }
pub struct ExpansionInfo { /* source, selected terms, limit, truncated, fallback */ }
// prepare_plan(conn, request, normalization, policy) -> Result<PreparedPlan, PlanError>
// PreparedPlan::to_match_query() -> MatchFtsQuery
// PreparedPlan::to_legacy_match() -> Result<MatchFtsQuery, UnsupportedLegacyPlan>
```

`LegacyText`は現行normalize→sanitizeのliteral列をそのまま利用し、quoteは最終serializerだけで行う。
構造化入力でも既存正規化と索引側の同じ設定を要求する。`Literal`はsanitizerの語の単位を表す。
SQLite tokenizerが記号をさらに分割する場合まで、文字列の完全一致を保証しない。
`Phrase`は明示された一境界を保持し、構成語を短語groupへ置き換えない。
trigramの1〜2文字phraseにliteral fallbackだけを指定してもhitは得られないため、その制限を返す案とする。
入力の引用符は`LegacyText`では引き続きdata。`Structured`では型でphraseを指定する。

展開はsourceと選択語を保持し、capは「候補語数」であって返却文書数の上限ではない。
新APIでは25という候補を比較の基準にし、無制限展開や新しいhard limitを採用しない。
`truncated`、`MissingVocab`、`NoCandidates`を区別し、非missingのSQLite障害を成功fallbackへ変えない。
Booleanを将来追加するなら別の明示node/入口と文法versionが必要。現行`foo OR bar`をOR nodeへ変換しない。
空planや空groupはconstructorで拒否し、ユーザー文字列を未引用SQLとして注入しない。

[plan.py](plan.py)はこの最小の構造とserializer・legacy adapterを実行する参照例であり、
製品constructor、正規化、全文法parser、vocabの安全な識別子APIは実装していない。
legacy adapterは旧wire形状を保つがdiagnosticsを文字列へ載せない。Phraseは現行consumerへ黙って落とさず拒否する。
旧wire形状の維持はliteral内容のround-trip保証ではない。固定consumerの内部引用符欠落も適用条件に含める。
同じSQL文字列でもPhraseとLiteralの型の意図は回復できないため、構造の転送をSQLの逆parseだけに頼らない。

## #315との分担とconsumerへの影響

#315の`FtsIndexSpec`は正規化、tokenizer/options、consumer前処理の意味を識別する未採用案。
今回のphrase、vocab参照、候補上限、tie order、fallbackは別の`FtsSearchPolicy`とする。
planだけの変更で索引recordを変えない。NFKCをquery側だけOFFにする変更や別ngram索引は
索引構成の変更であり、検索時policy変更へ偽装しない。vocabと索引の対応はconsumerが確認する。

amiciのwireをruricoが生成することは[現行source](../../../src/storage/search.rs)の契約として確認した。
固定語`"..."`、展開group `("..." OR "...")`、top-level ` AND `を維持する。
amici固定版`547f9ee2ed734a2eab316fdbd62f194849875ee4`のsource hashは
[#307 manifest](../issue-307/results/search/amici-source.json)に保存済みである。
当初の固定版取得はscout exit 75などの通信失敗で未完了だった。
ホストで同じarchiveを取得し、archiveとparser・既存testsのhashが#307 manifestと一致することを確認した。
acceptedのADR-0008も本文とsourceを照合した。過去の品質値を今回の成功には流用していない。
既存round-tripを、今回のrurico sourceへpath patchした隔離コピーで1件実行し、終了0を確認した。

[consumerの実記録](consumer-cases.json)には、今回のproducer→固定parser→cleaner→実MATCHの26観測を残した。
追加の再現は一時コピーだけに[patch](consumer-cases.patch)を適用し、製品consumerを変更していない。
両tokenizerで、入力`"foo bar"`はcleaner後に2件、未採用の明示phrase例は1件を返した。
`foo OR bar`のliteral `OR`はcleanerの3文字未満除外で失われ、固定語`foo`・`bar`だけとなり3件を返した。
これは前節のproducer wireを直接MATCHした1件／0件とは別の経路である。
`日 月 login`の25×25展開は、既存consumerの100組上限によりgroupを落とし、`login`だけで61件を返した。
この既存上限を新planへ採用したわけではない。打切り・fallback・term種別の診断を型で渡す提案の根拠となる。
`%`・`_`・backslash、稀語、短語、同頻度も同じ共有fixtureから記録した。
当時の全fixtureの観測であり、内部引用符入力は含まれていなかった。
任意のquery・consumer全体への互換性保証ではない。

consumerのnative SQLiteは3.53.2。Python参照例はPython 3.14.8／SQLite 3.53.1で再実行し、
68観測のwire・hit・品質と3構成の容量・品質が元の記録と一致した。
[ホスト再実行](results-host.json)も原本を上書きせず保存した。書込み時間は再び一回の観測で、速度順位を保証しない。
consumerの解決済み依存は[package記録](consumer-resolved-packages.json)と[検証要約](host-verification.json)に残した。
path patchはrurico sourceを指定するが、consumerは自分のlockを解決する。
rurico自身の固定Cargo.lockでの検証とは分け、元のCargo.lockとamici manifestは変更していない。

影響箇所はamiciの`src/storage/fts.rs`・同tests、wire-contract ADR-0008、依存rev更新とそれを呼ぶconsumerの検索入口。
旧wireのseparatorやquoteを変更する場合はamici parserの協調変更が必要。
さらに固定版の`parse_fts_segments`は引用内で最初の`"`に達すると走査を止め、`""` escapeを解釈しない。
現行producerの入力`say"hi`は`"say""hi"`となるが、sourceからはfixedが`["\"say\""]`、
groupsが空、cleanerが`"say"`になると読める。残る`"hi"`は3文字未満として落ちる。
R1-2の独立評価ではparser再構成とSQLite 3.53.1の3文書MATCHで、trigramの直接hit `[1]`に対し
再構成後hit `[1,2,3]`だった。この独立評価の観測とは別に、固定consumerのnative SQLite 3.53.2でunicode61とtrigramの双方を確認した。[4観測](consumer-quotes-host.json)では正常な`say`は両経路3件、`say"hi`は直接1件・consumer3件。合成評価のconsumer precisionは1/3、recallは1だった。
さらにgroup走査は引用状態を考慮せず最初の`)`で終了する。[追加の2観測](consumer-group-host.json)は、trigramで正常な`日本語`と内部括弧を持つ`日)本`を対照した。入力`日`から前者は`("日本語")`、後者は`("日)本")`に展開する。直接MATCHはいずれも文書1を返したが、後者のparserはfixed/groupとも空、cleanerはNoneとなり、consumerの検索入口は空結果を返す。合成評価のrecallは1から0へ落ちた。固定語のescapeだけでなく、展開groupの引用内終端認識もconsumer対応の対象となる。製品parserは今回変更していない。

この欠陥はwire変更がなくてもliteralを失って検索条件を広げる。legacy adapterを同じparserへ渡す場合も
内部引用符入力の互換性を主張できない。入力を禁止したりproducerのescapeを弱めたりする案は採用しない。
影響特定に基づく改修案には、amici parserのescape-awareな引用走査とgroup内引用状態の保持、producer→parser→cleaner→MATCHの対照検証が必要。
parser改修とその契約・テスト更新の採否は後続合意へ残し、このIssueで他repoを製品改修しない。
typed planへ移る場合は、新型を受けるtrigram adapter、diagnostics転送、policyの記録、旧入口との対照検証が必要になる。
amiciを経由しないconsumerの範囲はこの固定版の確認から推測しない。実装対象repoの追加は別の合意とする。

移行順の提案は、既存round-tripと今回の内部引用符実測で確認した旧契約の欠落を対象にする。
既知のliteral欠陥へのconsumer対応を別途合意し、次にruricoの追加API案を合意する。
amiciのtyped adapterを旧入口と並行して比較し、phrase/短語のhitと既存品質を確認してからconsumerごとに切り替える。
索引構成が同じならplan更新だけで再索引を要求しない。ngramや前処理を変えるなら#315の構成比較と
旧索引保持・別索引生成・rollbackを含む移行の再合意が必要。
今回は他repoの製品変更、consumer移行、strip、再索引を実行しない。

## 短語不足を確認した後の別ngram比較

上の短語欠落を受け、同じ6文書（`日本語`、`日本海`、`休日`、`日`、`本日`、`日と本`）を各256回登録した。
trigramはSQLite builtin、bigram/unigramは重なりgramを空白で区切りunicode61へ登録する参照例。
専用tokenizerの実装・採用ではない。queryはAND結合とし、原文substringを関連文書の定義にした。

| 構成 | 登録後page数（共通初期6、4096 byte/page） | token出現数 | 登録／全件content更新の秒数 | 短語品質 |
| --- | ---: | ---: | --- | --- |
| trigram | 19 | 768 | 0.001228／0.004281 | `日`、`日本`、`休日`はliteral MATCHでrecall 0 |
| bigram例 | 24 | 2048 | 0.001244／0.003204 | `日本`・`休日`はprecision/recall 1、`日`はrecall 0 |
| unigram例 | 27 | 3584 | 0.001264／0.003157 | `日`はprecision/recall 1。`日本`はrecall 1・precision 1/2（`本日`・`日と本`が混入） |

ページ数には本文保存とSQLite管理領域を含み、index-only byte数ではない。
insert/updateは同じ1536文書・同じtransaction条件で測り、Pythonのgram生成時間は含まない。
updateは同一本文を再設定するSQLiteのdelete/add経路。一回の極短時間の観測でばらつきは測っていない。
キャッシュ・順番・語彙・原文保存方式の影響があり、速度の優劣、実DB容量、保守時間の改善を主張しない。
unigramの位置phraseや原文post-filterは混入を減らす候補だが、今回は効果を測っていない。
新しい索引を採用するには容量・更新条件と検索品質を実consumerで比較し、1文字対応の費用を合意する必要がある。
現段階では索引を維持し、planの制限と打切りを呼出元へ返す設計を推奨する。

## 検証の価値と未確認範囲

既存の引用・operator・日本語展開・正規化・SQLite障害の検証を維持した。
追加Rust一群は共有fixtureからphraseの誤認、上限境界の稀語欠落、短語のLIKE escape脱落を製品入口で照合する。
同頻度の現行順序を固定するassertionは置かず、選択数と稀語除外を確認する。
既存が既に守るoperator literal等は研究測定では比較するが、新しいRust群で重複実行しない。

Pythonの二群は、新serializer/adapterでのphraseの無断downgrade、quote/groupの崩れ、
lookaheadの境界誤判定、tie keyの脱落、fallback原因の混同、非missing障害の隠蔽を防ぐ。
モデル・network・時間閾値を使わず、二群のunittestは0.001秒で成功した。
研究runnerも68件のwire/hit観測と3構成の書込み比較を完了した。68件は独立反復数ではない。
既存テストの削除や意味変更はなく、失う既存検出条件はない。
Rust一群は上記ホストログで0.032秒。長期保守費用は未測定。新しい公開APIの製品検証は追加していない。
内部引用符は既存のproducerテストが守るが、下流parserの欠落は検出しない。
[引用符の追加再現](consumer-quotes.patch)は固定consumerの一時testsだけへ追加する。
同じ3文書で正常な`say`と内部引用符入力を対照し、回復構造、cleaner、直接MATCHとの差を記録する。
これは欠陥の観測であり、広いhitを望ましい製品仕様とする回帰テストではない。
既存26観測を再実行するだけでは埋まらない条件を補うため維持する。引用符の再現filterは1件・テスト0.01秒（build込み6.319秒）。不安定さは未測定で、
モデル・通信・時間閾値には依存しない。恒久テストへの採用はconsumer修正の合意時に検討する。

### ホスト確認と残る検証

ホストではPython二群と68観測・3構成の再実行が終了0。
固定consumerの既存round-trip1件と追加MATCH26観測は終了0だった。
終了0は観測の実行成功であり、そこで失うliteralやgroupまで受け入れたという意味ではない。
最終のconsumer binaryでは、追加再現とSQLite版確認を含む初回のFTS suite12件が終了0（0.03秒）。内部引用符再現追加後の同じconsumer依存解決でFTS suite13件も終了0（0.03秒）だった。
既存commentsの古いテストIDと装飾を整理し、assertionと処理は変更していない。
source・依存・結果の範囲は[検証要約](host-verification.json)を参照する。

標準setupの`cargo fetch --locked`、`bash scripts/check.sh`、変更文書を含む独立評価、同じheadの
CI test/coverage/security/zizmorは、この要約を反映した版で別途確認する。
当初の日本語lintはcache権限・offline依存不足で失敗した。ホストでは固定版lintを実行できた。
長い技術比較による語彙反復と読解負荷の情報指摘を確認し、範囲や未確認事項を省略せず整えた。
媒体は不要。captureはnullのまま、ブラウザーやサーバーは起動していない。

実モデル付き新planの品質、実データ、producer並行性、全文法・全consumer互換性はこの合成比較では保証しない。
#307の過去観測はその対象版・条件の証拠として残し、今回の測定結果や新planの採用へ昇格させない。
新しい許容差・検索言語・consumer移行は採用していない。

### 再評価への引き継ぎ

R1-1は確認前の状態説明を後半の証拠追加時に更新しなかったことが原因だった。
冒頭、製品入口、実行時間の説明とREADMEの見出し参照を更新した。
初回の停止応答・results.json・追加ホスト原本は維持し、現行説明だけを修正した。
R1-1は文書修正済みだが独立再評価待ち。R1-2はsourceで欠陥を確認し、影響と再現定義を補ったが、
固定consumerのnative実測4観測と13件のsuiteが終了0となった。R1-2の不足した証拠を補ったが、独立再評価のacceptedを自己判断で代替しない。
今回の局所checkは新しいpatchの適用確認、追加Rust構文・整形、差分の空白検査で成功した。
文書lintはこのnative結果更新後に再実行する。修正後の全体check・独立評価は未実行。文書の版・数量・条件・参照は上記原資料と手動で照合した。

前runの停止記録にある12記録と追加証拠の20ログはSHA-256が一致した。
保存成果物・生応答には内部引用符の観測がなく、追加26観測にも含まれない。
#307/#315の指定blobは現在ファイルと一致し、適用範囲と未採用状態を維持する。
追加証拠のpassedは受入として扱わず、固定版のsource、path依存解決、lock、実行ログに照合した。
この照合は新しい引用符再現の実行を代替しない。

追加filterは1件・4観測、終了0、buildを含む6.319秒、テスト時間0.01秒だった。[検証要約](consumer-quotes-verification.json)に固定consumer版・同じlock・製品source不変・binary hashを紐付けた。最終suiteは13件・終了0。最初のsuite実行では入力fixtureの環境変数に存在しない出力pathを指定して1件失敗したため、生ログを残して入力を修正し、同じsource・binaryで再実行した。広いhitを望ましい仕様とする製品テストは追加していない。

生ログはcheckout外の新しい原本に保存した。報告・要約を更新した版で変更文書と全体checkを独立評価へ戻す。全体checkは設定を変えずホストが実行する。実データ・実モデル・並行性・全consumerは引き続き未確認。

今回の追加評価応答は、fresh host-returnで評価履歴が空なのに旧runのR1-1/R1-2をupdatesとして返したため、契約検査でinvalid_reviewとなった。応答・停止記録は保存し、acceptedとして扱わない。内容上の残る文書不整合を修正し、冗長な同一Planのserializeを一度の局所変数へまとめた。wireの正確さと実MATCHの検査は残し、Phrase拒否を弱めていない。group終端の欠落は固定版の新しいnative原本で補った。次のfresh独立評価では、空の評価履歴に対して旧指摘IDの更新を返さず、現在の対象に対する初回評価として完全なitemsを返す必要がある。

最終のconsumer FTS suiteは14件成功・終了0。引用符4観測、group終端2観測、既存fixture26観測を同じbinaryで実行した。[版・実行要約](consumer-group-verification.json)に同じconsumer lock、製品source不変、binary hashを記録した。追加group filterは1件、build込み6.919秒、テスト部分0.01秒。調査用Python二群も終了0で、実行時間の改善量やflake率は測っていない。
