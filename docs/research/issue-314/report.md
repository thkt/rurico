# Issue #314 phrase・短語・query planの設計比較

既存の文字列APIを保ち、明示的な入力型からliteral・phrase・展開groupを組み立てる追加APIを推奨する。
入力の引用符やORを自動で検索言語へ昇格させない。今回の調査は製品採用ではない。
公開head `6b5a3855f35120c39423a1a49c656f5b58c98ee3`までの固定amici観測は歴史証拠として残す。
最新mainとの統合版では製品search・normalizerのCPU限定検証と、固定consumerのnative FTS suiteを再実行した。
内部引用符・group終端の欠陥を含む旧観測から、新APIや全consumerの互換性は保証しない。

合成phrase例は、現行経路でunicode61が2件、trigramが0件、明示phraseなら両方1件だった。
25件上限は両tokenizerで61文書中50件を返し、同頻度の語と稀な語に11件の取りこぼしが残った。
順序と打切りを明示する案はこれを可視化するが、品質改善にはならない。

## 根拠の版と適用範囲

要求と合意の正本は[Issue #314](https://github.com/thkt/rurico/issues/314)（提供本文の更新時刻2026-10-07T17:05:41Z）。
初回調査の開始commitは`24725a72be44300afc24186b82b14bcb3f5f9d3d`。当時の追加main取得は通信制約で未完了だった。
競合修正は公開head `6b5a3855f35120c39423a1a49c656f5b58c98ee3`から始め、ホスト取得済みmain
`56f412c8401d6a3b501d7effb0aff18df1f344e5`（#377 merge）と三者統合した。共通祖先は初回調査の開始commit。
Cargo.lockのGit blobは`ead843554d0d5f2019f6f15b29a12c02100fdbab`、
SHA-256は`743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。最新main・統合版も同じlockだった。
upstreamの製品改善、検証、文書、依存定義を取り込み、#314から依存版を変更していない。

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

以下の「現行」は初回調査のSQLとfixtureのlegacy tokenによる観測。最新main統合後も参照例のwire・hit・品質は一致した。
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

既存の引用・operator literal・日本語展開・正規化・SQLite障害の検証を再利用した。
共有fixtureとのRust照合はphraseの誤認、25候補境界の稀語欠落、vocab不在時の取りこぼしを製品入口で確認する。
同頻度の順序は現行契約にないため固定せず、25候補・50 hit・稀語除外だけを確認する。

最新mainの `expand_special_chars_escaped` はtrigramの実vocabとwire・hitを使い、
`%`・`_`のwildcard化とbackslashの消失を独立した期待値で検出する。
同じ失敗条件を守る共有fixture側の3入力を通常Rust照合から外した。
失う検出条件は、この3入力と研究原本のwire/hitをRustで直接結び付ける照合である。
製品のescape検証と、研究runnerでの同じ入力の実MATCHは残る。unicode61側の記号分割も研究測定と既存literalテストで確認する。
検索意味・期待値を変えてcheckを通す整理ではない。

Python二群はserializer/adapterのphraseの無断downgrade、quote/groupの崩れ、
lookahead境界の誤判定、tie keyの脱落、fallback原因の混同、非missing障害の隠蔽を防ぐ。
既存の製品検証では未採用planのこれらの診断を検査しないため維持する。
固定consumerの引用符・group終端の再現patchも、既存round-tripでは検出しない条件を補うため残す。
そのassertionは固定版の欠陥観測を確認するもので、望ましい製品仕様への採用ではない。

### 最新mainとの統合版の確認

[CPU限定検証](integration-verification.json)は製品search・query_normalize本体と既存testsをbyte単位でコピーし、
同じCargo.lockに対応する既存rlibのhashを#312のmanifestへ照合してコンパイルした。
整理前後を同じrustfmt、Rust 1.99.0、`-O -C lto`、依存・入力で各1回実行し、両方55件成功だった。
テスト部分は0.06秒／0.02秒、process壁時計は0.472秒／0.257秒。
ビルド・文書更新の費用を含む改善量でも、ばらつき・flake率の測定でもない。速度改善を主張しない。
実装本体は両比較版で同一、テスト差分は重複fixture選択とコメント、文書差分は現在の手順と版の説明である。
保守費用は未測定だが、wildcardの期待値を二箇所で保守する必要を減らした。

最初の依存選択は複数serde rlibで停止した。続く一時harnessの2ビルドはmoduleパスの指定誤りで失敗した。
通常のmodule配置に直して再実行し、同じ製品sourceと期待値で成功した。失敗ログもcheckout外で保持している。
この限定実行はworkspace全体・MLX・FFIのcheckを代替しない。

[統合版のPython原本](results-integrated.json)はPython 3.14.8／SQLite 3.53.1で68意味観測と3ngram構成を再実行した。
wire・hit・品質・展開診断、およびngramの時間以外の測定値は[初回原本](results.json)と一致した。
Python二群は0.003秒で成功した。書込み時間は新たな1回の観測で、順位・資源上限の採用には使わない。
製品入口は上記Rustで別に照合しており、Pythonのlegacy tokenを製品のsanitizerと同一視しない。

共有済み#307/#315報告は初回基準、公開head、最新main、統合版とも指定blobと一致する。
#312では内部Cow、query内memo、長語のschema検査が追加された。検索時planの比較と索引構成識別の役割は変わらない。
新plan、ngram、新検索言語、品質許容差、資源上限、strip、consumer移行は採用していない。

### 歴史証拠と現在のホスト引き継ぎ

公開head以前の固定consumer `547f9ee2ed734a2eab316fdbd62f194849875ee4`では、
[26fixture](consumer-cases.json)、[内部引用符4観測](consumer-quotes-host.json)、[group終端2観測](consumer-group-host.json)を確認した。
最終native FTS suiteは14件成功、SQLite 3.53.2、consumer自身のresolved lockを使用した。
[公開済み報告](https://github.com/thkt/rurico/blob/6b5a3855f35120c39423a1a49c656f5b58c98ee3/docs/research/issue-314/report.md)と
[公開済み再現手順](https://github.com/thkt/rurico/blob/6b5a3855f35120c39423a1a49c656f5b58c98ee3/docs/research/issue-314/README.md)はその版の説明として保持する。
[初回要約](host-verification.json)、[引用符要約](consumer-quotes-verification.json)、[group要約](consumer-group-verification.json)と
Pythonの[ホスト原本](results-host.json)を変更せず、固定版の証拠として残す。
これらのsource/binary hashは最新main統合版のhashと異なる。旧accepted・check・CIを今回へ流用しない。

初回評価後の文書不整合は、確認前の状態を証拠追加後も現在の状態として残したことが原因だった。
現在の操作説明を本節とREADMEへ更新し、旧観測は対象版付きで保持する。
空の評価履歴へ旧指摘IDのupdatesを返した応答は契約検査でinvalid_reviewとなった。
次のfresh独立評価は `updates=[]`、現在の指摘はIDなし `newItems` とし、Issue全体と今回の修正要求、変更文書を確認する。
前評価のacceptedを今回の評価へ変換しない。

sandboxではGit共有領域へ書き込めず、統合済みファイルとpatch・tree/hashを保持した。
ホストでは最新mainを再取得し、作業ファイルを保全して通常の未commit mergeを整えた。
`MERGE_HEAD`は上記mainで、未解決indexはない。mainの製品source・追加検証・依存定義を保持している。

新しい一時コピーで固定amiciの製品sourceを原本へ照合し、統合後のproducerへpath解決した。
[新しいconsumer要約](consumer-integrated-verification.json)のnative FTS suiteは14件成功し、
26fixture・quote4・group2のwire・回復構造・cleaner・直接/consumer MATCH・合成品質は旧原本と一致した。
consumer自身のresolved packagesとlockも前回と一致し、SQLiteは3.53.2だった。
rurico自身のlockとは分け、consumerの製品parserを変更していない。
内部quoteによるhit拡大と、引用内group終端での空結果は引き続き欠陥の観測であり、採用仕様ではない。

再現patchはすべて同じ原本末尾への追加なので、順番に`git apply`する試行は2枚目で失敗した。
一枚ずつ個別コピーへ適用し、原本が不変であることを確認して追加blockを合成した。
READMEの手順もこの条件へ直し、歴史的patchと旧ログは保持した。
標準checkはこのconsumer確認を実行せず、captureもnullなので、この新しい証拠を別に保存した。
変更文書を含む標準checkとfresh独立評価はこの統合状態から改めて実行する。

sandboxでの固定版日本語lintはSudachiPy依存不足で実行できず、手動確認までだった。
ホストで同じ固定版lintを実行できた。出典・版・数量・未確認範囲を保ち、
情報的な語彙反復・読解負荷の指摘は、条件や技術語を削る理由にしていない。
変更文書もfresh独立評価の対象に含める。

実データ、実モデル付き新planの品質、並行性、全文法・全consumer互換性、長期保守費用は未確認。
#307の数値と検索品質を新planの実測へ読み替えない。媒体は不要。GPU・実モデル測定は今回の確認には含めていない。
