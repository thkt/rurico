# Issue #314 検索意味とquery planの比較

[報告](report.md)は、現行literal契約、phrase・短語の実MATCH、型付きplanと小さなAPI修正の比較、consumerへの影響を示す。
製品API・検索言語・索引方式は採用していない。

`cases.json`は公開合成入力と評価意図、`results.json`はSQLiteでの測定原本。
`plan.py`は未採用の参照例で、文字列からphraseやBooleanをparseしない。
`run.py`は同じ入力を現行SQLの参照例・決定的順序案・vocab不在・明示phraseで比較する。
製品のsanitizerを移植せず、fixtureに明示したlegacy tokenを使う。
その差を見逃さないため、Rustの既存searchテストに追加した一群が製品の
`prepare_match_query`でphrase・候補上限・fallbackのwireとhitを照合する。
LIKE escapeは最新mainの既存Rustテストへ統合した。Rustの実行は標準checkに含まれる。

Python標準ライブラリだけで実行できる。SQLiteにFTS5/trigramが必要で、モデルやサーバーは使わない。

```sh
python3 -B -m unittest discover -s docs/research/issue-314 -v
python3 -B docs/research/issue-314/run.py --output /tmp/issue-314-new-results.json
```

出力先は未使用のファイルを指定する。既存の測定原本を自動で上書きしない。
品質は結果内の`quality`、実MATCHは`wire`と`hits`を参照する。
報告の表はこのJSONを解釈したもので、再実行時はSQLite版・source hash・入力・評価意図を揃えて比較する。
短い書込み時間の一回測定は速度の優劣を保証しない。

標準setup/checkは[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)どおり。
Python研究検証は標準checkへ登録していない。初回・追加ホスト・最新main統合版で上の両検証を実行した。
captureは不要。変更文書も既存の独立評価へ渡す。

## 最新main統合版のホスト確認

競合修正の開始headは `6b5a3855f35120c39423a1a49c656f5b58c98ee3`、
統合mainは `56f412c8401d6a3b501d7effb0aff18df1f344e5`、共通祖先は
`24725a72be44300afc24186b82b14bcb3f5f9d3d`。Cargo.lockはmainと同じSHA-256
`743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`だった。
query内memo・内部Cow・schema検査を含む製品入口は統合済みで、CPU限定の
[検証要約](integration-verification.json)と[Python再実行](results-integrated.json)を保存した。
この結果はworkspace checkや固定consumerの新しい実行を代替しない。

Git共有領域への書込みが拒否された場合は、統合済みファイルとcheckout外のpatch/hashを保ち、
ホストが同じcheckoutに通常の未commit merge状態を整える。設定・前run・認証情報は変更しない。
現在のsourceと固定consumerを結び直すため、下の固定版取得・source照合・path patch手順を使い、
新しい一時コピーで3つの再現patchをtestsだけへ適用する。
各patchは同じ原本末尾への追加差分なので、個別コピーで一枚ずつ適用を確認してから、
原本が不変であることを確認した追加blockを一つのtestsへ合成する。
同じファイルへ順に`git apply`すると2枚目以降が失敗する。
consumerのlockは今回の依存解決結果として記録し、旧lockと違う場合は差と理由を示す。
旧lockへの強制置換や、rurico自身のCargo.lock更新は行わない。

`RURICO_314_CASES`は今回のcheckoutの `cases.json` を指定する。
同じpath patch設定で `storage::fts::tests` を実行し、既存round-tripと26fixture・quote4・group2の
各filterが実行されたことを確認する。旧suiteは14件だったが、その件数だけを合否条件にしない。
期待する結果は旧観測との比較であり、quote/groupの欠陥を望ましい製品仕様として採用しない。
source・resolved packages・lock差分・binary・rurico差分のhash、実行対象と終了コード、
回復構造・cleaner・直接/consumer MATCHのhitと品質、所要時間を新しい証拠へ保存する。
既存の原本を上書きしない。結果が違えば原因を調べ、固定consumerの製品parserは修正しない。

新しい要約を報告へ反映してから、変更後の `cargo fetch --locked` と設定済み
`bash scripts/check.sh`、変更文書を含むfresh独立評価へ戻る。
初回評価は `updates=[]`、現在の指摘はIDなし `newItems`。PR全体と今回の採用済み競合修正を両方確認する。
既存本文の適用される説明・限界・公開済みリンクを新しいassessments/handoffへ明示し、旧成功の主張を引き継がない。
commit・push・公開は今回のsandbox作業では行わない。

## 固定amiciの再現手順と過去のホスト確認

以下の旧観測の対象はrurico開始版`24725a72be44300afc24186b82b14bcb3f5f9d3d`＋当時の差分と固定Cargo.lock、
amici `547f9ee2ed734a2eab316fdbd62f194849875ee4`。
再実行では上の最新main統合版を使い、producer・lock・binaryの新しいhashを記録する。
ネットワークが使える担当AIが、checkout外の新しい一時領域へこのamici版を取得する。
[既存のsource manifest](../issue-307/results/search/amici-source.json)の
`src/storage/fts.rs`と`src/storage/fts/tests.rs`のSHA-256へ照合し、parser、cleaner、既存round-trip testを読む。
同じ版の`docs/decisions/0008-pin-the-fts5-trigram-wire-format-contract-with-rurico.md`も読み、
wire契約の適用条件と合意状態を現行sourceへ照合する。
当初のGitHub取得不能はホストで解消した。本文・hash・実行結果は[検証要約](host-verification.json)に残した。

未公開のrurico差分を使用するため、amiciの通常/dev依存のgit URLに対するCargo path patchを
コマンドの`--config`で今回のcheckoutへ指定し、一時領域内のlockだけを解決する。
[既存手順](../issue-307/search_quality.py)と[過去のpath patch観測](../issue-307/results/search-manual/provenance.json)を参照し、
`cargo metadata`でruricoが一つ、sourceがpath、manifest_pathが今回のcheckoutであることを確認する。
対象checkoutのCargo.lockや他repoの作業checkoutを書き換えない。

amiciの一時checkoutで、同じpatch設定を付けて次を実行する。

```sh
cargo test --locked --lib parse_fts_segments_recovers_rurico_or_group_wire_format
```

filterに対応するテストが実行されたことを確認し、0件を成功扱いしない。
テスト内容を読んだうえで、今回のphrase・引用符・operator literal・`%`/`_`/backslash・展開groupが
既存parser→cleaner→MATCH経路で何を保持／失うか確認する。不足する条件は一時コピーの小さな再現で補う。
PhraseとLiteralの型の区別、打切り/fallback診断は現行文字列から回復できない、という提案の前提も確認する。
新しい言語の採用やconsumerの変更は行わない。

期待する証拠は、source照合、依存解決とlock差分、対象rurico差分hash・Cargo.lock hash、
実行対象名・終了コード・実MATCH/hit・parserから回復した構造と失う情報。
生ログはcheckout外へ保存し、要約を[報告](report.md#ホスト確認と残る検証)へ反映してから同じ差分のcheck・独立評価に戻る。
標準checkはruricoだけを実行し、captureはnullなので、このconsumer確認を代替しない。
実モデル再測定は今回の変更には必要ない。#307の品質値を新planの品質へ読み替えない。

2026-10-08のホスト実行は既存round-trip1件と追加MATCH照合26観測を確認した。
[再現patch](consumer-cases.patch)は一時コピーのtestsだけに追加する。固定consumerの製品変更ではない。
`RURICO_314_CASES`に今回の`cases.json`を指定し、同じpath patch・resolved lockで
`issue314_host_current_producer_parser_cleaner_match_cases`を1件実行する。
結果と限界は[報告](report.md)・[consumer記録](consumer-cases.json)を参照する。
Python側の再実行は[別の原本](results-host.json)に保存し、既存`results.json`を上書きしない。

## 内部引用符の再現

26観測の旧原本には内部引用符入力がない。新しい確認は[consumer-quotes.patch](consumer-quotes.patch)を使う。
過去のcheckout・lock・生ログは変更せず、新しい隔離コピーで固定amici版とsource hashを照合する。
上のpath patchで今回のrurico checkoutへ解決し、consumerのlockを
[既存要約](host-verification.json)のSHA-256と比較する。依存版が変わった場合は同条件の証拠とせず理由を報告する。
再現patchは固定版の`src/storage/fts/tests.rs`へ追加するだけで、製品parserを変更しない。

追加filterは`issue314_host_internal_quote_round_trip_observation`。
`say`を正常な対照、`say"hi`を内部引用符入力とし、同じ3文書・両tokenizer・正規化全OFFで、
現在のproducer wire、parserのfixed/groups、cleaner出力、直接MATCHとconsumer MATCHのhitを記録する。
期待する差は`say`で両経路 `[1,2,3]`、`say"hi`で直接 `[1]`、consumer `[1,2,3]`。
後者の合成評価意図は文書1のみ関連で、consumerのprecisionは1/3、recallは1となる。
2026-10-08の固定consumer native SQLite 3.53.2で両tokenizer・4観測を確認した。[実測](consumer-quotes-host.json)と[版・実行要約](consumer-quotes-verification.json)を参照する。再実行で差があれば原因を調べ、checkを通すために弱めない。

これは調査用の欠陥観測で、広いhitを受入仕様にするテストではない。
ホストでfilterが1件実行されたこと・終了コード・SQLite版・所要時間を記録し、
`issue314-quotes:`のJSONと生ログを新しい原本へ保存する。既存consumer-cases.jsonやpatchは上書きしない。
新しいsource/依存/lock/binary hashとrurico差分hashを紐付け、
[報告の再評価への引き継ぎ](report.md#再評価への引き継ぎ)と検証要約を実測へ更新する。
対象repoのCONTRIBUTINGがMetalを使用し得るcheckをsandbox外ホストに指定しているため、
このnative Rust検証も担当ホストAIへ引き継ぐ。通常checkとcaptureだけでは固定consumerの追加filterは実行されない。

## 引用内のgroup終端の再現

[consumer-group.patch](consumer-group.patch)を固定consumerの新しい一時コピーのtestsへ適用し、同じpath依存解決で`issue314_host_quoted_group_terminator_observation`を実行する。trigram索引の`日本語`と`日)本`を対照に、入力`日`の展開wire、回復構造、cleaner、直接MATCHとconsumerの結果を記録する。[2観測](consumer-group-host.json)では後者だけが空結果へ落ちた。広い互換性や製品改修の採用を保証する観測ではない。
