# Issue #314 検索意味とquery planの比較

[報告](report.md)は、現行literal契約、phrase・短語の実MATCH、型付きplanと小さなAPI修正の比較、consumerへの影響を示す。
製品API・検索言語・索引方式は採用していない。

`cases.json`は公開合成入力と評価意図、`results.json`はSQLiteでの測定原本。
`plan.py`は未採用の参照例で、文字列からphraseやBooleanをparseしない。
`run.py`は同じ入力を現行SQLの参照例・決定的順序案・vocab不在・明示phraseで比較する。
製品のsanitizerを移植せず、fixtureに明示したlegacy tokenを使う。
その差を見逃さないため、Rustの既存searchテストに追加した一群が製品の
`prepare_match_query`で必要な入力のwireとhitを照合する。Rustの実行は標準checkに含まれる。

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
Python研究検証は標準checkへ登録していない。初回sandboxと追加ホストで上の両検証を実行した。
captureは不要。変更文書も既存の独立評価へ渡す。

## 固定amiciのホスト確認

対象はrurico開始版`24725a72be44300afc24186b82b14bcb3f5f9d3d`＋今回の差分と固定Cargo.lock、
amici `547f9ee2ed734a2eab316fdbd62f194849875ee4`。
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

## 内部引用符の追加ホスト確認（R1-2）

26観測の旧原本には内部引用符入力がない。新しい確認は[consumer-quotes.patch](consumer-quotes.patch)を使う。
過去のcheckout・lock・生ログは変更せず、新しい隔離コピーで固定amici版とsource hashを照合する。
上のpath patchで今回のrurico checkoutへ解決し、consumerのlockを
[既存要約](host-verification.json)のSHA-256と照合する。依存版を変えた場合は同条件の証拠とせず理由を報告する。
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
