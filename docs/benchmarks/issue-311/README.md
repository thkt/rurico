# Issue #311 前処理の比較とホスト検証

[Issue #311](https://github.com/thkt/rurico/issues/311)の意味を維持する最適化を、開始版
`fb62091fcf34b01bce080ea7d4ec588d4aa6a988`と比較する。
Cargo.lockのSHA-256は`743d754ae4fb6f0f44d169feab408a1b80caa50d17c38b058ec156c66811bf30`。
ホストの通常check・追加実測・独立評価は別に記録する。媒体は不要。

## 根拠と今回の変更

引き継いだ[#307報告](../../research/issue-307/report.md)の開始版blobは
`7dfa0a8c4248548bb81edbf183e0ea3ee72e45c5`で、着手時のファイルと一致した。
報告の27 embedding行・16 reranker行とamiciの結果はd063版・当時の条件の観測であり、
今回の変更版の実測や全入力の同値性ではない。報告の後続提案から新しい許容差や
CPU backendの導入許可を得たとは扱わない。
現在の[#307手順](../../research/issue-307/README.md)は#353でPython依存を更新しているため、
過去の報告の参照環境と同一とは扱わない。今回の前処理検証はRustの同じ固定tokenizerを使う。

Issueが見直したc8f250版から開始版までに、本番の結果組立は
`execute_document_chunks`へ移り、再sortも除去済みになっている。
このIssueで再sortを除去したという利益は計上しない。cloneとsortを一括した改善率も出さない。
既存の`token_budget=383`のforward順・端数・pause検証を再利用する。

- paddingは既存の`pad_sequences`でIndexedChunkのsliceを直接借用する。
  token Vecと、その一覧のcloneを作らない。rerankerなどのowned Vec入力とmask指定は維持する。
- prefix encodeを共通化し、plannerの全文判定と超過候補ではID・maskのVecを作らない。
  採用した候補だけIDをコピーし、長経路では全文Encodingを解放してからoffsetを取得する。
  短経路は従来どおり全文token列を使い、offset取得・分割を必要時にだけ行う。
- 全文のprefix付きencodeと本文offset用encodeの統合、chunkごとのprefix再tokenize省略、
  bufferの長期保持、粗い縮小・二分探索は採用していない。
  prefix境界mergeと非単調なtoken数があるため、IDsやoffsetを流用できるという仮説だけでは
  同じchunk内容を保証できない。意味を変える案は#316等の別範囲で判断する。

コードの小さい変更だけではCPU時間の改善を保証しない。保守費用は共通prefix関数と
AsRef実装・generic paddingの増加、比較用ハーネスの維持にある。実測で利益を確認できない
候補を複雑化して採用しない。行数やファイル移動は利益に数えない。

## モデル不要の検証

標準checkは[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)の契約をそのまま使う。
既存の本番組立テストへ借用paddingのID・0埋め・identity maskを統合した。
混在長・複数chunk・空bucket・空入力・非default予算・端数・失敗後の停止を維持する。
独自のflatten＋sortで順序を再現していたテストは、本番組立を検証しないため削除した。
その独自再構成だけの検出条件は失うが、4bucketを通るchunkを本番組立テストへ統合し、
本番結果の順序とbucket内のforward順は残る。

既存のWordLevelのprefix・overlap・末尾・文書順・縮小検証は維持した。
追加のAddedTokenはprefixと最初の日本語tokenを結合する。本文offsetだけを切ってprefix IDsを
連結する誤実装を検出し、UTF-8の末尾まで期待ID列を固定する。長い候補の方がtoken数が少ない
例も製品のencodeと縮小関数で確認する。実モデル・ネットワークを追加せず、毎回同じ入力を使う。
この追加保証はWordLevelのみでは得られない。実tokenizerの全merge規則までは保証しない。

## CPU比較

ホストで依存を先にビルドし、計測中はcheck・ビルド・他の重いCPU/GPU処理を止める。
実行前後の時刻、機種・OS・Rust/Xcode/Metal版、停止した負荷と計測中のprocess観測を残す。
次はrepo rootからの例。`--deps`はこのCargo.lockでビルドした依存を指定する。
通常のdebug依存ならLTO用bitcodeのリンク条件を追加せず再利用できる。
ハーネス自身と取り込む製品関数は`rustc -O`で同時にコンパイルするが、debug依存の時間を
製品releaseの絶対時間へ一般化しない。比較の両側で依存・flagsを揃える。`--prepare-only`を指定するとsnapshotとコンパイルだけを行い、
manifestの`command`で指定したバイナリを測定枠内で実行できる。実測時はsource・依存・
tokenizer・バイナリのhashを前後で照合し、rawの再集計と実行条件を別に保存する。

```sh
cargo build --locked --features test-support,test-mlx,smoke
python3 docs/benchmarks/issue-311/run.py --deps target/debug/deps \
  --out /tmp/rurico-311-synthetic-NEW
python3 docs/benchmarks/issue-311/run.py --deps target/debug/deps \
  --tokenizer /path/to/fixed-ruri-v3-310m/tokenizer.json \
  --out /tmp/rurico-311-tokenizer-NEW
```

固定tokenizerは[#307](../../research/issue-307/report.md)と同じ310m revision
`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`の既存cacheを使い、revision・実path・内容hashを照合する。
複数rlibがある場合は、例えば`--rlib libc=target/debug/deps/liblibc-HASH.rlib`で
同じビルドの実体を明示する。scriptは対象名・pathを検査し、全依存hashを保存する。
出力先はcheckout外の新規ディレクトリを要求する。失敗時の途中出力を成功と扱わない。

`run.py`は開始版と未commitの変更版sourceを保存し、製品のtokenization、planner、縮小、
indexing、bucket分配、padding関数をそのままコンパイルする。モデル読込み・forward型は
ハーネスのstubであり、この比較でforwardや結果組立の受入を代替しない。
clone前後のadapterは実`forward_sub_batch`のpadding呼出しと対応する。
sort比較は現行bucket分配へ、既に除去されたsortを加えた対照であり、開始版との変更利益ではない。
plannerや縮小を検証用に再実装しない。

raw JSONLの各caseは独立に比較する。
`padding_clone`、`redundant_sort`、`short_plan`、実際に3chunk以上を生成する`long_plan`、
超過候補から1tokenずつ縮小する`shrink`を7組、順序を交互にして記録する。
全caseで変更前後のIDs・pad/mask・chunk数・採用end・bucket順の一致を先に確認する。
実tokenizerではprefix長を実際に取得し、縮小経路に入らない場合はassertで停止する。

CPU時間はgetrusageのprocess user＋system、wall時間はInstant。
allocation/reallocation数とpeakは単独呼出しでのRust allocator要求を数える。
peakは呼出し直前のlive要求量からの最大増分で、RSS・native tokenizer・GPU peakではない。
allocatorのatomic観測費用は時間に含まれ、両版で同じ観測を行う。絶対時間や小さい差にはこの限界を付す。
入力・tokenizer構築、warm-up、コンパイルは計測区間外。sort比較のinput cloneは両側に含む。
中央値・範囲をrawから集計し、短経路から長経路へ推定しない。
source・Cargo.lock・依存・binary・tokenizerのhashをmanifestへ保存し、終了時にsource不変性を確認する。

## 2026-10-10のホスト実測

測定枠は回答後30分。実行は14:04:39〜14:18:51 JST、852.35秒で終了0。
測定の完了確認後に枠を解放した。実測中は別のビルド・GPU検証を止め、
既知の実行名を0.25秒間隔で2829回観測し、競合は0件だった。
短い処理や未知の実行名のGPU利用まで否定する観測ではない。
source・lock・準備した依存・バイナリ・固定モデル52ファイルのhashは前後で一致した。

固定310m tokenizerで7組、順序を交互にした比較を行った。
planner・縮小・padの新旧結果は、比較対象の入力で一致した。
CPU時間はuser＋system、peakはRust allocatorへの追加要求量であり、RSSやGPU使用量ではない。

| 処理 | CPU中央値 ms（開始版→変更版） | allocation/reallocation | 追加要求peak bytes |
|---|---:|---:|---:|
| paddingのclone | 0.01280 → 0.00751 | 131 → 2 | 626432 → 524288 |
| 短経路 | 0.06795 → 0.06732 | 439 → 438 | 17022 → 17022 |
| 長文planner | 81.3224 → 80.5790 | 320635 → 320630 | 4455304 → 4347982 |
| 縮小 | 102.3552 → 102.8274 | 407476 → 407459 | 2073656 → 2073656 |

paddingのCPU時間は開始版0.01255〜0.01372ms、変更版0.00739〜0.00774msだった。
短経路・長文・縮小の測定範囲は重なり、速度改善とは結論しない。
小さいallocation削減を採用理由とし、複雑なencoding再利用や縮小の変更は見送った。
ハーネスのdebug依存を含む比較であり、製品release全体の応答時間を示さない。

sortの別比較は、既にsortがない開始版に対する仮想的な追加処理である。
CPU中央値はsortあり0.1340ms、なし0.1295ms、allocationは1061→1057だった。
#311の変更による利益には計上しない。

実モデル検証は既存の`measure-records`で、W1/W2/W3、defaultと
`token_budget=256, forward_pause=1ms`、通常／計測・batch／singletonを実行した。
97推論記録、24群×3回のwarm測定と24summary、空入力が成功した。
既存fixtureの数値・chunk形状・順序を既定の許容差で検証し、forward形状、readback量と
pause回数も一致した。rawから`summary`を再生成して完全一致を確認した。
公式Python比較やamici品質の新しい評価ではなく、製品全体の新旧速度比較でもない。

通常checkも終了0。Rust 419件、doc test 3件、Python 4件、clippy・fmtが成功した。
初回のclippyはテストのabsolute path指定3件で失敗し、useへの修正後に成功した。
テスト全体の時間短縮・保守費用改善は未測定。
変更文書と実測証拠を独立評価へ渡し、PRと同じheadのCIを公開後に確認する。

- [CPU raw](cpu-raw.jsonl) / [再集計](cpu-summary.json) / [計測ハーネスmanifest](cpu-manifest.json)
- [実モデルraw](mlx-raw.jsonl) / [再集計](mlx-summary.jsonl) / [検証集計](validated.json)
- [測定条件と版](provenance.json) / [ビルド依存](dependency-build.json) / [環境](environment.json) / [固定モデル](models.json)

測定後は結果説明だけを更新した。runtime source・比較ハーネスは測定した版のままである。
合成入力の予備記録はホスト側に保存し、今回の受入や改善率には使っていない。
