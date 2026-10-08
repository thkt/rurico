# Embedding fixtureの保存と読込み

[Issue #304](https://github.com/thkt/rurico/issues/304)の形式実装は
`src/embed/fixtures.rs`を正本とする。`w1.bin`（9,240 byte）、`w2.bin`（308,004 byte）、
`w3.bin`（30,804 byte）は既存のlegacy基準出力で、今回再生成・移行していない。
生成由来はファイル内に記録されておらず、現在のModelIdのrevisionを遡って付与しない。

## 形式と制限

legacyはu32 LEのdocument数、各documentのchunk数、各chunkの次元、f32 LE列からなる。
version 1はその前にu32 LEのmagic `0xffffffff`、version `1`、JSONのbyte長、
UTF-8 JSONの生成条件を置く。magicはlegacyのdocument数として予約する。
JSONはproducer（`rurico`・`official_reference`・`synthetic`）、model、model_revision、
tokenizer、文書順のinputs、generation_code、settingsを必須とする。
識別文字列は空白だけを許さず、input数はdocument数と一致させる。未知のJSON field・versionは拒否する。
これらは生成者による記録であり、model互換性や真正性の証明ではない。

`load`のserialized byte上限は64 MiB。既存最大fixtureを受理し、破損headerから
数GiBを確保しないための形式側の既定値で、推論資源のhard limitではない。
必要なconsumerは`load_fixture(reader, max_bytes)`で明示的な上限を渡す。
読込みは上限+1 byteまでに制限し、上限超過を`InvalidData`として拒否する。
デコード前に実データを読み、countの最小header長と次元×4をchecked演算で照合する。
headerをVecのcapacityに使わず、各vectorのpayload範囲確認後にallocationする。
空vector・document内の次元不一致・NaN/±Infは#301の検証を再利用して拒否する。
ゼロdocumentは有効。document間の次元差は許容し、model/indexの同一性は推定しない。
truncationは`UnexpectedEof`、乗算overflow・余剰byte・不正metadataは`InvalidData`。
readerのI/Oエラーは伝播する。読込みは一つの完全なfixtureがEOFまであることを前提とし、
連結fixtureや終わらないstreamのためのAPIではない。byte上限はデコード後の総RSS上限ではない。

## APIの移行

`load`は両形式を読み、vectorだけを返す。由来を確認する利用側は`load_fixture`を使う。
戻り値の`generation: None`はlegacyの由来不明を示す。
`save`は互換用のlegacy writerで、内容・総byte数を検証しない。
新規生成には`save_versioned`と明示した`GenerationConditions`を使う。
新writerは内容と生成条件、64 MiB上限を検証してから書込み、失敗は`io::Error`で返す。
上限はversion header・生成条件JSON・legacy payloadを含むserialized byte数に適用する。
検証・サイズ確認の失敗では外部writerへ1 byteも書かない。
`capture-fixture`は新しいFileを直接渡す。versioned writerが検証・計数・符号化を
内部で完了し、完成byte列を外部writerへ一度の`write_all`で渡すため、
呼出側で短い書込みを集約するbufferは不要である。
書込み開始後のwriterエラーは伝播するが、部分出力のrollbackは保証しない。
正常legacyを読んで新形式へ保存する場合も、由来を確認できない情報で埋めてはいけない。
由来を回収できなければlegacyとして保つ。

`compare`のエラーは`ShapeMismatch`から`CompareError`へ変わる。
旧`Err(ShapeMismatch::Dim { .. })`のmatchは`Err(CompareError::Shape(ShapeMismatch::Dim { .. }))`
へ移行する。内容エラーは`InvalidContent { side, doc, source }`で位置と
`EmbeddingValidationError`を返し、非有限metricは`NonFiniteMetric { doc, chunk }`で失敗する。
利用側は新variantと将来のvariant用wildcardも処理する。
legacy constructorからの値も両側とも比較前に検証し、shapeが同じでも不正内容を成功にしない。
内積・normはf64で計算し、f32の大きな有限値でも計算overflowを避ける。
f32で表現できない絶対差はエラーにする。許容差（cosine ≥ 0.99999、絶対差 ≤ 1e-5）は変更しない。
同一ゼロvectorはcosine 1、ゼロ対非ゼロは0、両側ゼロdocumentも完全一致を維持する。

## 再生成と検証

公開合成の正常legacy・version 1、破損次元・Inf・余剰byteの例は
[`../embedding_format`](../embedding_format)に置く。私的本文・モデル推論を含まない。
checkout rootで次を実行すると、同じ小さなfixtureを再生成できる。

```sh
python3 tests/fixtures/embedding_format/generate.py
cargo test --locked --lib embed::fixtures::tests
```

実モデルの新しい基準を意図して採取する場合は、公開入力のW1/W2/W3と固定cacheモデルを使い、
別checkoutで再ビルドしてから実行する。既存基準を上書きして今回の検証を通してはいけない。

```sh
cargo run --locked --bin mlx_smoke --features smoke --release -- capture-fixture
```

このmodeはversion 1を保存する。固定model/tokenizer revision、入力、設定、
既存smoke contextの現在commit・差分hash・untracked hash・Cargo.lock hash・binary hashを記録する。
必須code識別情報が取得できない場合は停止する。contextのcommitは実行時のcheckout版であり、
その版からbinaryをビルドしたことを自動証明しないため、再ビルドが必要。
cacheのローカル変更をhash検証していないこともtokenizer欄とcontextに明記する。
公式参照出力はproducerを`official_reference`とし、実際に使ったmodel/tokenizer・入力・参照コードと
設定を記録して別の出力先へ保存する。[#307の既存手順](../../../docs/research/issue-307/README.md)は維持する。
[#315の設計例](../../../docs/research/issue-315/README.md)は提案であり、この形式の採用前提にはしない。

標準checkはfixture単体テストとsmokeのモデル不要テストを実行する。
この変更は読込み・比較・生成記録の契約を検証し、モデル数値や検索品質の再測定を主張しない。
実モデルの比較は既存の`verify-fixture`を別に使う（[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)）。

### 保存上限の回帰検証

`versioned_save_byte_limit_is_checked_before_writing`は、公開`save_versioned`が
64 MiBを渡す内部helperと同じ保存経路へ、小さいfixtureと上限を渡す。
serialized byte数をheader・JSON・document/chunk数・値数から求め、上限以下と境界では
公開APIと同じbyte列、1 byte超過では`InvalidData`と`fixture exceeds byte limit`を確認する。
超過時には外部writerの既存prefixが変わらないことも確認する。
巨大fixtureやByteCounter単体の検査でこの保証を代用しない。

検出力を確認する際は、変更版とCargo.lockを固定したcheckout外の一時コピーを使う。
通常版で次を実行した後、そのコピーだけで`ByteCounter::write`の
`if self.bytes > self.max_bytes { ... }`による上限拒否を外し、同じテストを実行する。
`oversized fixture wrote to external writer`のassertionで失敗することを確認する。
元のsourceをコピーへ復元し、同じテストの成功を確認する。コンパイル失敗は検出成功に数えない。

```sh
cargo test --locked --lib embed::fixtures::tests::versioned_save_byte_limit_is_checked_before_writing -- --exact
```

source・差分・Cargo.lockのhash、ツールチェーン、各実行の終了コードとstdout/stderrを
checkoutと既存runの外へ保存する。通常checkは正常版のテストを実行するが、この一時改変は実行しない。
今回のsandboxではMLXビルド用sourceの取得がDNS制約で失敗し、テスト実行へ到達していない。
その後のホスト検証では、通常版・復元版は各1件成功し、上限拒否を外した版は
`oversized fixture wrote to external writer`のassertionで終了101となった。
[結果と版](issue-304-save-boundary.json)・[対象出力](issue-304-save-boundary.txt)を保存した。
traitのflushや有限入力から通常到達しないcosine防御経路を、件数合わせで実行する検査や
coverage除外は追加しない。

Issue #304の追加ホスト検証では、sandbox外のApple Silicon・Metal環境で、
同じ変更版とCargo.lockから次をビルド・実行する。
`cl-nagoya/ruri-v3-310m`のrevision `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`を
既存のcache取得手順で配置する。標準checkはこの実モデルmodeを実行しない。

```sh
cargo run --locked --bin mlx_smoke --features smoke --release -- verify-fixture
```

W1/W2/W3それぞれのcosine・最大絶対差、全workloadの成功表示、終了コード、
開始commitと変更差分、Cargo.lockと使用binaryのhash、Rust/Xcode/Metal版を証拠に残す。
seatbeltによる78終了、モデル未配置、推論失敗は成功にしない。
既存fixtureを変更せず、同じ版のcheck・文書を含む独立評価へ戻す。
UI媒体は不要で、capture契約の追加・変更は行わない。

## 2026-10-08の追加ホスト検証

開始版`24725a72be44300afc24186b82b14bcb3f5f9d3d`に今回の未commit差分を加え、
既存コメントの重複・不要な説明を整理した後、同じCargo.lockからrelease binaryをビルドした。
Apple M3 / macOS 27.0.1 / Rust 1.99.0 / Xcode 27.0 / Metal 32023.921で
`verify-fixture`を実行し、約24.95秒・終了コード0で全workloadが成功した。

| workload | cosine表示値（小数6桁） | 最大絶対差の表示値 |
| --- | ---: | ---: |
| W1 | 1.000000 | 9.537e-7 |
| W2 | 1.000000 | 4.172e-7 |
| W3 | 1.000000 | 5.960e-7 |

既存のcosine ≥ 0.99999・最大絶対差 ≤ 1e-5を満たした。
[出力](issue-304-verification.txt)と[版・条件・hash](issue-304-verification.json)を保存した。
source・fixture・Cargo.lock・binaryは実行前後で一致し、既存fixtureは再生成していない。
model/config/tokenizerのローカル内容も実行後に固定revisionの公開済みhashへ再照合した。

このタスクの別ビルド・GPU検証を重ねず、事前10秒と実行中0.25秒間隔で負荷を観測した。
実行中85観測に既知の競合processはなかったが、短い活動や未知のGPU利用を完全には除外しない。
総RSS、32-bit固有のoverflow分岐、公式参照との再比較、consumer検索品質、速度改善は未確認。
測定後にはこの説明と証拠を追加し、さらに読込みの二重検証を修正した。
各vectorの即時検証は保ち、検証済みの所有データを最後に構築する際の再走査を除いた。
空chunkの原因型、値・入力順・IDは保持し、推論・比較計算・入力・fixture・Cargo.lockは変更していない。
JSONのhashと上記数値は修正前の測定版の記録であり、修正後のbinaryを実測した結果ではない。
同じ値を構築する局所修正のため既存の数値観測を参照するが、読込み時間や総RSSの改善は未測定。
修正後の内容拒否・原因型・値・順序・IDはfixture単体テストで検証し、標準checkと独立評価へ戻す。
PR #376のhead `fef306fc2e60211f8cc6653c3a75b00230a5e6eb`からの保存上限検証の補強では、
公開writerの既定64 MiBを内部helperへ渡し、小さい上限で同じ保存経路を検証できるようにした。
この変更は保存経路・テスト・説明に限り、読込み・推論・比較計算・入力・fixture・Cargo.lockは変えていない。
`verify-fixture`はlegacyの読込みと比較を使い、versioned保存は呼ばないため、今回の保存経路変更の
検証を旧W1/W2/W3観測で代用しない。保存境界テストと上限拒否を外す検出力確認を使う。
続くcapture呼出側の修正では、不要になったBufWriterを削除し、同じ保存先のFileへ直接渡す。
保存境界のホスト証拠が対象とした保存実装・テスト・Cargo.lockはこの修正でも不変で、
その検出力確認を再利用できる。呼出側のコンパイルとモデル不要smoke検証は標準checkで確認する。
この証拠は実モデルのcapture実行やFileへの書込み結果を実測したものではない。
旧出力・JSON・数値は引き続き二重走査修正前の測定版に限る。今回のbinaryの実モデル検証、
保存上限64 MiBそのものの巨大fixtureによる実測、実行時間・総RSS・保守費用の改善は未確認である。
