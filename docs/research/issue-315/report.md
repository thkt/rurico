# Issue #315 embedding・FTS構成識別の設計報告

索引とは別に構成記録を保存し、追加書込みと検索を別のAPIで照合する案を推奨する。
embeddingは共通モデル情報と操作別条件に分け、FTSは独立した索引構成として扱う。
既存vector・`ChunkedEmbedding`・fixture形式を変えず、由来のない旧データは不明のまま残す。
この結論は設計提案であり、公開API・保存契約・前処理変更の採用合意ではない。

## 根拠と適用範囲

要求の正本は[Issue #315](https://github.com/thkt/rurico/issues/315)（提供された更新時刻 `2026-09-27T11:55:00Z`）。
2026-09-27の開始commitは `eede511287ab8b7fa131169afa0afa351504373c`、ローカルの`origin/main`も同じだった。
Cargo.lockのSHA-256は `05fa187fa27454c4afef44554be20360b7af181c8704690b95c448cdbae4c1a2`。
`git ls-remote origin refs/heads/main`はDNS解決に失敗し、リモートの最新mainとの差は未確認。
以下の資料は開始commitの内容と作業開始時のファイルを照合した。引き継ぎ版の差し替えはない。

| 根拠 | 照合した版・状態 | 今回に適用する内容 |
| --- | --- | --- |
| [#307報告](../issue-307/report.md) | blob `26d0ed6993eff2ca7bde591481224af7206261eb`、提供blobと一致。過去の観測 | 同一token/shapeのFP32比較とwrapperの前処理差。索引互換性の保証には使わない |
| [ADR-0011](../../decisions/0011-adopt-uniform-vector-length-contract-for-the-embed-trait.md) | blob `e313d9b8c2c205484b937939c5d9a54edf778ca1`、一致、accepted | 単一implementorの同一次元契約を維持。次元一致だけで構成互換とはしない |
| [ADR-0012](../../decisions/0012-adopt-symmetric-phase-5-query-normalization.md) | blob `27f2d15c5279aa5fcc0b81c23074571482577c2e`、一致、accepted | FTSの対称正規化、過去baseline欠落時の全OFFを維持 |
| [ADR-0006](../../decisions/0006-eval-harness-migration-to-amici.md) | blob `1e2403400b586d3a09907eb12998f4b6c2f2fa81`、一致、accepted | 検索品質評価の所有者はamici。今回の照合例で品質を判定しない |
| [ADR-0001](../../decisions/0001-typed-fts-query-contract.md) | 開始commit、accepted、#297の訂正を含む | `MatchFtsQuery`境界を維持。古いoperator実行の説明でなく訂正を適用 |
| [#306報告](../../benchmarks/issue-306-metrics.md) | 開始commit。最終隔離実測と以前の負荷重複記録を区別 | model proof、source manifest、runtime metricsの分担を再利用。時間差を今回の費用推定に転用しない |

上表のコード・資料はすべて[固定した開始版](https://github.com/thkt/rurico/tree/eede511287ab8b7fa131169afa0afa351504373c)から辿れる。
Issueの合意は調査と設計例まで。以下の型・識別方式・検索pairは今回比較する提案であり、accepted ADRを置換しない。
[README](../../../README.md)のEmbed/FTS契約と[CONTRIBUTING](../../../CONTRIBUTING.md#テスト)のMLX検証方針を適用する。
対象に`docs/wiki/`の開発方針は存在しない。現行操作はこのディレクトリのREADME、今回の観測と判断は本報告に分ける。

## 現行コードから確認した境界

- [`Embed`](../../../src/embed.rs)はquery、chunked document、任意prefixのtextを持ち、options付きbatchはbest-effort hints。query/textは8192 tokenへtruncateし最後をEOSにする。`tokenize_with_prefix`はprefixと本文を連結してからspecial token付きでencodeし、外側のtrimを行わない。
- [`plan_document_chunks`](../../../src/embed/processing.rs)は短文をそのままencodeし、長文は本文tokenのoffsetから候補を切り出す。prefixを付けて再tokenizeし、8192を超える間は候補末尾を縮め、2048 tokenのoverlapで進む。単なる「8192/2048」では境界決定や再tokenizeの意味を表せない。
- [`pooling`](../../../src/embed/pooling.rs)はmask付きmean（有効なprefix/special tokenを含む）とL2、下限`f32::MIN_POSITIVE`を使う。重み精度、演算・pool精度、出力f32は別項目として識別すべきである。
- [`ModelArtifact`](../../../src/model_io.rs)のrepo/revisionは取得先の宣言。[`VerifiedArtifacts`](../../../src/artifacts.rs)は内容hashや由来の公開型ではなく、検査後の変更を防がない。[MLX constructor](../../../src/embed/mlx.rs)は検査時のconfig/tokenizerを利用する一方、重みはpathから再ロードする。後から3ファイルをhashしても、その組がロード時の内容だったとは証明できない。
- [`QueryNormalizationConfig`](../../../src/storage/query_normalize.rs)はserde対応済み。NFKC → ASCII lowercase → whitespace collapseの順、runtimeは全ON。全OFFの`pre_phase_5_disabled`は過去baselineのserde-default用であり、構成識別情報の欠落を補う規則ではない。
- [fixture](../../../src/embed/fixtures.rs)は件数、各chunkの次元、f32列で生成構成を持たない。数値比較と構成識別は別の検証になる。

## 3案の比較

| 案 | 誤った構成の受理 | 既存APIへの影響 | 取得・保存・照合の費用 | 保守量・判断 |
| --- | --- | --- | --- | --- |
| consumer独自version文字列 | version更新漏れ、query/documentの対応漏れ、既定値変更を検出しにくい | rurico変更なし | 文字列の保存・比較は小さい。意味と内容の確認は各consumerへ残る | 最小の基準案。項目定義・migration判断が重複するため共通契約としては見送る |
| 構成記録＋照合API | 必須情報とrole別比較で既知の差を拒否。不明・虚偽のproducer申告には別の取得根拠が必要 | 既存traitを変えず追加可能。索引を開く/構成を替える箇所に導入 | 初回内容hashは総byte数に比例。保存は索引世代ごと1記録、比較は記録サイズに比例。毎推論hash不要 | 推奨。schemaと意味の版を保守する費用は増えるが、責務を一箇所に置ける |
| 全vector/`ChunkedEmbedding`にIDを埋め込む | 結果と記録の対応を追いやすいが、誤ったIDや古いproducerの申告は防げない | trait返却型、storage、fixture、mockに波及 | 各vectorへIDまたは参照を複製。対応表と移行も必要 | 必須性を確認できず見送る。今回の索引単位照合には過大 |

費用は構造上の比較で、hash時間、メモリ、運用・保守時間は未測定。行数削減や性能改善は主張しない。

## 推奨する公開型と取得API

以下はRustでの追加面を示す設計宣言で、現在のcrateからimportできるコードではない。
`EmbeddingSpec`は意味の記録、`SpecSnapshot`は取得根拠とproducer世代を持つ。前者だけをhash対象にする。
各structはフィールドを非公開にし、検証するconstructorとread-only accessorを提供する案とする。

```rust,ignore
pub struct EmbeddingSpec { common: ModelSpec, operation: OperationSpec }
pub struct ModelSpec {
    producer: ProducerIdentity,       // namespace + model/revision/implementation semantics
    artifacts: ArtifactDigests,       // weights/config/tokenizer: SHA-256 of exact bytes
    tokenizer_semantics: SemanticId,  // parser/normalizer behavior, separate from file identity
    dimensions: u32,
    pooling: PoolingSpec,
    normalization: OutputNormalization,
    precision: PrecisionSpec,         // weights / compute / accumulation / output
}
pub enum EmbeddingOperation { Query, Document, Text { prefix: String } }
pub struct OperationSpec {
    role: EmbeddingOperation,
    preprocessing: PreprocessingSpec, // exact prefix, order, outer whitespace semantics
    sequence: SequenceSpec,           // special tokens, truncation/EOS or chunk algorithm
}
pub struct FtsIndexSpec {
    normalization: QueryNormalizationConfig, // reuse rurico::storage's existing type
    normalization_semantics: SemanticId,
    consumer: ConsumerFtsSpec,        // explicit tokenizer/options/preprocessing + owner/version
}
pub enum SpecAvailability<T> { Known(T), Unknown(Vec<UnknownReason>) }
pub struct SpecSnapshot<T> { spec: T, evidence: AcquisitionEvidence, generation: Generation }
pub enum UnknownReason {
    Legacy, UnsupportedProducer, LoadedContentUnbound, ArtifactChanged, StaleGeneration,
    UnsupportedSchema(u32), UnknownField(String), MissingField(String),
    UnsupportedSemantics(String),
}
pub trait DescribeEmbedding: Embed {
    fn describe(&self, op: &EmbeddingOperation)
        -> Result<SpecAvailability<SpecSnapshot<EmbeddingSpec>>, SpecReadError>;
}
pub struct DescribedEmbed<'a> { embed: &'a dyn Embed, /* optional matching descriptor */ }
pub enum Verdict { Match, Different, Unknown }
pub struct Comparison {
    verdict: Verdict,
    differences: Vec<FieldDifference>, // path and stored/current value
    missing: Vec<UnknownReason>,       // path and unsupported/missing/acquisition reason
}
pub fn compare_append(stored: &EmbeddingSpec, current: &EmbeddingSpec) -> Comparison;
pub fn compare_search(stored: &EmbeddingSpec, query: &EmbeddingSpec,
                      approved: &SearchPair) -> Comparison;
pub fn compare_fts(index: &FtsIndexSpec, query_index_contract: &FtsIndexSpec) -> Comparison;
pub fn read_embedding_record(bytes: &[u8])
    -> Result<SpecAvailability<EmbeddingSpec>, SpecReadError>;
pub fn fingerprint(spec: &EmbeddingSpec) -> Fingerprint;
```

readerは不正JSONを`SpecReadError`、legacy・未知version/field・欠落を理由付きUnknownへ分ける。
比較関数へ渡せるのは検証済みの完全な型だけで、取得・読込みでUnknownになったものはcallerのgateが同じComparison形式へまとめる。
FTSとpairにも同じ読込み境界を用意する。Python参照例ではこの型境界を`problems`で検査し、相違値の複製を省いてfield pathを返している。

`EmbeddingSpec`という名前は操作別recordにも使えるので推奨する。`EmbeddingConfig`ではロード済み内容の証拠まで設定値と誤解しやすく、`EmbeddingFingerprint`だけでは相違理由を返せない。
`DescribeEmbedding`を別traitとしてopt-in追加し、具象`Embedder`の実装は採用後に追加する。
既存`Embed`、`dyn Embed`、mockの必須methodは増えない。`&dyn Embed`だけを持つcallerは
`DescribedEmbed::unknown(embed)`相当のadapterで`Unknown(UnsupportedProducer)`を得る。
対応producerは同じインスタンスからdescriptorと推論参照をまとめてadapterへ渡す。別producerのdescriptorを結び付けない。
既存traitへ`describe`のdefault methodを追加する案はdynから直接取得できる利点があるが、同名methodとの曖昧性やtraitの拡張を伴うので最小案では見送る。
全implementorへ必須methodを追加する案、downcast必須案も見送る。

取得時にはモデル内容の証拠が必要であり、この追加面だけで既存`Embedder`が完全なKnownを返せるわけではない。
採用実装ではloaderが使う重みbyte列と、parseに使うconfig/tokenizerのbyte列を同じ不変snapshotからhash・ロードし、
その組と結果をインスタンスへ保持する。ファイルを前後でhashするだけでは途中の差し替えを排除できない。
不変snapshotの作成・メモリ増分、mmapなどloaderとの具体的な接続は後続実装で検証する。
実装がその対応を証明できない場合は`Unknown(LoadedContentUnbound)`、読込み失敗は`SpecReadError`、
検査後のファイル変更は`Unknown(ArtifactChanged)`として書込み/検索のgateを通さない。
固定revisionだけの`DeclaredRevision`を`LoadedContent`へ昇格しない。

hashはmodel loadまたは信頼できる不変artifact snapshot作成時に一度計算して保持する。
hash対象はファイルの全byte列なので、意味が同じconfigを整形し直しただけでも内容IDは変わる。誤受理を避ける保守的な識別である。
署名のないhashは真実性や認証を保証しない。権限のないwriterが索引と記録を両方書き換える攻撃は別責務である。
mock/custom producerは所有者の名前空間（例`example.org/mock`）と不変な実装意味の版を用い、合成重み等のdigestは実物と混同しない。
artifactを持たないcustom producerの採用版では、名前空間付きの`CustomRecipe { implementation, parameters_digest }`を別variantにする。
内容が分からないサービスは`Unknown`に留める。実行例はartifact型の合成mockに限定し、このvariantのRust実装は試作していない。

### 記録する意味と実行条件の分担

| 意味のrecord（照合対象） | 取得証拠・実行記録（別保存） |
| --- | --- |
| repo/model/revision、内容hash、producer namespaceと意味の版 | 宣言元、何をいつhashしたか、ロードとの結合方法、失敗理由 |
| tokenizer全体のhashと実行意味の版（normalizer、pre-tokenizer、post-processor等を含む） | tokenizers等の依存版、Cargo.lock/source hash、検証に使ったtoken列 |
| pooling、L2の下限、重み/演算/蓄積/出力精度、推論の意味の版 | backend/library版、機種、OS、Metal、実測条件 |
| prefixのbyte列と付加順、外側の空白処理、truncate/EOS、chunk境界/overlap | bucket、batch size、token budget、待機時間、forward shape、計測値 |

依存版はそれだけで再索引を要求するkeyにはしないが、tokenizerや演算の意味を変えたなら意味の版も更新する。
意味が同じと確認できていない更新を、単に「runtime欄だから」と除外してよいわけではない。
record一致はbit単位の同値性を意味しない。shapeや実行環境による数値差は#306/#307の記録を添えて別に評価する。
`inference_semantics`はその変更責任をproducerが持つ。未検証の意味変更はKnownとして発行しない。

## 追加書込みと検索の判定

`Match`は、必要な完全recordに記録された構成がこの照合目的で一致するという意味に限定する。
`Different`は既知の相違（field pathと値）、`Unknown`は欠落・未対応version・取得不成立を返す。
不明同士を一致にしない。構文/未知fieldに問題があるrecordは部分decodeして一致判定せずUnknownへ隔離する。
独立した有効な比較に既知の差と不明が併存した場合はDifferentを優先し、不足理由も残す。許可するのはMatchだけ。

追加書込みは索引に保存したdocument生成spec全体と、新しく生成するoperationのspecを比較する。
検索は保存document specと許可済み`SearchPair.document`、現在query specと`SearchPair.query`をそれぞれ比較する。
pairの両側は共通モデル情報が完全一致しなければならない。role別prefixやchunk/truncateの違いはpairで明示する。
既定のpairを次元やprefix名から推測しない。将来のrurico既知profileまたはconsumerが根拠を確認したpairだけを登録する案である。
今回の`pair()`は候補を組み立てるだけで、登録・承認を代行しない。

| ケース（すべて次元768の例） | 追加書込み | 検索 | consumerで必要になる判断 |
| --- | --- | --- | --- |
| 同一document設定 | Match | そのdocumentに結び付くquery pairならMatch | 設定とproducerを固定して利用 |
| 同次元の別model/revision、weights/config/tokenizer、pooling、precision | Different | pair内の共通情報もDifferent | 同じ索引への混在を止め、別索引生成または旧producer維持 |
| 正しいquery prefix＋truncateとdocument prefix＋chunk | queryをdocumentへ追加すればDifferent | 明示pairと一致ならMatch | role差を許容。quality保証は別 |
| 任意prefixの`embed_text` | documentと違えばDifferent | 未登録ならDifferentまたはpair不足のUnknown | `Text` role、prefix、truncateを含むpairを別途評価 |
| documentのchunk方式/overlap、prefix、空白順が変更 | Different | 旧document pairに新documentを対応させればDifferent | 新生成方針を採用するなら別索引で再生成を検討 |
| queryの上限8192→4096だけ変更 | document specは不変なのでMatch | 旧pairではDifferent、新queryを許可したpairではMatch | query品質を検証してpair更新。保存documentの再生成は一律には不要 |
| 旧fixture/旧索引に識別記録がない | Unknown | Unknown | legacy移行方針を管理者が選択 |

違う構成が短い入力で偶然同じvectorを返しても構成はDifferentである。
Differentは自動的な再生成命令ではない。新しいdocument構成へ移る場合に再生成を判断し、宣言や整形だけの差である根拠が揃う場合も、管理者が明示的な移行として扱う。
逆にrecordが同じでも破損vector、NaN、producerの虚偽、tokenizerのバグ、数値非決定性、品質低下は検出しない。

### 末尾空白の選択

現行維持を今回の提案の前提とする。#307では同一token/shapeのembedding 27行・reranker 16行が診断基準内だったが、
Sentence Transformers 5.1.1のwrapperがprefix込みの文字列全体を`strip()`するため3入力でembedding差を観測した。
数値の正本は[#307の同一token比較とwrapper差](../issue-307/report.md#公開wrapperとの相違は末尾空白)と
[比較JSON](../issue-307/results/numerical-rechecked/comparison-embedding.json)。再測定ではない。

| 選択 | 記録と索引への影響 | 採用前の検証 |
| --- | --- | --- |
| ruricoの現行維持 | `outer_whitespace=none/v1`。既存由来が分かるdocumentの生成方針を変えない | 公式wrapperと入力前処理が違うことをconsumerへ説明 |
| 公式wrapperへ合わせる | `strip-after-prefix`の厳密な文字集合/版を識別。旧documentとの構成はDifferent | consumerの空入力・全空白・Unicode空白・任意prefix・上限/境界chunk・保存vectorで影響を確認し、別索引と検索品質を比較 |
| 本文だけtrim | `trim-body-before-prefix`として別構成。公式と同一とはしない | prefix境界を含むtoken差と、query/document各側の影響を確認 |

例えばprefixが`"検索文書: "`、本文が`""`なら本文trim後の連結は末尾spaceを残し、全体stripは残さない。
prefixが非空白で始まる場合、本文先頭spaceは全体stripでは内部spaceとなる。
どの空白を除去するかも言語/runtimeで異なり得るため、単なる`trim=true`にまとめない。
長文documentにはoffsetで切り出す処理もあるため、stripをどの段階に置くかをalgorithm版と結び付ける。
参照例の`strip-after-prefix/python-3.12/v1`は、#307が使ったPython 3.12の`str.strip()`を連結後に適用する仮の意味IDである。
`trim-body-before-prefix/v1`は同じ文字集合で本文だけを処理する比較候補とする。どちらもPython例では識別だけを行い、strip自体は実装しない。
`offset-retokenize-shrink/v1`は上記の開始commitのplanner、`truncate-eos/v1`はそのtruncate/EOS処理を指す提案ID。
`fixed-token-chunks/v1`は再tokenizeせずtoken位置で切る対照案であり、現行方式と同じとしない。

#307の再利用条件は、固定ruri-v3-310m revision、FP32、同じtoken/shape、公開合成入力、記録された実装・環境に限る。
27行/16行は独立反復の回数ではない。記録の内容hashを現在ロードしたモデルのhashへ流用しない。
#307の限られた品質評価もconsumer全体や新前処理の互換性を保証しない。今回stripは実装しない。

## FTSの構成記録

`FtsIndexSpec`は既存`QueryNormalizationConfig`の3 boolと、順序・文字集合を含む意味の版を保持する。
consumerが所有するtokenizer名、options、その解釈の版、追加前処理と適用順を必須にする。
例の`fts.json`は`identity-before-rurico-normalize/v1`の後、既存正規化、consumer tokenizerの順。
実consumerがtrigramであることやそのoptionsをruricoから推測しない。未取得ならUnknown。
同じ文字列のoptionsでもconsumerの意味の版が違えばDifferentとする。

索引とqueryの両側に同じ前処理契約を渡して照合する。runtime全ONも、明示全OFFも完全な構成として表せる。
過去baselineの欠落時だけ全OFFとする既存契約を維持し、新しい記録の欠落にはdefaultを設けない。
serdeが既存configの未知fieldを捨て得る場合も、新record readerはその前にnested keyを検査する。

phrase、短語展開、vocab参照名、query plan等は別の`FtsSearchPolicy`としてconsumerが記録する。
この方針だけの変更で索引recordを変えない。ただし、vocabが実索引に対応しているかやphrase/短語の検索意味は別の利用側検証が要る。
query policyを新recordの未知fieldとして混ぜたらUnknownとなり、黙って捨てない。
query側のNFKCだけOFFにすることは検索時方針の変更ではなく、対称性を破るDifferentになる。
検索意味・wire-formatの採否は[#314](https://github.com/thkt/rurico/issues/314)の範囲で、`MatchFtsQuery`を変更しない。

## 保存・fingerprint・version

保存は閉じたJSON recordを推奨する。schemaは形、意味のIDは処理内容、fingerprint prefixは符号化/hash方式の版であり、役割を混ぜない。
本例の完全recordは[`document.json`](document.json)と[`fts.json`](fts.json)。JSONの余白、key順、保存先pathはidentityに含まれない。
producer世代、取得時刻、ファイルpath、runtime計測は別の証拠recordへ保存し、意味のrecordに未知keyとして足さない。

| fingerprint候補 | 比較結果 |
| --- | --- |
| Rust `Hash`/`DefaultHasher`やJSON文字列そのまま | process/実装やmap順・書式に依存する契約になりやすく、保存用には見送る |
| 完全recordのcanonical bytesをそのまま比較 | 衝突がなく理由を返せる。これを照合の正本とするが、参照IDとしては長い |
| version付きcanonical bytesのSHA-256 | 決定的で小さいID。完全recordも保存し、hashのみの一致を許可根拠にしない。推奨 |

小さな参照実装ではJSON文字列の正規化方式に依存しないtag付きbyte列を使う。
UTF-8 stringは`s<byte長>:<内容>`、非負u64整数は`i<十進>;`、boolは`t`/`f`、mapは`m<件数>:`とkey/value列。
map keyはUTF-8 bytesで昇順。数値表記に先頭0を付けず、string内容にはUnicode正規化・trimを適用しない。
例の`{"b":"検索 ","a":1}`は`m2:s1:ai1;s1:bs7:検索 `となる（末尾spaceもhash対象）。
配列/float/負数/孤立surrogateはこのschemaでは不要なので拒否する。nullはlegacyの転送表現`n`にできるがfingerprintを発行しない。
hash入力は`b"rurico/spec-example/1\0" + canonical(record)`、表示は`spec-example-1:sha256:<hex>`。
これは検証用の方式であり、製品採用時には別言語実装の共通vectorと保存期間のversion運用も合意する。

`{"schema":2,...}`は未対応version、`{"schema":1,...,"future":true}`は未知field、必須の`tokenizer_sha256`欠落は情報不足。
いずれもUnknownとなり、fingerprintを出さない。重複JSON key、不正な型や数値も拒否する。
通常のserde decodeで未知fieldを捨ててからhashする手順は採らない。
共通情報の名前空間付き意味IDはproducerが所有する不変なopaque IDとしてbyte比較する。readerが中身の効果を理解したという保証ではない。
参照例で解釈するoperation enumとFTS正規化版には既知値の集合があり、新しい値はUnknown。
将来schema/enumを増やすときは明示変換で未知の意味を保持できる場合だけ移行し、元recordを保全する。

## consumerの最小利用例と移行

以下は参照例の関数を使う疑似コードで、DBスキーマや自動migrationを提案するものではない。
`require_match`はMatch以外なら処理を止め、Comparisonの相違/不足を呼出元へ返す。
`producer.lease()`はdescriptor取得から生成・書込みまで設定を固定するconsumer側の仕組みを表す。

```python
# 新索引: producerに対応するdescriptorを同じleaseで取得する。
with producer.lease() as lease:
    snapshot = lease.describe_document()
    require_match(acquire(snapshot, lease.generation))
    document_spec = snapshot.spec
    # consumerが選んだ保存先へJSON全体を保存する。hash単独にしない。
    metadata = {"spec": document_spec, "id": fingerprint(document_spec)}
    consumer.create_index_and_metadata_atomically(metadata)

# 追加: recordはstrictに読み、保存hashも再計算する。
with producer.lease() as lease:
    stored = read(consumer.load_spec_json())
    require_equal(consumer.load_spec_id(), fingerprint(stored))
    now = lease.describe_document()
    require_match(acquire(now, lease.generation))
    require_match(append(stored, now.spec))
    consumer.append_atomically(lease.embed_documents(texts))

# 検索: pairはconsumerが根拠を確認して登録したものを読む。
with producer.lease() as lease:
    stored = read(consumer.load_spec_json())
    require_equal(consumer.load_spec_id(), fingerprint(stored))
    current = lease.describe_query()
    require_match(acquire(current, lease.generation))
    require_match(search(stored, current.spec, consumer.approved_pair()))
    hits = consumer.search(lease.embed_query(query_text))
```

既存ruricoにはこのlease/descriptor APIはない。設定を不変インスタンスに閉じる方法、consumer側でlockする方法が候補となる。
Snapshot取得後のproducer差し替え・設定変更・索引世代の切替で古いMatchを流用しない責任はconsumerにある。
世代番号はprocess間fingerprintの一部にはしない。比較と実行を同じleaseに置けなければ取得不能として止める。
索引metadataとvectorのatomicity、複数writer、検索中の索引切替はconsumerのDB設計で保証する。
関数`acquire`の合成世代検査だけではTOCTOUを解決しない。

FTSでは索引作成前にconsumer設定を含む`FtsIndexSpec`を保存し、追加時に現在の索引側spec、検索前にquery側の索引前処理契約と`fts`で照合する。
query plan等の検索時方針の更新は、その別記録と既存検索検証で扱う。

legacyは「識別記録なし」として開き、Unknownを通常gateで許可しない。管理者は次から明示的に選ぶ。

1. 生成時source/model内容/前処理の証拠が揃う場合、対象索引/fixtureへの対応を確認して構成を移行記録として付与する。現在のload hashを過去生成の証拠にせず、根拠と承認を残す。
2. 由来を復元できなければ別索引へ再生成し、consumerの品質・件数・切替/rollbackを検証する。旧索引は保持する。
3. 旧構成を復元して従来経路を継続する。証拠不足が残る場合はUnknownの例外運用であることを明示し、通常のMatchへ偽装しない。

どれも今回自動実行しない。取得不能時の自動削除、再索引、現行IDの付与は行わない。
[#301](https://github.com/thkt/rurico/issues/301)はvectorの次元・有限性、#315は構成を識別する。
[#304](https://github.com/thkt/rurico/issues/304)とはmodel/tokenizer/config hash、操作条件、精度、生成sourceの項目を共有する案で、
fixture側に別の意味定義を作らない。#304には安全な読込み上限、入力/workload hash、保存形式、生成結果との対応が残る。
既存fixtureには外部sidecarを将来追加できるが、今回その形式を採用しない。#304の読込み修正を待たせない。
#308/#309の性能比較、#321のmodel/precision比較もこのAPIの採用待ちにはしない。

## 検証結果と引き継ぎ

2026-09-27、Python 3.14.7で[READMEのコマンド](README.md)を実行し、6つのテスト群が成功した。
これは公開合成recordに対する設計例の検証であり、Rustの公開API試作・実モデル推論ではない。
同じrecordのmap順を替え、別process・作業ディレクトリ・Python hash seedでも同じIDを得た。
保存JSONのファイル名も替え、再読込み後の一致を確認した。
既知のcanonical byte列との照合に加え、同次元のmodel/revision・tokenizer・prefix・空白順・chunking・pooling・precisionを変えた例で相違を確認した。
queryだけの変更では既存pairを拒否し、documentを保持した新pairで検索を許す例も確認した。

新しい検証が防ぐのは、次元だけでの誤受理、role差の過剰拒否、unknown/defaultの誤用、未知field脱落、
保存順やprocessに依存したID、取得失敗・古い世代の成功扱いである。
既存のplanner・pooling・fixture・FTS正規化テストはこの新しいmetadata比較を扱わないため、小さな純粋関数の検証を追加した。
意味の差は表形式の変更例へまとめ、process起動は決定性を確認する2回だけにした。GPU・network・時間閾値は不要である。
実loaderの不変snapshot、並行producer、DB atomicity、未知producerの真実性は模擬値では検証できず、採用実装で検証する必要がある。
一時コピーで「次元だけで追加を許可」「未知fieldを無視」の2つの誤実装を入れると、対応する検証がいずれもassertionで失敗した。
元ファイルを変更せず、import失敗を検出力の証拠には数えていない。

既存テストの追加複製・削除・統合はなく、失われる既存の検出条件はない。
数値・tokenization・FTS処理の再実装テストを追加せず、既存の検証と#307の限定した観測を参照する。
型検査を済ませた製品API、全言語間のhash互換、性能・保守時間の改善、検索品質向上は未確認。

初回作成時の`cargo fetch --locked`は終了0。標準`bash scripts/check.sh`、変更文書を含む既存の独立評価、
同じheadの`test`・`coverage`・`security`・`zizmor`確認はホストの工程へ引き継ぐ。
この時点ではそれらの成功、公開PR、ユーザーによる採用を主張しない。撮影を要する成果物はない。
製品API・前処理・推論・FTS検索意味・fixture・accepted ADR・wiki・Cargo.lockは変更していない。
採用判断では、記録＋照合API、取得時snapshotの費用、pairの登録責任、保存version運用を確認し、consumerごとの移行は別途合意する。

### 入力型と重複検査の修正後の検証

2026-09-27、保存JSONの`kind`がobjectだと、`fingerprint`が型検査前の集合検索で`TypeError`を漏らすことを再現した。
非文字列の`kind`は、理由`kind: wrong type`を持つ`Incomplete`で拒否するよう修正した。
また、fingerprintは検証時のcanonical bytesをhashにも使い、検索はpair内で検査済みの子recordを同じ呼出し内で再検査しない構造にした。
保存documentと現在queryは引き続き独立して検査する。別呼出し・producer世代を跨ぐcacheは設けていない。

Python 3.14.7で[同じ検証コマンド](README.md)を再実行し、8つのテスト群が成功した。
既存の不明入力検査に、JSON読込みからfingerprintまでの所定例外・理由の確認を追加した。
新しい検査は重複走査の再発と、検査共有による未知field/version・数値範囲・role制約の見落としを防ぐ。
これらは既存の結果比較だけでは検出できないため追加し、入力境界の変更例は一つの表にまとめた。
呼出し回数の検査は内部関数名に依存する保守費用があるが、時間閾値や追加processを使わず、今回確認した重複を直接検出する。
既存検査は削除せず、失う検出条件はない。変更前の例外漏れ・重複回数で回帰検査が失敗することも確認した。

修正前のコピーと修正後を同じ合成入力で比較し、追加・検索・fingerprintの2,718ケースで判定・理由・IDが一致した。
非文字列kindのfingerprintは意図した例外変更なのでこの一致比較から除き、上記の回帰検査で確認した。
`sys.setprofile`による同条件の計数では、fingerprintのdocument符号化は2回から1回、
searchの承認済みdocument/query各々の形検査と符号化は各3回から1回になった。実時間・メモリの改善量は未測定。
開始commit、Cargo.lock、上表の引き継ぎblobに変更はなく、#307の数値を再測定した結果でもない。
修正後の標準checkと同じheadのCIはホストで確認する。実loader・並行producer・DB atomicity・Metal推論については従来の未確認範囲を維持する。
