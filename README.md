# rurico

Apple Silicon (MLX) 上で日本語テキストのembedding・reranking・類似検索を行うためのRustライブラリ。

[cl-nagoya/ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) ファミリー (ModernBERT) をMLX backendで推論する。embedモデルは256〜768次元のembeddingを生成し（モデルサイズにより異なる）、rerankerは検索結果をcross-encoderでスコアリングする。生成したembeddingはSQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) でベクトル検索できる。

## 解決する問題

日本語semantic search CLIを複数構築する際に、embedding生成・reranking・モデル管理・ベクトルストレージが各CLIで重複する。ruricoはこの共通基盤を1 crateに集約し、downstreamのCLIは検索ロジックに集中できるようにする。

## モジュール構成

| モジュール        | 役割                                                                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed`           | embedding 生成（MLX 推論、tokenization、pooling、probe）                                                                                              |
| `reranker`        | 検索結果の reranking（cross-encoder スコアリング、probe）                                                                                             |
| `modernbert`      | ModernBERT モデル定義と config                                                                                                                        |
| `storage`         | SQLite + sqlite-vec のベクトル検索プリミティブ（FTS sanitize / `MatchFtsQuery`、`QueryNormalizationConfig`）                                          |
| `retrieval`       | 5-stage retrieval pipeline contract — `Candidate` / `MergedHit` / `MergeStrategy` / `Aggregator` / `HybridSearchConfig` / `RecencyConfig`             |
| `text`            | テキスト分割（段落 > 行 > 文字境界で UTF-8 安全に分割）                                                                                               |
| `artifacts`       | モデルファイルの型付き検証。downstream は `VerifiedArtifacts<K>` を受け取る（`CandidateArtifacts<K>` は内部 staging）                                 |
| `model_init`      | embed / reranker 共通の初期化エラー型 `ModelInitError`                                                                                                |
| `model_lifecycle` | kind 汎用の `download_model` / `cached_artifacts`（`embed` / `reranker` からも re-export）                                                            |
| `model_probe`     | サブプロセス probe 基盤（`ProbeStatus`、`Embedder::probe` / `Reranker::probe` の実装基盤）                                                            |
| `dispatch`        | top-level probe dispatcher。crate root から `handle_probe_if_needed` を提供し、embed と reranker の probe を一括 wire する                            |
| `sandbox`         | Codex seatbelt 検出（`exit_if_seatbelt` / `require_unsandboxed_mlx_runtime`）。MLX/Metal が abort する環境で smoke / runtime テストを早期に skip する |

検索品質の評価ハーネス（Recall@k / MRR@k / nDCG@k）は [`amici`](https://github.com/thkt/amici) に移譲した。

## 要件

- macOS (Apple Silicon) — MLX backend必須
- Rust 1.96+ (edition 2024)
- Xcode と Metal Toolchain（`xcrun metal --version` が成功すること）

依存は `Cargo.lock` で固定する。ビルド準備とCI相当の検証手順は
[CONTRIBUTING.md](CONTRIBUTING.md#テスト) を参照。

## 使い方

```toml
[dependencies]
rurico = { git = "https://github.com/thkt/rurico", rev = "cf13d32" }
```

`rev` は rurico の commit SHA で固定する。新しい更新を取り込む場合は [rurico の最新 commit](https://github.com/thkt/rurico/commits/main) から sha を確認して上書きする。

MLXの初期化失敗はプロセスをabortする可能性がある。`probe` でモデルのロード可否を子プロセスで検証してからEmbedderを作成する。

```rust
use rurico::embed::{Embed, Embedder, ModelId, download_model};
use rurico::handle_probe_if_needed;
use rurico::model_probe::ProbeStatus;

// main() の冒頭でprobeハンドラを登録
handle_probe_if_needed();

// 未キャッシュならHF Hubからダウンロード
let paths = download_model(ModelId::DEFAULT)?;

match Embedder::probe(&paths)? {
    ProbeStatus::Available => {}
    ProbeStatus::BackendUnavailable => {
        eprintln!("MLX backend not available");
        std::process::exit(1);
    }
}

let embedder = Embedder::new(&paths)?;

// 次元数はモデル依存。MAX_SEQ_LEN超過時は自動truncate。
let query_vec = embedder.embed_query("検索クエリ")?;

// 長文はoverlapping chunksに分割される。
let doc = embedder.embed_document("長いドキュメント...")?;
for chunk_vec in doc.chunks() {
}
```

### 定数

| 定数              | 値               | 用途                                                                               |
| ----------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `EMBEDDING_DIMS`  | `768`            | デフォルトモデル (310m) の出力次元数。実行時は `Embedder::embedding_dims()` で取得 |
| `MAX_SEQ_LEN`     | `8192`           | モデル入力全体長（BOS + prefix + text + EOS）                                      |
| `QUERY_PREFIX`    | `"検索クエリ: "` | クエリ埋め込みときに先頭へ付加                                                     |
| `DOCUMENT_PREFIX` | `"検索文書: "`   | ドキュメント埋め込みときに先頭へ付加                                               |
| `SEMANTIC_PREFIX` | `""`             | semantic/clustering タスク用（プレフィックスなし）                                 |
| `TOPIC_PREFIX`    | `"トピック: "`   | 分類・クラスタリングタスク用                                                       |

### Document Embedding

`embed_document` は `ChunkedEmbedding` を返す。テキストが `MAX_SEQ_LEN` 以内なら `chunks().len() == 1` で従来と同等のembeddingを返す。超過時はprefixを保持したoverlapping chunksに分割され、各chunkが独立したembeddingになる（次元数はモデルにより異なる）。

`embed_documents_batch` は入力件数と同数の `Vec<ChunkedEmbedding>` を返し、入力順を保持する。

custom producerやmockで内容を検証して構築するには、
`ChunkedEmbedding::try_new_validated(chunks)` を使う。chunksと各vectorが非空で、
単一結果内の次元が同じ、全要素が有限という条件を検査する。
`EmbeddingValidationError` は空chunks・空vector・次元不一致・非有限値を区別し、
該当するchunkと要素の位置を0始まりで返す。値と順序、`c0`からのIDを保ち、
ゼロvector・非単位長・負値・有限の大きな値も受理する。正規化や補正は行わない。
既存の `try_new` と `EmptyChunksError` は引き続き利用でき、保証はchunksの非空性に限る。

[ADR-0011](docs/decisions/0011-adopt-uniform-vector-length-contract-for-the-embed-trait.md)の
別document・query・呼出し間でも同じモデル次元を返す契約は、引き続き `Embed` 実装の責任となる。
内容検証だけではモデルの識別や保存索引との互換性を保証しない。
fixtureの `load` は同じ検証をdocumentごとに使い、不正内容を `InvalidData` として拒否する。
document間の次元照合やmodel/index互換性は保証しない。
`save` は互換用のlegacy writerとして維持する。新規生成には由来を明示する`save_versioned`を使う。
`load`はlegacyとversion 1を64 MiB上限で読み、余剰byteも拒否する。
`load_fixture`は上限を指定でき、legacyの由来不明を`generation: None`として保持する。
`compare`は両側の内容を検証し、不正内容・shape差・非有限metricを`CompareError`で返す。
形式・生成条件・呼出し移行は[fixtureの説明](tests/fixtures/phase2_baseline/README.md)を参照。
MLXは [ADR-0002](docs/decisions/0002-gpu-side-pooling-embed.md) の境界に従い、
readback済みbufferの既存走査で共通の有限性検査を行う。構築時の再走査は追加しない。

`dyn Embed`でも `embed_documents_batch_with_options_and_metrics(&texts, &options)` を使うと、
一度の推論の結果（`embeddings`）と利用可能な計測（`metrics: Option<InferenceMetrics>`）を受け取れる。
MLXでは `Duration` の精度でhost側の区間時間、forwardごとのshape、pause回数を返す。
既存implementorのdefault methodはoptions付き処理へ一度だけ委譲し、計測は`None`となる。
optionsは引き続きbest-effort hints。既存の `BatchMetrics` と具象型のmetrics APIも利用できるが、
時間は整数msでtokenization未分離をゼロ表示するため、新しい計測には上記APIを使う。
区間の包含関係は `InferenceMetrics` のrustdoc、記録手順は
[CONTRIBUTING](CONTRIBUTING.md#options付き推論の計測issue-306) を参照。


`embed_text` はプレフィックスを明示指定して埋め込む低レベルAPI（chunkingなし、超過時はtruncate）。検索用途では `embed_query` / `embed_document` を使う。

```rust
use rurico::embed::{TOPIC_PREFIX, SEMANTIC_PREFIX};

let topic_vec = embedder.embed_text("ニュース記事...", TOPIC_PREFIX)?;
let semantic_vec = embedder.embed_text("任意テキスト", SEMANTIC_PREFIX)?;
```

### モデルキャッシュの確認

ネットワークアクセスなしでモデルがローカルにあるか確認できる。

```rust
use rurico::embed::{ModelId, cached_artifacts};

if let Some(artifacts) = cached_artifacts(ModelId::DEFAULT)? {
    // Embedder::newに渡せる
} else {
    // ダウンロードが必要
}
```

### probe なしの簡易利用

abortリスクを許容できるスクリプト等ではprobeを省略できる。

```rust
use rurico::embed::{Embed, Embedder, ModelId, download_model};

let artifacts = download_model(ModelId::DEFAULT)?;
let embedder = Embedder::new(&artifacts)?;
let vector = embedder.embed_query("検索クエリ")?;
```

### storage（ベクトル検索）

sqlite-vecはプロセスレベルのauto-extensionとして登録が必要。`Connection::open` の前に一度だけ呼ぶ。

```rust
use rurico::storage::ensure_sqlite_vec;
use rusqlite::Connection;

ensure_sqlite_vec().expect("sqlite-vec initialization failed");
let conn = Connection::open("my.db")?;
```

### FTS クエリパイプライン

ユーザー入力をFTS5 `MATCH` に安全に渡すには `prepare_match_query` を使う。内部で normalize → sanitize → expand の3段階を経て、最終出力時に各literalを一度だけ引用した `MatchFtsQuery` を返す。sanitizeの中間表現にはFTS構文の引用符を追加しない。

```rust
use rurico::storage::{prepare_match_query, QueryNormalizationConfig, SanitizeError};

let normalization = QueryNormalizationConfig::default(); // 全 step ON（推奨）

match prepare_match_query(&conn, user_input, "fts_chunks_vocab", &normalization) {
    Ok(matched) => {
        // matched.as_str() を MATCH に渡す
        stmt.query_map([matched.as_str()], |row| { /* ... */ })?;
    }
    Err(SanitizeError::EmptyInput) => {
    }
    Err(SanitizeError::NoSearchableTerms) => {
        // NEAR() グループのみ等、検索可能な語がない
    }
    Err(SanitizeError::InvalidVocabTable(name)) => {
        // 呼び出し側のスキーマ設定ミス
        eprintln!("invalid vocab table: {name}");
    }
    Err(SanitizeError::VocabLookupFailed(reason)) => {
        eprintln!("fts vocab lookup failed: {reason}");
    }
}
```

第3引数の `vocab_table` は `fts5vocab` 仮想テーブル名を受け取る。呼び出し側のスキーマ規約に応じて `"fts_chunks_vocab"` や `"messages_vocab"` などを指定する。`row` または `col` 型の vocabulary のみ対応する（`instance` 型には `cnt` カラムが無いため）。値はSQLにエスケープなしで埋め込まれるため、SQL identifier として妥当な文字列（先頭が ASCII 英字または `_`、以降 ASCII 英数字または `_`）のみ許容され、違反した場合は `SanitizeError::InvalidVocabTable` を返す。

第4引数の `normalization` は Phase 5 (#69) で追加。runtime default は NFKC + ASCII lowercase + 連続空白の collapse がすべて ON で、indexing 側 (`docs_fts.body`) と querying 側で folding が一致するよう設計されている。明示的に旧挙動が必要な呼び出しは `pre_phase_5_disabled()` を渡す（pre-#69 のスナップショットも `BaselineSnapshot.normalization` の serde-default としてこの値を参照する）。

`NEAR()` グループ、`^`/`+`/`-` プレフィックス、各語の端の括弧は除去される。コロン・語中のハイフン・入力の引用符はliteralの文字として保持し、最終出力で `"` を `""` にエスケープする。不均衡な引用符を補完したり、入力の引用符からphrase境界を解釈したりはせず、語の区切りは空白のままとする。`AND`/`OR`/`NOT` のようなoperator-like keywordは、前後に非operatorの語がある場合のみliteral termとして引用符で囲まれ、Boolean演算子にはならない。前後が欠けたdangling operator（例: 先頭の `NOT`、NEAR除去後に孤立した `OR`）は除去される。短い語（1-2文字）は指定した vocab テーブルがあればprefix展開されるが、operator-like keywordは展開しない。vocab テーブルが存在しない場合だけはそのまま引用に劣化し、それ以外の SQLite 障害は `SanitizeError::VocabLookupFailed` を返す。展開なしの短語がhitするかはtokenizerに依存し、trigramでは3文字未満のliteralはhitしない。

短語のprefix照会では `%`・`_`・backslashをLIKEのwildcardやescapeとして解釈せず、literalとして扱う。同じ短語の照会結果は1回の `prepare_match_query` 内でだけ再利用し、次の呼出しはDB更新を反映する。長語のみの場合もvocabのidentifier・schemaを検査し、別接続で呼出し前に確定したschema変更によるエラーも省略しない。内部の正規化は変更不要な入力を借用するが、公開の `normalize_for_fts` は引き続き `String` を返す。測定条件と限界は [Issue #312の検証記録](docs/benchmarks/issue-312/README.md) を参照。

amiciの `parse_fts_segments` が読むwire-formatは、固定語の `"..."`（内部引用符は `""`）、展開語群の `("..." OR "...")`、語・語群間の明示的な ` AND ` を維持する。[#297](https://github.com/thkt/rurico/issues/297)では、入力 `rate-limit` の出力を `"""rate-limit"""` から `"rate-limit"` へ修正した。構文形状や公開APIは変えないが、literalの内容と検索結果は修正される。amici側の互換性は依存rev更新時に既存のround-trip testで確認する（[契約の参照元 #249](https://github.com/thkt/rurico/issues/249)）。

phrase・短語上限・型付きquery planの未採用比較は[Issue #314の調査](docs/research/issue-314/README.md)を参照する。現行入力の引用符とORのliteral扱いは変えていない。
固定amici版のparserには内部引用符を途中で区切る欠陥と、引用内の`)`をgroup終端と誤認する欠陥があるため、上記wire形状の維持や既存round-trip成功だけではliteral内容の保持を保証しない。
[consumerへの影響と検証範囲](docs/research/issue-314/report.md#315との分担とconsumerへの影響)に固定版のnative実測で確認した欠落と、未確認の適用範囲を記載している。

### query normalization 単体利用

`prepare_match_query` を経由せずに同じ folding を適用したい場合（例: indexing 側の body 正規化）は `normalize_for_fts` を直接呼ぶ。

```rust
use rurico::storage::{QueryNormalizationConfig, normalize_for_fts, pre_phase_5_disabled};

let config = QueryNormalizationConfig::default();
let folded = normalize_for_fts("ＡＢＣ\u{3000}DEF", &config);
assert_eq!(folded, "abc def");

let nfkc_only = QueryNormalizationConfig {
    nfkc: true,
    ascii_lowercase: false,
    collapse_whitespace: false,
};

// 旧挙動への opt-out
let off = pre_phase_5_disabled();
```

### ハイブリッド検索 RRF — `WeightedRrf`

FTS5 とベクトル検索の結果を Reciprocal Rank Fusion で統合する canonical fusion strategy。weight 調整・recency 加味・複数 source 対応に加え、default config (`rrf_k=60.0`, weight=1.0) でランク位置のみによる折りたたみも担う。詳細な 5-stage pipeline contract は [Retrieval Pipeline](#retrieval-pipeline5-stage-contract) を参照。

```rust
use rurico::retrieval::{Candidate, CandidateSource, MergeStrategy, WeightedRrf};

let candidates = vec![
    Candidate { source: CandidateSource::Fts, doc_id: "1".into(), chunk_id: None, score: 0.9, rank: 0 },
    Candidate { source: CandidateSource::Fts, doc_id: "2".into(), chunk_id: None, score: 0.7, rank: 1 },
    Candidate { source: CandidateSource::Vector, doc_id: "2".into(), chunk_id: None, score: 0.95, rank: 0 },
    Candidate { source: CandidateSource::Vector, doc_id: "4".into(), chunk_id: None, score: 0.8, rank: 1 },
];
let merged = WeightedRrf::default().merge(&candidates);
```

### Retrieval Pipeline（5-stage contract）

`retrieval` モジュールは 5 ステージの pipeline contract を提供する。`storage::prepare_match_query` と組み合わせる際の標準配線で、aggregation hook、hybrid weight/recency（`RecencyConfig` + `merge_with_recency`）、chunk-level retrieval を備える。

| Stage | 入力 → 出力                               | 提供型・関数                                                                                                                                             |
| ----- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `&str` → `Vec<Candidate>`                 | `Candidate { source, doc_id, chunk_id, score, rank }`、`CandidateSource` 閉enum (`Fts` / `Vector`)                                                       |
| 2     | `&[Candidate]` → `Vec<MergedHit>`         | `MergeStrategy` trait、default impl `WeightedRrf`、設定 `HybridSearchConfig { rrf_k, source_weights }`、`merge_with_recency` + `RecencyConfig`           |
| 3     | `&[MergedHit]` → `Vec<MergedHit>`         | `Aggregator` trait + 4 impl (`IdentityAggregator` / `MaxChunkAggregator` / `DedupeAggregator` / `TopKAverageAggregator { k }`)、`group_by_parent` helper |
| 4     | `(&str, &[MergedHit], corpus)` → 並べ替え | 既存の `Rerank` trait に corpus lookup を組み合わせて呼ぶ                                                                                                |
| 5     | rerank 結果 → top_k                       | downstream の表示層が責任を持つ                                                                                                                          |

```rust
use std::collections::HashMap;
use rurico::retrieval::{
    Aggregator, Candidate, CandidateSource, HybridSearchConfig, MaxChunkAggregator,
    MergeStrategy, RecencyConfig, WeightedRrf,
};

// Stage 1: 呼び出し側が FTS / vector の各 source から Candidate を集める
let candidates: Vec<Candidate> = /* ... */;

let mut weights = HashMap::new();
weights.insert(CandidateSource::Fts, 0.6);
weights.insert(CandidateSource::Vector, 1.4);
let merger = WeightedRrf::new(HybridSearchConfig {
    rrf_k: 60.0,
    source_weights: weights,
});
let merged = merger.merge(&candidates);

// age_lookupは呼び出し側のcorpus schemaに依存
let recency = RecencyConfig { weight: 0.3, half_life_days: 30.0 };
let merged_with_recency = merger.merge_with_recency(
    &candidates,
    &recency,
    |doc_id| { /* Option<f64>: 経過日数を返す。None なら recency を skip */ None },
);

// Stage 3: chunk-level retrieval なら parent に集約
let aggregator = MaxChunkAggregator;
let aggregated = aggregator.aggregate(&merged);

```

#### Aggregator の使い分け

`MergedHit.chunk_id` が `Some(_)` のとき（chunk-level retrieval）に意味のある集約を提供する。`None` のままで Stage 2 が parent ごとに一件を返し、整列済みの有限スコアを渡す場合、`k > 0` の各 aggregator は identity と等価になる。任意の未整列入力や `k = 0` にはこの説明を適用しない。

| Aggregator                    | 振る舞い                                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| `IdentityAggregator`          | パススルー。chunk-level identity を Stage 4 に届ける。default                          |
| `MaxChunkAggregator`          | parent ごとに最高スコアの chunk を残し、`chunk_id = None` で parent 単位に折りたたむ   |
| `DedupeAggregator`            | parent ごとに先頭 1 件のみ残す（順序保持の dedupe）                                    |
| `TopKAverageAggregator { k }` | parent ごとに上位 `k` chunk スコアの平均を採用（`TopKAverageAggregator::new(k)` も可） |

Stage 3 の標準入力は、Stage 2 が返す有限な score と source contribution を持つ、score 降順・`doc_id` 昇順・`chunk_id` 昇順の結果である。
Identity と Dedupe は入力順を保持するため、降順出力の保証はこの整列済み入力を前提とする。
任意の未整列入力を渡しても整列せず、Dedupe は最初の hit を選び、後続の高スコアに置換しない。

MaxChunk は最高スコアが同点なら最初の chunk の score と source map を残す。
TopKAverage は parent 内を score の `total_cmp` 降順で安定整列し、同点の選択境界でも入力順を保つ。
選択 chunk の source contribution だけを平均し、欠落 source はゼロとして数える。
MaxChunk と TopKAverage の出力は `chunk_id = None` の parent 単位で score 降順・`doc_id` 昇順になる。

重複候補の契約案と内部最適化の未採用比較は [Issue #313 の調査](docs/research/issue-313/README.md) を参照。

独自の `Aggregator` を実装する場合は `group_by_parent(&merged) -> HashMap<&str, Vec<&MergedHit>>` で parent 単位にバケットできる。

#### スコア計算の有限性と失敗時の扱い

`WeightedRrf` は重みがゼロ・非有限、分母が非正・非有限、または除算結果が非有限となる候補の寄与をスキップする。
有限の寄与を足した結果、合計スコアか source ごとの総和が overflow する場合は、その `(doc_id, chunk_id)` の hit 全体を除外する。
一部の総和だけを返したり、上限値に丸めたりはしない。負の有限重みも利用でき、分母が正であれば負の `rrf_k` も利用できる。
`merge_with_recency` は boost 自体または加算後のスコアが非有限になる場合、その boost をスキップして元の RRF スコアを保つ。
source contribution は recency を含まない。並び順はスコア降順、`doc_id` 昇順、`chunk_id` 昇順を維持する。

`TopKAverageAggregator` は通常の総和が overflow する場合に計算を組み替え、`f64::MAX` 2 件の平均も有限の `f64::MAX` として返す。
負値を含むスコアと source contribution を同様に平均し、選択した hit にない source はゼロとして件数に含める。
上位 `k` 件に非有限のスコアまたは source contribution が含まれる parent は除外する。`k = 0` と空入力は空の結果を返す。
これらの失敗は `Result` エラーではなく、寄与・hit の除外または boost のスキップとして扱う。

### ベクトルのバイト変換

`sqlite-vec` にベクトルをバインドする際は `bytemuck::cast_slice` で zero-copy に `&[f32] → &[u8]` を行う。rurico は little-endian ターゲットでのみビルドされるため、変換結果は sqlite-vec が期待する byte layout と一致する。

```rust
let vector: Vec<f32> = embedder.embed_query("検索")?;
let bytes: &[u8] = bytemuck::cast_slice(&vector);
stmt.execute(rusqlite::params![bytes])?;
```

### 重みの読込み契約

`Embedder::new`・`Reranker::new` と両者の probe は、MLX の parameter 構築前に
safetensors ヘッダーを共通検証する。config から決まる全必須キーと shape、F32 dtype、
データ範囲を照合し、欠損・不整合・未知キーを拒否する。F16・BF16・整数型は受理しない。
ヘッダー検証は重み本体を読み込まず、MLX が一度読み込んだ配列も実際の parameter 集合へ
過不足なく割り当てる。これにより、未ロードの初期値が推論に残ることを防ぐ。

embedding のキーは `embeddings.`・`layers.`・`final_norm.`、reranker の backbone は
`model.` 配下で、head と classifier は直下に置く。名前の変換や共有重みの別名による
欠損補完は行わない。`__metadata__` は tensor として扱わない。先頭層の attention norm は
Identity 相当で重みを持たず、RoPE の非永続 buffer や masked-LM の共有 decoder は
対象モデルの必須重みに含まれない。これらの tensor キーがファイルにあれば未知キーとして拒否する。

`attention_bias`・`mlp_bias`・`norm_bias`・`classifier_bias` は、対応する公式モデルに合わせて
`false` のみ対応し、config JSON での省略時も `false` とする。`true` は config エラーになる。
reranker の `head.dense.bias`・`head.norm.bias` は parameter として作らず、
最終出力の `classifier.bias` は別の必須重みとして維持する。
[Issue #300 の合意](https://github.com/thkt/rurico/issues/300)により、不要なランダム bias の
除去に伴う reranker の logit・score・順位の変化は許容する。embedding の既存数値基準は維持する。
検索品質の改善を示す変更ではなく、公式実装全体との比較・品質評価は
[#307](https://github.com/thkt/rurico/issues/307)で扱う。

`VerifiedArtifacts` はファイル・config・tokenizer・モデル種別の確認結果であり、全重みの
検証完了を意味しない。重み検証は load 時に毎回行う。ファイルの暗号学的真正性、tensor 値の
全要素の有限性、外部プロセスによる同時書換えに対する一貫したスナップショットは保証しない。
検証方法と測定条件は [CONTRIBUTING](CONTRIBUTING.md#重みの読込み検証) を参照。

### エラー型

エラー型はフェーズごとに分離されている。公開 error enum は
`#[non_exhaustive]` のため、downstream の `match` では wildcard arm を
置く。

**`ArtifactError`** — モデルファイルの取得・検証フェーズ

| variant            | 発生条件                                               |
| ------------------ | ------------------------------------------------------ |
| `MissingFile`      | 重みファイル / config / tokenizer が存在しない         |
| `InvalidConfig`    | config.json の読み込み・パース・設定検証失敗            |
| `InvalidTokenizer` | tokenizer.json のロード失敗                            |
| `WrongModelKind`   | safetensors のテンソルキーが期待するモデル種別と不一致 |
| `DownloadFailed`   | HF Hub からのダウンロード失敗                          |

**`ModelInitError`** — `Embedder::new` / `Embedder::probe` / `Reranker::new` / `Reranker::probe` フェーズ（embed と reranker で共通）

| variant        | 発生条件                                                   |
| -------------- | ---------------------------------------------------------- |
| `Backend`      | MLX バックエンド初期化・重みロード・probe サブプロセス失敗 |
| `ModelCorrupt` | probe 子プロセスがモデル読込み失敗を報告した               |

`Backend` は `message: String` と `source: Option<Box<dyn Error + Send + Sync>>` を持ち、`std::error::Error::source()` で原因チェーンを辿れる。
重み契約の違反は、直接の `new` では `Backend`、probe では `ModelCorrupt` として返る。
診断には `weights: missing key`・`shape`・`dtype`・`unknown key`・`header`・`offsets` と
該当キーや理由を含む。メッセージ全文は安定 API ではない。

**`EmbedError`** — `Embed` トレイトメソッド（推論）フェーズ

| variant               | 発生条件                                    |
| --------------------- | ------------------------------------------- |
| `EmptySequence`       | モデルが seq_len=0 の出力を返した           |
| `BufferShapeMismatch` | 推論出力のバッファサイズが期待値と不一致    |
| `EmptyChunks`         | chunked embedding が chunk なしで構築された |
| `Inference`           | MLX 推論失敗                                |
| `Tokenizer`           | tokenizer エンコード失敗                    |
| `NonFiniteOutput`     | embedding 出力に NaN または Inf が含まれる  |

`Inference` / `Tokenizer` は `message: String` と
`source: Option<Box<dyn Error + Send + Sync>>` を持ち、
`std::error::Error::source()` で原因チェーンを辿れる。

**`RerankerError`** — `Rerank` トレイトメソッド（reranker 推論）フェーズ

| variant           | 発生条件                                                            |
| ----------------- | ------------------------------------------------------------------- |
| `Inference`       | MLX forward/eval失敗、出力形状不一致、lockのpoison                   |
| `Tokenizer`       | tokenizer エンコード失敗                                            |
| `NonFiniteOutput` | sigmoid変換前のlogitに NaN / +Inf / -Inf が含まれる                 |
| `InitFailed`      | `LazyReranker` 初回呼び出し時の初期化失敗（cache 参照・ロード・DL） |

`Inference` / `Tokenizer` は `message: String` と
`source: Option<Box<dyn Error + Send + Sync>>` を持つ。
`InitFailed` は同じ名前のフィールドを持ち、キャッシュした原因を複数回の失敗で共有するため
`source` は `Option<Arc<dyn Error + Send + Sync>>` となる。
`std::error::Error::source()` でMLX・tokenizer・初期化の原因を辿れる。
`Inference` の形状不一致やlockのpoisonは、元の原因を保持せず `source: None` とする。
表示の接頭辞と失敗分類は維持するが、backendのメッセージ全文は安定APIではない。

既存の `LazyReranker::new` は `Result<R, String>` を返すclosureを引き続き受け取る。
この場合に保持できるのは文字列だけであり、呼出側で既に文字列化した元の原因は復元できない。
型付きの原因を残すには `LazyReranker::with_error` を使い、closureから元のエラーを返す。
cache取得・download・loadの異なるエラーをまとめる場合は
`Result<Reranker, Box<dyn std::error::Error + Send + Sync>>` として `?` で伝播できる。

有限logitのscoreは従来のf32 sigmoidのままで、極端な有限値では0または1へ丸められる。
同点は入力indexの昇順とし、raw logitでは並べ替えない。
公開variantの移行方法は [CHANGELOG](CHANGELOG.md)、少数例の飽和確認とconsumerの照合範囲は
[Issue #302の確認記録](docs/benchmarks/issue-302-reranker-errors.md) を参照。

公開APIの失敗契約はrustdocの `# Errors` に記載する。repoの運用ルールは
[`docs/errors.md`](docs/errors.md) を参照。

### ログ出力

内部の警告は `tracing` crate 経由で `warn` レベルで出力される。`tracing_subscriber` を初期化し、`RUST_LOG=rurico=warn` または EnvFilter に `rurico=warn` directive を含めることで観測できる（amici を使う CLI は `amici::logging::init_subscriber` 経由で自動的に観測される）。

### Codex seatbelt 検出

Codex Desktop の seatbelt sandbox では MLX / Metal 初期化が abort する。MLX を駆動する downstream は `sandbox` モジュールでこの環境を早期に検出して skip / panic できる。

```rust
use rurico::sandbox::{exit_if_seatbelt, require_unsandboxed_mlx_runtime};

// smoke binary は SEATBELT_SKIP_EXIT (78, BSD EX_CONFIG) で抜ける
exit_if_seatbelt(env!("CARGO_BIN_NAME"));

// MLX ランタイムテストは sandbox 配下で panic させる
require_unsandboxed_mlx_runtime();
```

### テストサポート（downstream 向け）

`test-support` featureを有効にすると、downstream crateのテストで使えるモックが利用できる。

```toml
[dev-dependencies]
rurico = { git = "https://github.com/thkt/rurico", rev = "cf13d32", features = ["test-support"] }
```

| struct                | 振る舞い                                                                             |
| --------------------- | ------------------------------------------------------------------------------------ |
| `MockEmbedder`        | 入力位置ごとに決定的な one-hot ベクトルを返す（バッチ時は入力順 `i`、単発は `0`）    |
| `FailingEmbedder`     | 設定に応じてエラーを返す                                                             |
| `MismatchEmbedder`    | batch で入力より少ないベクトルを返す                                                 |
| `AlternatingEmbedder` | `embed_document` が成功と失敗を交互に返す（初回は失敗）                              |
| `MockChunkedEmbedder` | 指定数の chunk を返す（multi-chunk テスト用）                                        |
| `MockReranker`        | `Rerank` トレイト用。全ペアに固定スコア（default 0.5、`with_score(s)` で指定）を返す |

```rust
use rurico::embed::{Embed, MockEmbedder};

let embedder = MockEmbedder::default();
let v = embedder.embed_query("テスト")?;
assert_eq!(v.len(), 768);
```

## Development

### Setup

Run once after cloning:

```sh
git config --local core.hooksPath .githooks
```

This installs a pre-commit hook that runs `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets --all-features -- -D warnings` before each commit. Violations abort the commit. To skip for one commit: `git commit --no-verify`.

### Common commands

```sh
bash scripts/test.sh                                                   # 通常テスト（smokeのモデル不要テストを含む、doctestを除く）
cargo test --locked --doc --workspace --features test-support,test-mlx,smoke # nextest対象外のdoctest
cargo clippy --workspace --all-targets --all-features -- -D warnings    # lint
cargo fmt --all -- --check                                              # format check
```

## テスト

```sh
bash scripts/check.sh                                                        # 標準check（モデル不要、FFIテストはMetalを使用）
cargo nextest run --workspace --features test-mlx --run-ignored=ignored-only   # MLX ランタイムテスト（通常 Terminal 推奨）
cargo run --bin mlx_smoke --features smoke --release -- verify-fixture         # embed 数値同等性検証（smoke binary）
```

Codex Desktop の `CODEX_SANDBOX=seatbelt` 環境では、MLX / Metal 初期化が abort することがあるため、
smoke binary は `sandbox::exit_if_seatbelt` 経由で `SEATBELT_SKIP_EXIT` (78) で停止し、`test-mlx` は ignored のままにしている。
実検証は通常の Terminal か、sandbox 外の実行環境で行う。

`mlx_smoke` binary は `smoke` Cargo feature でゲートされており、library として rurico を取り込む downstream には `tracing-subscriber` を持ち込まない。harness 用 just recipe（`just embed-verify` / `just embed-baseline` / `just probe-embed` / `just probe-reranker` 等）は `justfile` を参照。

通常CI・checkは `test-support,test-mlx,smoke` を有効にし、実モデルのignoredテストは実行しない。
`--all-features` の意味と実行対象・選択検査は[CONTRIBUTING](CONTRIBUTING.md#テスト)を参照。

性能判定と基準rev比較は[CONTRIBUTING](CONTRIBUTING.md#性能判定と基準rev比較issue-359)を参照。
`measure-baseline` のprimary成功はW1/W3の速度や全workloadの目標達成を保証しない。
`compare-records BASE CURRENT` は同条件のraw recordから、batch/sequential効率と版間の遅延変化を別に表示する。

rerankerのモデル不要検証・ignoredの実モデル検証・50ペアの基準版benchmark比較は
[CONTRIBUTING](CONTRIBUTING.md#重複検証の整理とreranker遅延の計測issue-364)を参照。
benchmarkは性能を測定し、標準checkの合否や検索品質評価とは分けて扱う。

検索品質の評価（Recall@k / MRR@k / nDCG@k）は [`amici`](https://github.com/thkt/amici) で行う。`CandidateSource` は `{ Fts, Vector }` の閉 enum に固定されている（prefix-ensemble は採用していない）。

## ライセンス

MIT
