# rurico

Apple Silicon (MLX) 上で日本語テキストのembedding・reranking・類似検索を行うためのRustライブラリ。

[cl-nagoya/ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) ファミリー (ModernBERT) をMLX backendで推論する。embedモデルは256〜768次元のembeddingを生成し（モデルサイズにより異なる）、rerankerは検索結果をcross-encoderでスコアリングする。生成したembeddingはSQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) でベクトル検索できる。

## 解決する問題

日本語semantic search CLIを複数構築する際に、embedding生成・reranking・モデル管理・ベクトルストレージが各CLIで重複する。ruricoはこの共通基盤を1 crateに集約し、downstreamのCLIは検索ロジックに集中できるようにする。

## モジュール構成

| モジュール        | 役割                                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed`           | embedding 生成（MLX 推論、tokenization、pooling、probe）                                                                                                    |
| `reranker`        | 検索結果の reranking（cross-encoder スコアリング、probe）                                                                                                   |
| `modernbert`      | ModernBERT モデル定義と config                                                                                                                              |
| `storage`         | SQLite + sqlite-vec のベクトル検索プリミティブ（FTS sanitize / `MatchFtsQuery`、`QueryNormalizationConfig`）                                                |
| `retrieval`       | 5-stage retrieval pipeline contract — `Candidate` / `MergedHit` / `MergeStrategy` / `Aggregator` / `HybridSearchConfig` / `RecencyConfig`                   |
| `text`            | テキスト分割（段落 > 行 > 文字境界で UTF-8 安全に分割）                                                                                                     |
| `artifacts`       | モデルファイルの型付き検証パイプライン（`CandidateArtifacts<K>` → `VerifiedArtifacts<K>`）                                                                  |
| `model_init`      | embed / reranker 共通の初期化エラー型 `ModelInitError`                                                                                                      |
| `model_lifecycle` | kind 汎用の `download_model` / `cached_artifacts`（`embed` / `reranker` からも re-export）                                                                  |
| `model_probe`     | サブプロセス probe 基盤（`ProbeStatus`、`Embedder::probe` / `Reranker::probe` の実装基盤）                                                                  |
| `dispatch`        | top-level probe dispatcher。crate root から `handle_probe_if_needed` を提供し、embed と reranker の probe を一括 wire する                                  |
| `sandbox`         | Codex seatbelt 検出（`exit_if_seatbelt` / `require_unsandboxed_mlx_runtime`）。MLX/Metal が abort する環境で smoke / runtime テストを早期に skip する        |

検索品質の評価ハーネス（Recall@k / MRR@k / nDCG@k）は [`amici`](https://github.com/thkt/amici) に移譲した。

## 要件

- macOS (Apple Silicon) — MLX backend必須
- Rust 1.95+ (edition 2024)

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

// probe 子プロセスのハンドラ登録（main() の冒頭で呼ぶ）
// crate root の re-export。実体は `rurico::dispatch` で embed と reranker をまとめて wire する。
handle_probe_if_needed();

// モデルをダウンロード（初回のみ、HF Hub にキャッシュ）
let paths = download_model(ModelId::default())?;

// probe でモデルのロード可否を事前検証
match Embedder::probe(&paths)? {
    ProbeStatus::Available => {}
    ProbeStatus::BackendUnavailable => {
        eprintln!("MLX backend not available");
        std::process::exit(1);
    }
}

// Embedder を作成して embedding を生成
let embedder = Embedder::new(&paths)?;

// クエリ: Vec<f32> (次元数はモデルにより異なる。デフォルト 768)。MAX_SEQ_LEN 超過時は自動 truncate。
let query_vec = embedder.embed_query("検索クエリ")?;

// ドキュメント: ChunkedEmbedding。短文は chunks.len()==1、
// 長文は overlapping chunks に分割される。
let doc = embedder.embed_document("長いドキュメント...")?;
for chunk_vec in &doc.chunks {
    // chunk_vec: &Vec<f32> (次元数はモデルにより異なる)
}
```

### 定数

| 定数                   | 値               | 用途                                                                               |
| ---------------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `EMBEDDING_DIMS`       | `768`            | デフォルトモデル (310m) の出力次元数。実行時は `Embedder::embedding_dims()` で取得 |
| `MAX_SEQ_LEN`          | `8192`           | モデル入力全体長（BOS + prefix + text + EOS）                                      |
| `QUERY_PREFIX`         | `"検索クエリ: "` | クエリ埋め込みときに先頭へ付加                                                     |
| `DOCUMENT_PREFIX`      | `"検索文書: "`   | ドキュメント埋め込みときに先頭へ付加                                               |
| `SEMANTIC_PREFIX`      | `""`             | semantic/clustering タスク用（プレフィックスなし）                                 |
| `TOPIC_PREFIX`         | `"トピック: "`   | 分類・クラスタリングタスク用                                                       |

### Document Embedding

`embed_document` は `ChunkedEmbedding` を返す。テキストが `MAX_SEQ_LEN` 以内なら `chunks.len() == 1` で従来と同等のembeddingを返す。超過時はprefixを保持したoverlapping chunksに分割され、各chunkが独立したembeddingになる（次元数はモデルにより異なる）。

`embed_documents_batch` は入力件数と同数の `Vec<ChunkedEmbedding>` を返し、入力順を保持する。

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

if let Some(artifacts) = cached_artifacts(ModelId::default())? {
    // キャッシュ済み — そのまま Embedder::new に渡せる
} else {
    // 未ダウンロード
}
```

### probe なしの簡易利用

abortリスクを許容できるスクリプト等ではprobeを省略できる。

```rust
use rurico::embed::{Embed, Embedder, ModelId, download_model};

let artifacts = download_model(ModelId::default())?;
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
// conn で vec0 仮想テーブルが利用可能
```

### FTS クエリパイプライン

ユーザー入力をFTS5 `MATCH` に安全に渡すには `prepare_match_query` を使う。内部で normalize → sanitize → expand の3段階を経て、全トークンを引用符で囲んだ `MatchFtsQuery` を返す。

```rust
use rurico::storage::{prepare_match_query, QueryNormalizationConfig, SanitizeError};

let normalization = QueryNormalizationConfig::default(); // 全 step ON（推奨）

match prepare_match_query(&conn, user_input, "fts_chunks_vocab", &normalization) {
    Ok(matched) => {
        // matched.as_str() を MATCH に渡す
        stmt.query_map([matched.as_str()], |row| { /* ... */ })?;
    }
    Err(SanitizeError::EmptyInput) => {
        // 空のクエリ
    }
    Err(SanitizeError::NoSearchableTerms) => {
        // NEAR() グループのみ等、検索可能な語がない
    }
    Err(SanitizeError::InvalidVocabTable(name)) => {
        // 呼び出し側のスキーマ設定ミス
        eprintln!("invalid vocab table: {name}");
    }
    Err(SanitizeError::VocabLookupFailed(reason)) => {
        // vocab テーブル参照中の想定外 SQLite 障害
        eprintln!("fts vocab lookup failed: {reason}");
    }
}
```

第3引数の `vocab_table` は `fts5vocab` 仮想テーブル名を受け取る。呼び出し側のスキーマ規約に応じて `"fts_chunks_vocab"` や `"messages_vocab"` などを指定する。`row` または `col` 型の vocabulary のみ対応する（`instance` 型には `cnt` カラムが無いため）。値はSQLにエスケープなしで埋め込まれるため、SQL identifier として妥当な文字列（先頭が ASCII 英字または `_`、以降 ASCII 英数字または `_`）のみ許容され、違反した場合は `SanitizeError::InvalidVocabTable` を返す。

第4引数の `normalization` は Phase 5 (#69) で追加。runtime default は NFKC + ASCII lowercase + 連続空白の collapse がすべて ON で、indexing 側 (`docs_fts.body`) と querying 側で folding が一致するよう設計されている。明示的に旧挙動が必要な呼び出しは `pre_phase_5_disabled()` を渡す（pre-#69 のスナップショットも `BaselineSnapshot.normalization` の serde-default としてこの値を参照する）。

`NEAR()` グループ、`^`/`+`/`-` プレフィックス、コロン、不均衡な引用符は内部で無害化される。`AND`/`OR`/`NOT` のようなoperator-like keywordは、前後に非operatorの語がある場合のみliteral termとして引用符で囲まれる。前後が欠けたdangling operator（例: 先頭の `NOT`、NEAR除去後に孤立した `OR`）は除去される。短い語（1-2文字）は指定した vocab テーブルがあればprefix展開される。vocab テーブルが存在しない場合だけはそのまま引用に劣化し、それ以外の SQLite 障害は `SanitizeError::VocabLookupFailed` を返す。

### query normalization 単体利用

`prepare_match_query` を経由せずに同じ folding を適用したい場合（例: indexing 側の body 正規化）は `normalize_for_fts` を直接呼ぶ。

```rust
use rurico::storage::{QueryNormalizationConfig, normalize_for_fts, pre_phase_5_disabled};

let config = QueryNormalizationConfig::default();
let folded = normalize_for_fts("ＡＢＣ\u{3000}DEF", &config);
assert_eq!(folded, "abc def");

// step 単位の選択（NFKC のみ ON など）も可
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

| Stage | 入力 → 出力                              | 提供型・関数                                                                                                                                                |
| ----- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `&str` → `Vec<Candidate>`                | `Candidate { source, doc_id, chunk_id, score, rank }`、`CandidateSource` 閉enum (`Fts` / `Vector`)                                                          |
| 2     | `&[Candidate]` → `Vec<MergedHit>`        | `MergeStrategy` trait、default impl `WeightedRrf`、設定 `HybridSearchConfig { rrf_k, source_weights }`、`merge_with_recency` + `RecencyConfig`              |
| 3     | `&[MergedHit]` → `Vec<MergedHit>`        | `Aggregator` trait + 4 impl (`IdentityAggregator` / `MaxChunkAggregator` / `DedupeAggregator` / `TopKAverageAggregator { k }`)、`group_by_parent` helper |
| 4     | `(&str, &[MergedHit], corpus)` → 並べ替え | 既存の `Rerank` trait に corpus lookup を組み合わせて呼ぶ                                                                                                   |
| 5     | rerank 結果 → top_k                      | downstream の表示層が責任を持つ                                                                                                                             |

```rust
use std::collections::HashMap;
use rurico::retrieval::{
    Aggregator, Candidate, CandidateSource, HybridSearchConfig, MaxChunkAggregator,
    MergeStrategy, RecencyConfig, WeightedRrf,
};

// Stage 1: 呼び出し側が FTS / vector の各 source から Candidate を集める
let candidates: Vec<Candidate> = /* ... */;

// Stage 2: weighted RRF（default は FTS / Vector が等しい重み、rrf_k = 60）
let mut weights = HashMap::new();
weights.insert(CandidateSource::Fts, 0.6);
weights.insert(CandidateSource::Vector, 1.4);
let merger = WeightedRrf::new(HybridSearchConfig {
    rrf_k: 60.0,
    source_weights: weights,
});
let merged = merger.merge(&candidates);

// 任意: recency boost を畳み込む（age_lookup は呼び出し側の corpus schema 依存）
let recency = RecencyConfig { weight: 0.3, half_life_days: 30.0 };
let merged_with_recency = merger.merge_with_recency(
    &candidates,
    &recency,
    |doc_id| { /* Option<f64>: 経過日数を返す。None なら recency を skip */ None },
);

// Stage 3: chunk-level retrieval なら parent に集約
let aggregator = MaxChunkAggregator;
let aggregated = aggregator.aggregate(&merged);

// Stage 4: 既存の Rerank trait を呼んで並べ替え
// Stage 5: top_k cutoff
```

#### Aggregator の使い分け

`MergedHit.chunk_id` が `Some(_)` のとき（chunk-level retrieval）に意味のある集約を提供する。`None` のままだと Stage 2 fusion が `doc_id` だけで折りたたんでいるため、すべての aggregator が identity と等価になる。

| Aggregator                       | 振る舞い                                                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `IdentityAggregator`             | パススルー。chunk-level identity を Stage 4 に届ける。default                             |
| `MaxChunkAggregator`             | parent ごとに最高スコアの chunk を残し、`chunk_id = None` で parent 単位に折りたたむ      |
| `DedupeAggregator`               | parent ごとに先頭 1 件のみ残す（順序保持の dedupe）                                       |
| `TopKAverageAggregator { k }`    | parent ごとに上位 `k` chunk スコアの平均を採用（`TopKAverageAggregator::new(k)` も可）   |

独自の `Aggregator` を実装する場合は `group_by_parent(&merged) -> HashMap<&str, Vec<&MergedHit>>` で parent 単位にバケットできる。

### ベクトルのバイト変換

`sqlite-vec` にベクトルをバインドする際は `bytemuck::cast_slice` で zero-copy に `&[f32] → &[u8]` を行う。rurico は little-endian ターゲットでのみビルドされるため、変換結果は sqlite-vec が期待する byte layout と一致する。

```rust
let vector: Vec<f32> = embedder.embed_query("検索")?;
let bytes: &[u8] = bytemuck::cast_slice(&vector);
stmt.execute(rusqlite::params![bytes])?;
```

### エラー型

エラー型はフェーズごとに分離されている。

**`ArtifactError`** — モデルファイルの取得・検証フェーズ

| variant            | 発生条件                                               |
| ------------------ | ------------------------------------------------------ |
| `MissingFile`      | 重みファイル / config / tokenizer が存在しない         |
| `InvalidConfig`    | config.json の読み込み/パース失敗                      |
| `InvalidTokenizer` | tokenizer.json のロード失敗                            |
| `WrongModelKind`   | safetensors のテンソルキーが期待するモデル種別と不一致 |
| `DownloadFailed`   | HF Hub からのダウンロード失敗                          |

**`ModelInitError`** — `Embedder::new` / `Embedder::probe` / `Reranker::new` / `Reranker::probe` フェーズ（embed と reranker で共通）

| variant        | 発生条件                                                  |
| -------------- | --------------------------------------------------------- |
| `Backend`      | MLX バックエンド初期化・重みロード・probe サブプロセス失敗 |
| `ModelCorrupt` | 重みは読めたがモデルが破損/非互換                         |

`Backend` は `message: String` と `source: Option<Box<dyn Error + Send + Sync>>` を持ち、`std::error::Error::source()` で原因チェーンを辿れる。

**`EmbedError`** — `Embed` トレイトメソッド（推論）フェーズ

| variant               | 発生条件                                   |
| --------------------- | ------------------------------------------ |
| `EmptySequence`       | モデルが seq_len=0 の出力を返した          |
| `BufferShapeMismatch` | 推論出力のバッファサイズが期待値と不一致   |
| `Inference`           | MLX 推論失敗                               |
| `Tokenizer`           | tokenizer エンコード失敗                   |
| `NonFiniteOutput`     | embedding 出力に NaN または Inf が含まれる |

**`RerankerError`** — `Rerank` トレイトメソッド（reranker 推論）フェーズ

| variant           | 発生条件                                                                  |
| ----------------- | ------------------------------------------------------------------------- |
| `Inference`       | MLX forward pass 失敗                                                     |
| `Tokenizer`       | tokenizer エンコード失敗                                                  |
| `NonFiniteOutput` | reranker 出力に NaN または Inf が含まれる                                 |
| `InitFailed`      | `LazyReranker` 初回呼び出し時の初期化失敗（cache 参照・ロード・DL）       |

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

| struct                | 振る舞い                                                                          |
| --------------------- | --------------------------------------------------------------------------------- |
| `MockEmbedder`        | 入力位置ごとに決定的な one-hot ベクトルを返す（バッチ時は入力順 `i`、単発は `0`） |
| `FailingEmbedder`     | 設定に応じてエラーを返す                                                          |
| `MismatchEmbedder`    | batch で入力より少ないベクトルを返す                                              |
| `AlternatingEmbedder` | `embed_document` が成功と失敗を交互に返す（初回は失敗）                           |
| `MockChunkedEmbedder` | 指定数の chunk を返す（multi-chunk テスト用）                                     |
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
cargo test --workspace                                                  # all tests
cargo clippy --workspace --all-targets --all-features -- -D warnings    # lint (matches CI)
cargo fmt --all -- --check                                              # format check
```

## テスト

```sh
cargo test --workspace                                                   # MLX ランタイム不要のテスト
cargo test --workspace --features test-mlx -- --ignored                  # MLX ランタイムテスト（通常 Terminal 推奨）
cargo run --bin mlx_smoke --features smoke --release -- verify-fixture   # embed 数値同等性検証（smoke binary）
```

Codex Desktop の `CODEX_SANDBOX=seatbelt` 環境では、MLX / Metal 初期化が abort することがあるため、
smoke binary は `sandbox::exit_if_seatbelt` 経由で `SEATBELT_SKIP_EXIT` (78) で停止し、`test-mlx` は ignored のままにしている。
実検証は通常の Terminal か、sandbox 外の実行環境で行う。

`mlx_smoke` binary は `smoke` Cargo feature でゲートされており、library として rurico を取り込む downstream には `tracing-subscriber` を持ち込まない。harness 用 just recipe（`just embed-verify` / `just embed-baseline` / `just probe-embed` / `just probe-reranker` 等）は `justfile` を参照。

検索品質の評価（Recall@k / MRR@k / nDCG@k）は [`amici`](https://github.com/thkt/amici) で行う。`CandidateSource` は `{ Fts, Vector }` の閉 enum に固定されている（prefix-ensemble は採用していない）。

## ライセンス

MIT
