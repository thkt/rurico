# rurico

Apple Silicon (MLX) 上で日本語テキストのembedding・reranking・類似検索を行うためのRustライブラリ。

[cl-nagoya/ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) ファミリー (ModernBERT) をMLX backendで推論する。embedモデルは256〜768次元のembeddingを生成し（モデルサイズにより異なる）、rerankerは検索結果をcross-encoderでスコアリングする。生成したembeddingはSQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) でベクトル検索できる。

## 解決する問題

日本語semantic search CLIを複数構築する際に、embedding生成・reranking・モデル管理・ベクトルストレージが各CLIで重複する。ruricoはこの共通基盤を1 crateに集約し、downstreamのCLIは検索ロジックに集中できるようにする。

## モジュール構成

| モジュール    | 役割                                                                                 |
| ------------- | ------------------------------------------------------------------------------------ |
| `embed`       | embedding 生成（MLX 推論、tokenization、pooling、probe）                             |
| `reranker`    | 検索結果の reranking（cross-encoder スコアリング、probe）                            |
| `modernbert`  | ModernBERT モデル定義と config                                                       |
| `storage`     | SQLite + sqlite-vec のベクトル検索ユーティリティ（FTS、RRF merge、recency decay）    |
| `text`        | テキスト分割（段落 > 行 > 文字境界で UTF-8 安全に分割）                              |
| `artifacts`   | モデルファイルの型付き検証パイプライン（`CandidateArtifacts` → `VerifiedArtifacts`） |
| `model_probe` | サブプロセス probe 基盤（`handle_probe_if_needed`、`ProbeStatus`）                   |

## 要件

- macOS (Apple Silicon) — MLX backend必須
- Rust 1.94+ (edition 2024)

## 使い方

```toml
[dependencies]
rurico = { git = "https://github.com/thkt/rurico", rev = "main" }
```

MLXの初期化失敗はプロセスをabortする可能性がある。`probe` でモデルのロード可否を子プロセスで検証してからEmbedderを作成する。

```rust
use rurico::embed::{Embed, Embedder, ModelId, download_model};
use rurico::model_probe::{ProbeStatus, handle_probe_if_needed};

// probe 子プロセスのハンドラ登録（main() の冒頭で呼ぶ）
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
| `CHUNK_OVERLAP_TOKENS` | `2048`           | document chunk 間の overlap token 数                                               |
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

ユーザー入力をFTS5 `MATCH` に安全に渡すには `prepare_match_query` を使う。内部でsanitize → expandの2段階を経て、全トークンを引用符で囲んだ `MatchFtsQuery` を返す。

```rust
use rurico::storage::{prepare_match_query, SanitizeError};

match prepare_match_query(&conn, user_input, "fts_chunks_vocab") {
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

`NEAR()` グループ、`^`/`+`/`-` プレフィックス、コロン、不均衡な引用符は内部で無害化される。`AND`/`OR`/`NOT` のようなoperator-like keywordは、前後に非operatorの語がある場合のみliteral termとして引用符で囲まれる。前後が欠けたdangling operator（例: 先頭の `NOT`、NEAR除去後に孤立した `OR`）は除去される。短い語（1-2文字）は指定した vocab テーブルがあればprefix展開される。vocab テーブルが存在しない場合だけはそのまま引用に劣化し、それ以外の SQLite 障害は `SanitizeError::VocabLookupFailed` を返す。

### ハイブリッド検索ユーティリティ

FTS5とベクトル検索の結果をReciprocal Rank Fusionでマージする。

```rust
use rurico::storage::rrf_merge;

let fts_hits = vec![(1, 0.9), (2, 0.7), (3, 0.5)];
let vec_hits = vec![(2, 0.95), (4, 0.8), (1, 0.6)];
let merged = rrf_merge(&fts_hits, &vec_hits);
// ランク位置のみで統合 — スコア値は無視される
```

### recency decay

時間経過による減衰スコアを計算する。age=0で1.0、半減期で0.5。

```rust
use rurico::storage::recency_decay;

let score = recency_decay(7.0, 30.0); // 7日経過、半減期30日
```

### ベクトルのバイト変換

`sqlite-vec` にベクトルをバインドする際に使う。

```rust
use rurico::storage::f32_as_bytes;

let vector: Vec<f32> = embedder.embed_query("検索")?;
stmt.execute(rusqlite::params![f32_as_bytes(&vector)])?;
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

**`EmbedInitError`** — `Embedder::new` / `Embedder::probe` フェーズ

| variant        | 発生条件                               |
| -------------- | -------------------------------------- |
| `Backend`      | MLX バックエンド初期化・重みロード失敗 |
| `ModelCorrupt` | 重みは読めたがモデルが破損/非互換      |

**`EmbedError`** — `Embed` トレイトメソッド（推論）フェーズ

| variant               | 発生条件                                   |
| --------------------- | ------------------------------------------ |
| `EmptySequence`       | モデルが seq_len=0 の出力を返した          |
| `BufferShapeMismatch` | 推論出力のバッファサイズが期待値と不一致   |
| `Inference`           | MLX 推論失敗                               |
| `Tokenizer`           | tokenizer エンコード失敗                   |
| `NonFiniteOutput`     | embedding 出力に NaN または Inf が含まれる |

公開APIの失敗契約はrustdocの `# Errors` に記載する。repoの運用ルールは
[`docs/errors.md`](docs/errors.md) を参照。

### ログ出力

内部の警告は `log` crate経由で出力される。`env_logger::init()` 等でloggerを初期化すると観測できる。

### テストサポート（downstream 向け）

`test-support` featureを有効にすると、downstream crateのテストで使えるモックが利用できる。

```toml
[dev-dependencies]
rurico = { git = "https://github.com/thkt/rurico", rev = "main", features = ["test-support"] }
```

| struct                | 振る舞い                                                |
| --------------------- | ------------------------------------------------------- |
| `MockEmbedder`        | 決定的な one-hot ベクトルを返す                         |
| `FailingEmbedder`     | 設定に応じてエラーを返す                                |
| `MismatchEmbedder`    | batch で入力より少ないベクトルを返す                    |
| `AlternatingEmbedder` | `embed_document` が成功と失敗を交互に返す（初回は失敗） |
| `MockChunkedEmbedder` | 指定数の chunk を返す（multi-chunk テスト用）           |

```rust
use rurico::embed::{Embed, MockEmbedder};

let embedder = MockEmbedder;
let v = embedder.embed_query("テスト")?;
assert_eq!(v.len(), 768);
```

## テスト

```sh
cargo test                    # MLX ランタイム不要のテスト
cargo test --features test-mlx -- --ignored  # MLX ランタイムテスト（通常 Terminal 推奨）
```

Codex Desktop の `CODEX_SANDBOX=seatbelt` 環境では、MLX / Metal 初期化が abort することがあるため、
smoke binary は専用 exit code で停止し、`test-mlx` は ignored のままにしている。
実検証は通常の Terminal か、sandbox 外の実行環境で行う。

## ライセンス

MIT
