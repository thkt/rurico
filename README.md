# rurico

Apple Silicon (MLX) 上で日本語テキストのembeddingと類似検索を行うためのRustライブラリ。

[cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) (ModernBERT) をMLX backendで推論し、768次元のembeddingを生成する。生成したembeddingはSQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) でベクトル検索できる。

## 解決する問題

日本語semantic search CLIを複数構築する際に、embedding生成・モデル管理・ベクトルストレージが各CLIで重複する。ruricoはこの共通基盤を1 crateに集約し、downstreamのCLIは検索ロジックに集中できるようにする。

## モジュール構成

| モジュール   | 役割                                                                              |
| ------------ | --------------------------------------------------------------------------------- |
| `embed`      | embedding 生成（MLX 推論、tokenization、pooling、probe）                          |
| `modernbert` | ModernBERT モデル定義と config                                                    |
| `storage`    | SQLite + sqlite-vec のベクトル検索ユーティリティ（FTS、RRF merge、recency decay） |
| `text`       | テキスト分割（段落 > 行 > 文字境界で UTF-8 安全に分割）                           |

## 要件

- macOS (Apple Silicon) — MLX backend必須
- Rust 1.91+ (edition 2024)

## 使い方

```toml
[dependencies]
rurico = { git = "https://github.com/thkt/rurico", rev = "main" }
```

MLXの初期化失敗はプロセスをabortする可能性がある。`probe` でモデルのロード可否を子プロセスで検証してからEmbedderを作成する。

```rust
use rurico::embed::{Embed, Embedder, ProbeStatus, download_model, handle_probe_if_needed};

// probe 子プロセスのハンドラ登録（main() の冒頭で呼ぶ）
handle_probe_if_needed();

// モデルをダウンロード（初回のみ、HF Hub にキャッシュ）
let paths = download_model()?;

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

// クエリ: Vec<f32> (768 次元)。MAX_SEQ_LEN 超過時は自動 truncate。
let query_vec = embedder.embed_query("検索クエリ")?;

// ドキュメント: ChunkedEmbedding。短文は chunks.len()==1、
// 長文は overlapping chunks に分割される。
let doc = embedder.embed_document("長いドキュメント...")?;
for chunk_vec in &doc.chunks {
    // chunk_vec: &Vec<f32> (768 次元)
}
```

### 定数

| 定数                   | 値               | 用途                                          |
| ---------------------- | ---------------- | --------------------------------------------- |
| `EMBEDDING_DIMS`       | `768`            | 出力ベクトルの次元数                          |
| `MAX_SEQ_LEN`          | `8192`           | モデル入力全体長（BOS + prefix + text + EOS） |
| `CHUNK_OVERLAP_TOKENS` | `2048`           | document chunk 間の overlap token 数          |
| `QUERY_PREFIX`         | `"検索クエリ: "` | クエリ埋め込みときに先頭へ付加                |
| `DOCUMENT_PREFIX`      | `"検索文書: "`   | ドキュメント埋め込みときに先頭へ付加          |

### Document Embedding

`embed_document` は `ChunkedEmbedding` を返す。テキストが `MAX_SEQ_LEN` 以内なら `chunks.len() == 1` で従来と同等のembeddingを返す。超過時はprefixを保持したoverlapping chunksに分割され、各chunkが独立した768次元embeddingになる。

`embed_documents_batch` は入力件数と同数の `Vec<ChunkedEmbedding>` を返し、入力順を保持する。

### モデルキャッシュの確認

ネットワークアクセスなしでモデルがローカルにあるか確認できる。

```rust
use rurico::embed::model_paths_if_cached;

if let Some(paths) = model_paths_if_cached()? {
    // キャッシュ済み — そのまま Embedder::new に渡せる
} else {
    // 未ダウンロード
}
```

### probe なしの簡易利用

abortリスクを許容できるスクリプト等ではprobeを省略できる。

```rust
use rurico::embed::{Embed, Embedder, download_model};

let paths = download_model()?;
let embedder = Embedder::new(&paths)?;
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

match prepare_match_query(&conn, user_input) {
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
}
```

`NEAR()` グループ、`^`/`+`/`-` プレフィックス、コロン、不均衡な引用符は内部で無害化される。`AND`/`OR`/`NOT` のようなoperator-like keywordは、前後に非operatorの語がある場合のみliteral termとして引用符で囲まれる。前後が欠けたdangling operator（例: 先頭の `NOT`、NEAR除去後に孤立した `OR`）は除去される。短い語（1-2文字）は `fts_chunks_vocab` テーブルがあればprefix展開される（なければそのまま引用）。

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

`EmbedError` はembedding操作全般のエラーを表す。

| variant             | 発生条件                          |
| ------------------- | --------------------------------- |
| `ModelNotFound`     | 重みファイルが見つからない        |
| `DimensionMismatch` | 出力テンソルの次元が不一致        |
| `Config`            | config.json の読み込み/パース失敗 |
| `Inference`         | MLX 推論失敗                      |
| `Tokenizer`         | tokenizer のロード/エンコード失敗 |
| `Download`          | モデルダウンロード失敗            |
| `ModelCorrupt`      | 重みは読めたがモデルが破損/非互換 |

### ログ出力

内部の警告は `log` crate経由で出力される。`env_logger::init()` 等でloggerを初期化すると観測できる。

### テストサポート（downstream 向け）

`test-support` featureを有効にすると、downstream crateのテストで使えるモックが利用できる。

```toml
[dev-dependencies]
rurico = { git = "https://github.com/thkt/rurico", tag = "v0.2.0", features = ["test-support"] }
```

| struct                | 振る舞い                                  |
| --------------------- | ----------------------------------------- |
| `MockEmbedder`        | 決定的な one-hot ベクトルを返す           |
| `FailingEmbedder`     | 設定に応じてエラーを返す                  |
| `MismatchEmbedder`    | batch で入力より少ないベクトルを返す      |
| `AlternatingEmbedder` | `embed_document` が成功と失敗を交互に返す |

```rust
use rurico::embed::{Embed, MockEmbedder};

let embedder = MockEmbedder;
let v = embedder.embed_query("テスト")?;
assert_eq!(v.len(), 768);
```

## テスト

```sh
cargo test                    # MLX ランタイム不要のテスト
cargo test --features test-mlx  # MLX ランタイムテスト（SIGABRT の可能性あり）
```

## ライセンス

MIT
