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
rurico = { git = "https://github.com/thkt/rurico", tag = "v0.2.0" }
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
let vector = embedder.embed_query("検索クエリ")?;
// vector: Vec<f32> (768 次元)
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
        if matched.is_empty() {
            // vocab 展開後にトークンが残らなかった
            return Ok(Vec::new());
        }
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

`NEAR()` グループ、`^`/`+`/`-` プレフィックス、コロン、不均衡な引用符は内部で無害化される。`AND`/`OR`/`NOT` のようなoperator-like keywordはliteral termとして保持され、引用符で囲まれてMATCHに渡される。短い語（1-2文字）は `fts_chunks_vocab` テーブルがあればprefix展開される（なければそのまま引用）。

### ログ出力

内部の警告は `log` crate経由で出力される。`env_logger::init()` 等でloggerを初期化すると観測できる。

## テスト

```sh
cargo test                    # MLX ランタイム不要のテスト
cargo test --features test-mlx  # MLX ランタイムテスト（SIGABRT の可能性あり）
```

## ライセンス

MIT
