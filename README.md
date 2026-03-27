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
- Rust 1.85+ (edition 2024)

## 使い方

```toml
[dependencies]
rurico = { git = "https://github.com/thkt/rurico" }
```

```rust
use rurico::embed::{Embed, Embedder, download_model};

// モデルをダウンロード（初回のみ、HF Hub にキャッシュ）
let paths = download_model()?;

// Embedder を作成して embedding を生成
let embedder = Embedder::new(&paths)?;
let vector = embedder.embed_query("検索クエリ")?;
// vector: Vec<f32> (768 次元)
```

### probe（安全なモデル検証）

MLXの初期化失敗はプロセスをabortする可能性がある。`probe` はモデルのロード可否を子プロセスで検証し、クラッシュを隔離する。

```rust
use rurico::embed::{Embedder, ProbeStatus, handle_probe_if_needed};

// main() の冒頭で呼ぶ（probe 子プロセスのハンドラ登録）
handle_probe_if_needed();

let paths = download_model()?;
match Embedder::probe(&paths)? {
    ProbeStatus::Available => { /* モデル利用可能 */ }
    ProbeStatus::BackendUnavailable => { /* MLX 利用不可 */ }
}
```

## テスト

```sh
cargo test                    # MLX ランタイム不要のテスト
cargo test --features test-mlx  # MLX ランタイムテスト（SIGABRT の可能性あり）
```

## ライセンス

MIT
