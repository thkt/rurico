# rurico

Apple Silicon (MLX) 上で日本語テキストのembedding・reranking・類似検索を行うためのRustライブラリ。

[cl-nagoya/ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) ファミリー (ModernBERT) をMLX backendで推論する。embedモデルは256〜768次元のembeddingを生成し（モデルサイズにより異なる）、rerankerは検索結果をcross-encoderでスコアリングする。生成したembeddingはSQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) でベクトル検索できる。

## 解決する問題

日本語semantic search CLIを複数構築する際に、embedding生成・reranking・モデル管理・ベクトルストレージが各CLIで重複する。ruricoはこの共通基盤を1 crateに集約し、downstreamのCLIは検索ロジックに集中できるようにする。

## モジュール構成

| モジュール    | 役割                                                                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `embed`       | embedding 生成（MLX 推論、tokenization、pooling、probe）                                                                              |
| `reranker`    | 検索結果の reranking（cross-encoder スコアリング、probe）                                                                             |
| `modernbert`  | ModernBERT モデル定義と config                                                                                                        |
| `storage`     | SQLite + sqlite-vec のベクトル検索プリミティブ（FTS sanitize / `MatchFtsQuery`、`rrf_merge`、`recency_decay`、`QueryNormalizationConfig`） |
| `retrieval`   | 5-stage retrieval pipeline contract（ADR 0004）— `Candidate` / `MergedHit` / `MergeStrategy` / `Aggregator` / `HybridSearchConfig` / `RecencyConfig` |
| `eval` ※     | 検索評価ハーネス（Recall@k / MRR@k / nDCG@k、`tests/fixtures/eval/baseline.json`）。ADR 0003 / Issue #65                              |
| `text`        | テキスト分割（段落 > 行 > 文字境界で UTF-8 安全に分割）                                                                               |
| `artifacts`   | モデルファイルの型付き検証パイプライン（`CandidateArtifacts` → `VerifiedArtifacts`）                                                  |
| `model_probe` | サブプロセス probe 基盤（`handle_probe_if_needed`、`ProbeStatus`）                                                                    |

※ `eval` モジュールと `eval_harness` バイナリは `eval-harness` feature で gate されている。詳細は [テスト](#テスト) と [検索評価ハーネス](#検索評価ハーネス) を参照。

## 要件

- macOS (Apple Silicon) — MLX backend必須
- Rust 1.95+ (edition 2024)

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

### ハイブリッド検索プリミティブ — `rrf_merge`

FTS5 とベクトル検索の結果を Reciprocal Rank Fusion で統合する低レベル関数。スコア値を無視してランク位置のみで折りたたむ。weight 調整・recency 加味・複数 source 対応を扱いたい場合は [Retrieval Pipeline](#retrieval-pipeline5-stage-contract) の `WeightedRrf` / `merge_with_recency` を使う。

```rust
use rurico::storage::rrf_merge;

let fts_hits = vec![(1, 0.9), (2, 0.7), (3, 0.5)];
let vec_hits = vec![(2, 0.95), (4, 0.8), (1, 0.6)];
let merged = rrf_merge(&fts_hits, &vec_hits);
```

### recency decay プリミティブ

時間経過による減衰スコアを計算する。age=0 で 1.0、半減期で 0.5。fusion に組み込みたい場合は `RecencyConfig` + `merge_with_recency` を参照（次節）。

```rust
use rurico::storage::recency_decay;

let score = recency_decay(7.0, 30.0); // 7日経過、半減期30日
```

### Retrieval Pipeline（5-stage contract）

`retrieval` モジュールは ADR 0004 が固定する 5 ステージ pipeline contract を提供する。`storage` のプリミティブ（`prepare_match_query` / `rrf_merge` / `recency_decay`）を組み合わせる際の標準配線で、Phase 3 (#67) で aggregation hook、Phase 4 (#68) で hybrid weight/recency、#76 で chunk-level retrieval が揃った。

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
cargo test --workspace                                          # MLX ランタイム不要のテスト
cargo test --workspace --features test-mlx -- --ignored         # MLX ランタイムテスト（通常 Terminal 推奨）
cargo test --workspace --features eval-harness                  # 検索評価ハーネスの単体テスト
```

Codex Desktop の `CODEX_SANDBOX=seatbelt` 環境では、MLX / Metal 初期化が abort することがあるため、
smoke binary は専用 exit code で停止し、`test-mlx` は ignored のままにしている。
実検証は通常の Terminal か、sandbox 外の実行環境で行う。

## 検索評価ハーネス

`eval-harness` feature を有効にすると、検索品質を **Recall@k / MRR@k / nDCG@k** （95% bootstrap CI、`n = 1000`、`seed = 42`）で計測できる。Issue #53 (Phase 1〜6 + chunk-level retrieval #76) で構築した。

- Fixture: `tests/fixtures/eval/`（60 docs × 7 ドメイン、147 queries × 7 IR categories、graded relevance `{0, 1, 2, 3}`、表記ゆれを含む `variant_notation` 21 件を含む）
- Baseline スナップショット: `tests/fixtures/eval/baseline.json`（`schema_version: "1.1"`、`fnv1a64:` fixture hash 付き）
- Methodology: ADR 0003、再現手順は `docs/eval/baseline.md`
- Pipeline 設計: ADR 0004（5-stage contract）

```sh
# ベースラインを再生成（MLX 必要、Apple Silicon）
cargo run --release --features eval-harness --bin eval_harness -- \
    capture-baseline output=tests/fixtures/eval/baseline.json

# 既存スナップショットからのドリフトを検証（exit 0 = pass、1 = regression、2 = usage、3 = infra）
cargo run --release --features eval-harness --bin eval_harness -- \
    verify-baseline baseline=tests/fixtures/eval/baseline.json

# 複数 baseline のメトリクス差分を markdown matrix で比較（Phase 4 #68）
cargo run --release --features eval-harness --bin eval_harness -- \
    compare-baselines paths=baselines/default.json,baselines/fts-heavy.json
```

`capture-baseline` で受けるチューニング flag:

| flag                                              | 効果                                              |
| ------------------------------------------------- | ------------------------------------------------- |
| `aggregation=identity\|max-chunk\|dedupe\|topk-average` | Stage 3 aggregator を切り替える                   |
| `rrf_k=`                                          | RRF の k 値（default 60.0）                       |
| `fts_weight=` / `vector_weight=`                  | Stage 2 の per-source weight                      |
| `normalize_nfkc=` / `normalize_lowercase=` / `normalize_collapse_whitespace=` | Stage 1 の query normalization step を個別 on/off |

なお、Phase 6 prefix-ensemble（Issue #70）は ADR 0005 で **Not Adopted** と結論済みのため、`CandidateSource` は `{ Fts, Vector }` の閉 enum に固定されている。

## ライセンス

MIT
