---
status: "accepted"
date: 2026-05-14
decision-makers: thkt
---

# Adopt process-level sqlite-vec auto-extension registration

## Context and Problem Statement

rurico の storage 層は sqlite-vec の vec0 仮想テーブルを使うが、sqlite-vec は **process-level auto-extension** として登録する idiom を取る (`sqlite_auto_extension` がプロセス内の全 `Connection::open` に対して以後自動 load する)。`src/storage.rs::ensure_sqlite_vec` (`pub fn`) が registration を行い、未完了の状態で開かれた Connection からは vec0 にアクセスできない。downstream CLI (`yomu` / `sae` / `amici`) は `Connection::open` 前に必ず `ensure_sqlite_vec` を呼ぶ必要があるが、この ordering 契約はコード型システムでは強制できず文書化に頼る。

(Override note: select-adr-template.sh は "ordering" keyword で `process-change` を返したが、本 ADR は実装契約パターンの記録なので `architecture-pattern` を選択した)

## Decision Drivers

* downstream が `Connection::open` 前に `ensure_sqlite_vec` を呼ばないと vec0 仮想テーブルが silent fail (`no such module: vec0`)
* sqlite-vec docs だけ読んでも auto-extension idiom には気づきにくい
* per-connection で `load_extension` する代替案は libsqlite の FFI 拡張が必要、移植性低下
* per-connection registration への切替は `pub fn ensure_sqlite_vec` の public API 互換性破壊変更

## Considered Options

* Adopt process-level auto-extension via `pub fn ensure_sqlite_vec` (status quo to be pinned)
* Switch to per-connection `Connection::load_extension`
* Hide registration behind a `StorageConnect` wrapper that enforces ordering at the type level

## Decision Outcome

Chosen option: **"Adopt process-level auto-extension via `pub fn ensure_sqlite_vec`"**, because sqlite-vec の primary idiom と整合し、追加 FFI surface を要求せず、downstream の Connection 管理に介入しないため。ordering 契約は ADR + rustdoc + README で明示し型システムでの強制は意図的に放棄する。

### Consequences

* Good, because sqlite-vec の primary idiom と整合、追加の FFI 拡張不要
* Good, because downstream の Connection 管理 (pool / async etc.) に介入しない
* Bad, because `Connection::open` 前に `ensure_sqlite_vec` を呼ぶ ordering 契約が型システムで強制されない
* Bad, because per-connection registration への切替は public API の破壊変更となる (移行コスト)

### Confirmation

- `src/storage.rs::ensure_sqlite_vec` の rustdoc に「`Connection::open` 前に一度だけ呼ぶ」を明記
- `README.md` ストレージ節 (L142-L145) で downstream へ同じ規約を伝達
- registration 失敗時は `src/storage.rs:29` で `tracing::error!(rc, "sqlite-vec auto-extension registration failed")` を emit、downstream は `rurico=error` directive で検知可

## Pros and Cons of the Options

### Adopt process-level auto-extension via `pub fn ensure_sqlite_vec`

`sqlite_auto_extension` がプロセス内の全 Connection に対して以後自動 load する。

* Good, because sqlite-vec の primary idiom と整合
* Good, because 追加 FFI 拡張不要 (rusqlite の `bundled` feature だけで足りる)
* Good, because downstream の Connection 管理に介入しない
* Bad, because ordering 契約は文書化と慣習でしか守れない

### Switch to per-connection `Connection::load_extension`

各 `Connection::open` 後に明示的に load。

* Good, because per-connection 状態が明示的、テストの並列分離しやすい
* Bad, because rusqlite で `load_extension` を呼ぶには `load_extension` feature が必須、追加 FFI surface
* Bad, because 既存 downstream API が壊れる (breaking change)
* Bad, because downstream が `Connection::open` を抽象化している場合に hook 困難

### Hide registration behind a `StorageConnect` wrapper

`StorageConnect::new() -> Self` 内で auto-extension 登録 + Connection 開設をまとめる。

* Good, because ordering を型で強制
* Bad, because downstream が独自の Connection pool / management を持つ場合に介入できない
* Bad, because 1 ラッパーで全 storage ops を覆う必要、API surface 増

## More Information

### Implementation Guidelines

- `ensure_sqlite_vec` は process 立ち上げ早期に呼ぶ (`main()` 冒頭が無難)
- 並行 thread から `Connection::open` する場合、`ensure_sqlite_vec` が完了するまで barrier (`OnceLock` / `LazyLock` 等) で待つ
- テストでは `serial_test::serial` で auto-extension registration の競合を避ける

### Monitoring

`ensure_sqlite_vec` 失敗時の `tracing::error!` (rc 値付き) で検知。downstream は `EnvFilter` に `rurico=error` を含めること。
