# Undocumented Decisions Audit: 2026-05-14

Issue [#159](https://github.com/thkt/rurico/issues/159) — ADR 0001-0007 で覆われていない設計判断を `/audit-undocumented` skill で発掘し、`critic-design` で篩った記録。実 ADR 起票は本 scan の scope 外、候補リストまでに留める。

## Summary

| Metric                   | Value |
| ------------------------ | ----- |
| Large files scanned      | 8     |
| Documents scanned        | 5     |
| Decision candidates      | 12    |
| ADR-covered (excluded)   | 0     |
| Net new candidates       | 12    |
| ADR promotion candidates | 3 (keep) + 5 (downgrade) |
| Bug-fix follow-up        | 1 (#8) |

post-challenge verdicts: **keep 3 / downgrade 5 / drop 4** (drop rate 33%、scout dogfood 16-29% より高めだが各 drop に明確根拠あり)。

## Methodology

| Step | Action |
| ---- | ------ |
| 1    | 大型ファイル検出 (`>400` 行): bfs + wc で 12 ファイル。tests 系 (3 件) と binary (`mlx_smoke.rs`) を除外し 8 ファイルへ |
| 2    | prose document 検出: `README.md` / `CONTRIBUTING.md` / `docs/THREAT_MODEL.md` / `justfile` / `Cargo.toml [workspace.lints.*]` |
| 3    | reviewer-rust agent は skip し、各ファイルを直接 Read + module-doc + public symbol から判断を抽出 (`/audit-adr-drift` 同様の方針)。Issue #159 で名指された 6 ファイル mandatory、`embed/mlx.rs` (1058) と `artifacts.rs` (903) を core path として追加 |
| 4    | prose decision verbs scan (英: must/never/decide, 和: 決定/採用/禁止/方針) + External ADR ref cross-check |
| 5    | 各 candidate に impact (H/M/L) + reversibility (high/medium/low) + `documented?` (Yes/Partial/No) + `incomplete-contract?` (Yes/No) を付与 |
| 6    | initial ranking (impact=H AND reversibility=low|medium OR incomplete-contract=Yes) → `critic-design` で篩い |
| 7    | report 出力 (本ファイル) |

reviewer agent skip の根拠: prose docs + module-level doc-comment + 主要型 / 公開 fn シグネチャが Read 1-2 回で全把握できる規模 (8 ファイル × 平均 750 行)。bundle reviewer は 1 agent あたり 6000 行超で文脈圧迫の方が大きい。

## Large File Decisions

### `src/model_probe.rs` (655 行)

| # | Line     | Decision                                              | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | ----------------------------------------------------- | ----------- | -------------------- | ------ | ------------- |
| 1 | 345-387  | `FORWARD` env allowlist の保守ルール (新規 runtime dep が env 読むとき rationale 込で追加せよ) | Partial     | Yes                  | H      | medium        |
| 2 | 485-512  | `COLLECT_PIPE_TIMEOUT = 2s` + grandchild reader thread leak の accepted-risk 設計 | Partial     | Yes                  | M      | medium        |

probe child の wire format (`PROBE_EXIT_*` 7 個 + `PROBE_ACK` 1 個 = 計 8 個の IPC 契約) は L11-L25 module-doc table で集約され、SEC-002/003 として `THREAT_MODEL.md` から呼応する。ADR 化済の領域ではないが、文書化の定位置が確立されている。

### `src/modernbert/model.rs` (834 行)

| # | Line     | Decision                                                              | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | --------------------------------------------------------------------- | ----------- | -------------------- | ------ | ------------- |
| 3 | 384      | `MASK_FILL = -1e9` (NEG_INFINITY 不可) — Apple Silicon Metal kernel NaN 回避 | Partial     | Yes                  | H      | low           |
| 4 | 204-212  | `local_mask_cache` ≤4 entries 上限 (`BUCKET_BOUNDS` integrity との整合) | Partial     | Yes                  | M      | medium        |

`validate_attention_mask` / `forward` truncate 系の invariant は ADR 0002 と ADR 0007 で覆われている。

### `src/storage/search.rs` (659 行)

| # | Line     | Decision                                                     | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | ------------------------------------------------------------ | ----------- | -------------------- | ------ | ------------- |
| 5 | 211-218  | `is_valid_sql_identifier` validator: 「SQL identifier として interpolate するときは必ず通す」 | Partial     | Yes                  | H      | medium        |

ADR 0001 で `MatchFtsQuery` / `SanitizeError` 型契約は覆われ済。operator preservation (`AND/OR/NOT`) も ADR-bound。`LIMIT 25` (L247) 等の magic number は実装詳細扱い。

### `src/model_io.rs` (549 行)

| # | Line     | Decision                                                                | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | ----------------------------------------------------------------------- | ----------- | -------------------- | ------ | ------------- |
| 6 | 41       | `BUCKET_BOUNDS = [128, 512, 2048, 8192]` cross-module invariant (`assign_bucket` で round up、違反は MLX compile cache 暴発) | Partial     | Yes                  | H      | medium        |
| 7 | 277-285  | `TOKEN_BUDGET = 256_000` + `compute_sub_batch_size` formula (embed/reranker 両 caller pin) | Yes (test)  | No                   | M      | medium        |

`ModelArtifact::Kind` 関連型 (L75-L82) は型システムで強制、ADR redundant。

### `src/retrieval.rs` (1152 行)

| # | Line     | Decision                                                                 | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | ------------------------------------------------------------------------ | ----------- | -------------------- | ------ | ------------- |
| 12 | 322-327, 342 | `MergeStrategy::merge` / `Aggregator::aggregate` sort: score desc + `doc_id` ascending + `chunk_id` ascending tie-break | Partial     | Yes                  | M      | high          |

ADR 0004 が Stage 2/3 を覆うが tie-break 方向 (ascending/descending) は ADR 文面で言及なし。test T-104-001 のみ pin。`#[serde(default)]` / Phase 4-5-6 invariants は ADR 0004 / 0005 で document 済。

### `src/embed/mlx.rs` (1058 行)

ADR 0002 (`forward_sub_batch` / `gpu_pool_and_normalize` 系) と ADR 0007 (`split_pooled` 非 finite 検出 warn) が主要 invariant を覆う。`IndexedChunk` 構造体 (L18-L37) + `distribute_into_buckets` / `build_indexed_chunks` の bucket 順序保存 invariant は test (T-BKT-005/006/008) で pin 済、ADR 候補なし。

### `src/artifacts.rs` (903 行)

| # | Line     | Decision                                                                        | Documented? | Incomplete-contract? | Impact | Reversibility |
| - | -------- | ------------------------------------------------------------------------------- | ----------- | -------------------- | ------ | ------------- |
| 11 | 22-27    | `EmbedKind(())` / `RerankerKind(())` sealed-pattern: pub struct + private inner field で downstream の kind 構築を遮断 | No          | Yes                  | M      | low           |

`CandidateArtifacts<K>` → `VerifiedArtifacts<K>` の type state pattern (L29-L46) は module-doc で記述済。`delete_files` の partial-error 戦略 (L77-L90) は rustdoc 完備。

### `src/model_probe/tests.rs` (885 行)

`FailingWriter` / `FlushTrackingWriter` / `FlushFailingWriter` (L8-L53) は production code が `Write` を受け取る seam pattern を支える test infrastructure。test infrastructure のみで設計判断としての ADR 候補なし。Issue #159 が念のため指定したが、現状実害なし。

## Prose Document Decisions

### `README.md` (25KB)

| # | Line     | Decision Verb | Decision                                                            | ADR Coverage |
| - | -------- | ------------- | ------------------------------------------------------------------- | ------------ |
| 9 | 142-145  | (例外なし)    | `ensure_sqlite_vec` は `Connection::open` の前に一度だけ呼ぶ ordering 契約 | None         |
| - | 28       | (記述)        | 検索品質評価ハーネスは amici に移譲 (ADR 0006 反映)                | ADR 0006     |
| - | 39-42    | (記述)        | `Cargo.toml の version` と git tag を同期、downstream は tag 参照のみ | OUTCOME.md (Constraints) |
| - | 407      | (記述)        | `CandidateSource = { Fts, Vector }` 閉 enum (prefix-ensemble 不採用)  | ADR 0004 / 0005 |

### `CONTRIBUTING.md` (2.2KB)

| # | Line     | Decision Verb | Decision                                              | ADR Coverage |
| - | -------- | ------------- | ----------------------------------------------------- | ------------ |
| - | 42       | (規約)        | Conventional Commits: `feat:/fix:/refactor:/docs:/chore:/polish:/ci:` 等 | None (project convention) |

### `docs/THREAT_MODEL.md` (5.1KB)

| # | Line     | Decision Verb | Decision                                                            | ADR Coverage |
| - | -------- | ------------- | ------------------------------------------------------------------- | ------------ |
| 8 | (欠落)   | (記述漏れ)    | SEC-001 / SEC-002 が enumeration されていない (コード `src/model_probe.rs:99,258` 等で参照あり) | None — **bug** |
| - | 35-59    | (受容済 risk) | SEC-003 probe exit code を filesystem oracle として受容              | THREAT_MODEL.md 本体 |
| - | 61-83    | (受容済 risk) | SEC-004 `DYLD_LIBRARY_PATH` / `DYLD_FALLBACK_LIBRARY_PATH` を FORWARD で許可 | THREAT_MODEL.md 本体 |

### `justfile` (1.5KB)

| # | Line     | Decision Verb | Decision                                              | ADR Coverage |
| - | -------- | ------------- | ----------------------------------------------------- | ------------ |
| - | 31-43    | (記述)        | mlx_smoke harness 一式 (ADR 0002 reference)           | ADR 0002     |
| - | 45-54    | (記述)        | probe smoke recipe (embed / reranker)                  | None (binary wrapper) |

### `Cargo.toml` `[workspace.lints.*]`

| # | Line     | Decision Verb | Decision                                                            | ADR Coverage |
| - | -------- | ------------- | ------------------------------------------------------------------- | ------------ |
| 10 | 14-17    | (規約)        | `unsafe_code = "deny"` (not `"forbid"`) で FFI crate のみ `#[allow(unsafe_code)]` 許可 | None         |
| - | 19-37    | (config 詳細) | clippy lints 選定 (`absolute_paths`, `wildcard_imports`, `str_to_string` 等 11 個 deny) | None (statement-of-fact config) |

## External ADR Dependencies

| # | File:Line | External ADR ref | Recommended action |
| - | --------- | ---------------- | ------------------ |
| — | —         | (検出なし)       | —                  |

`ADR-NNNN` / `adr-nnnn` パターンを `src/` / `tests/` / `Cargo.toml` / `README.md` / `crates/` 全体に対して scan した結果、外部 ADR (dotclaude meta ADR 等) への ref は不検出。ADR 0001-0007 はすべて local。

## ADR Promotion Candidates (post-challenge)

| # | Candidate                              | Initial | Challenge | Final                  | Rationale |
| - | -------------------------------------- | ------- | --------- | ---------------------- | --------- |
| 1 | `model_probe.rs:345-387` FORWARD allowlist 保守ルール | promote | downgrade | THREAT_MODEL.md or doc-comment | maintenance rule であり frozen decision でない。SEC-004 隣に "FORWARD maintenance" section 追加が定位置 |
| 2 | `model_probe.rs:485-512` reader thread leak accepted-risk | promote | downgrade | THREAT_MODEL.md (SEC-005 新規) | SEC-003/004 同型の accepted-risk、threat model に並べる |
| 3 | `modernbert/model.rs:384` MASK_FILL = -1e9 | promote | **keep**  | ADR                    | Adoption Gate 3 条件通過: Apple Silicon Metal kernel hardware-specific、math defaults `-inf` から逆走 hard to reverse、precision vs NaN 回避 real trade-off |
| 4 | `modernbert/model.rs:204-212` local_mask_cache ≤4 | promote | downgrade | #6 ADR に統合          | #6 の derived consequence。`[BUCKET_BOUNDS]` intra-doc link で既に cross-reference 済、#6 ADR 化で自動カバー |
| 5 | `storage/search.rs:211-218` SQL identifier validator | promote | drop      | skip                   | call site 1 箇所のみで YAGNI Boundary (≥2) 失敗、speculative documentation。第 2 出現時に再評価 |
| 6 | `model_io.rs:41` BUCKET_BOUNDS cross-module invariant | promote | **keep**  | ADR                    | Adoption Gate 全通過: 違反すると MLX compile cache 暴発 (hard to reverse / observability 悪い)、4 magic number は ADR なしでは説明不能、固定 kernel 数 vs sequence 適応の trade-off |
| 7 | `model_io.rs:277-285` TOKEN_BUDGET formula | (not)   | drop      | skip                   | test `compute_sub_batch_size_matches_formula_per_bucket` で formula が pin 済、ADR は drift を生む側 |
| 8 | THREAT_MODEL.md SEC-001/SEC-002 enumeration 欠落 | promote | drop      | **bug-fix followup**   | decision ではなく orphaned identifier bug。コード 3 箇所 (`model_probe.rs:99,258`、`model_probe/tests.rs:595`) が SEC-002 を参照、THREAT_MODEL.md は知らない |
| 9 | `README.md:142-145` + `storage.rs:10-29` ensure_sqlite_vec ordering | promote | **keep**  | ADR                    | `pub fn` で downstream が必ず `Connection::open` 前に呼ぶ API contract、違反は silent failure、per-connection registration への切替は breaking change |
| 10 | `Cargo.toml [lints]` unsafe_code = deny | (not)   | drop      | skip                   | inline comment (L27-29) が source of truth、CI が必ず evaluate する既存 enforcement あり |
| 11 | `artifacts.rs:22-27` EmbedKind/RerankerKind sealed-pattern | promote | downgrade | rustdoc                | Rust idiom、rustdoc の定位置。"Sealed: external crates cannot construct..." を doc-comment に 1 行追加で意図伝達 |
| 12 | `retrieval.rs:322-327,342` sort tie-break ascending | promote | downgrade | ADR 0004 + rustdoc     | (a) `Aggregator::aggregate` doc に "ascending" 追記 + chunk_id 二次キー明記、(b) ADR 0004 Consequences に "score desc, doc_id asc, chunk_id asc" 1 行追加 |

Per-source summary line: `keep 3 / downgrade 5 / drop 4`.

### Keep (ADR 起票推奨 3 件)

1. **#3 `MASK_FILL = -1e9` (Metal NaN hazard)** — Apple Silicon Metal kernel の numerical contract。新規メンテナが `f32::NEG_INFINITY` に「直そう」とすると `9 token 周辺で NaN` regression を引き起こす。ADR で defense-of-the-status-quo を固定する価値あり
2. **#6 `BUCKET_BOUNDS` cross-module invariant** — `[128, 512, 2048, 8192]` がなぜ 4 つでこの値かを ADR で固定。違反 (caller が round up を忘れる、bucket を 5 つに増やす) すると `local_mask_cache` 膨張 + MLX compile cache 暴発。enforcement (`assign_bucket` panic) は「len > MAX_SEQ_LEN なら panic」しかカバーしない
3. **#9 `ensure_sqlite_vec` process-level ordering** — public API contract、downstream への commitment

### Downgrade (ADR ではなく既存文書/comment に absorb 5 件)

| # | Absorb target |
| - | ------------- |
| 1 | `THREAT_MODEL.md` の SEC-004 隣に `FORWARD maintenance` section 追加 |
| 2 | `THREAT_MODEL.md` に SEC-005 として grandchild reader thread leak 追加 |
| 4 | #6 ADR に統合 (作業重複避ける) |
| 11 | `artifacts.rs` の `EmbedKind`/`RerankerKind` doc-comment に "Sealed pattern" 追記 |
| 12 | (a) `Aggregator` doc に ascending + chunk_id 追記、(b) ADR 0004 Consequences に 1 行追加 |

### Drop (skip 4 件)

| # | Reason |
| - | ------ |
| 5 | YAGNI Boundary fail (call site 1 箇所のみ) |
| 7 | 既存 test enforcement で十分 |
| 8 | **bug** (decision ではなく orphaned identifier) |
| 10 | Cargo.toml inline comment + CI enforcement で十分 |

## Follow-up Hand-off

**Bug-fix issue 候補 (1 件)**: `#8` THREAT_MODEL.md に SEC-001 / SEC-002 を追加するか、コード側 (`model_probe.rs:99,258`、`model_probe/tests.rs:595`) の SEC-002 参照を削除する。判断は threat model owner に委ねるが、現状の「コードは SEC-002 を語るが文書は知らない」は無条件で潰すべき。

**ADR 起票候補 (`/adr` 経由) 3 件**:

- [ ] ADR: Apple Silicon Metal kernel NaN hazard — `MASK_FILL = -1e9` の選択理由と invariant
- [ ] ADR: `BUCKET_BOUNDS` cross-module invariant — 4 bucket 設計と `local_mask_cache` integrity
- [ ] ADR: `ensure_sqlite_vec` process-level extension registration ordering — public API contract

**comment/doc 補強 5 件 (downgrade)**:

- [ ] `THREAT_MODEL.md` に FORWARD maintenance section + SEC-005 (reader thread leak) 追加
- [ ] `artifacts.rs::EmbedKind`/`RerankerKind` の doc-comment に "Sealed pattern" 明記
- [ ] `retrieval.rs::Aggregator::aggregate` doc に "ascending" + chunk_id 二次キー追記
- [ ] ADR 0004 Consequences に sort key 順 (score desc, doc_id asc, chunk_id asc) 1 行追加

drop rate **33%** は scout dogfood の 16-29% より高め。各 drop に明確根拠あり (#5 YAGNI / #7,#10 既存 enforcement / #8 bug 扱い)、「決めかねて drop」は無し。
