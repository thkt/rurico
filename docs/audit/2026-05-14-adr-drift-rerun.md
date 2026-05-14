# ADR Drift Scan: 2026-05-14 (rerun)

Phase 1 (PR #170 ADR 0008-0010 起票) + Phase 2 (PR #171 Note/THREAT_MODEL/doc-comment 補強) merge 後の post-followup verification run。前回 audit `docs/audit/2026-05-14-adr-drift.md` で記録した 4 findings (H=0 / M=2 / L=2) が全て closure されているかと、新規 ADR 0008-0010 が code と整合しているかを `/audit-adr-drift` skill で再確認した記録。

## Summary

| Metric            | Value |
| ----------------- | ----- |
| ADRs scanned      | 10    |
| Drift findings    | 1     |
| H priority        | 0     |
| M priority        | 0     |
| L priority        | 1     |
| Unverifiable ADRs | 0     |

前回比: 4 findings → 1 finding。M priority 全消解 (drift findings #2 / #4 が closure)、L priority も 1 件のみ (ADR 0008 wording drift)。

## Prior findings closure check

| Prev # | ADR  | Description                                  | Closure path                                                            |
| ------ | ---- | -------------------------------------------- | ----------------------------------------------------------------------- |
| 1 (L)  | 0001 | `pub(crate)` 可視性の Note 不在              | ✓ PR #171 で ADR 0001 に Note (2026-05-14, visibility clarification) 追記 |
| 2 (M)  | 0002 | `gpu_pool_and_normalize` `pub fn` rationale stale | ✓ PR #168 で `pub(crate) fn` に narrow + ADR 0002 に Note 追記 |
| 3 (L)  | 0002 | FR-001a `is_finite` guard 拠点記述 stale     | ✓ PR #168 で ADR 0002 に Note (2026-05-14, FR-001a defense-in-depth update) 追記 |
| 4 (M)  | 0004 | Stage 4/5 amici 移行クロージャ未記録          | ✓ PR #171 で ADR 0004 に Note (2026-05-14, post-ADR-0006 stage ownership clarification) 追記 |

すべて intended path で closure 済。

## Per-ADR Findings

### ADR 0001-0007: Prior-audit ADRs

Status: Accepted (0001 / 0002 / 0004 / 0005 / 0006 / 0007), Superseded by amici (0003)

**No new drift findings.** 前回 finding は上記 closure check の通り解消。public API surface (`MatchFtsQuery` / `prepare_match_query` / `WeightedRrf` 等) と既存 ADR Decision は整合。

### ADR 0008: Adopt process-level sqlite-vec auto-extension registration

Status: Accepted (新規)

| # | File:Line                  | Description                                                                                                          | Direction  | Priority |
| - | -------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- | -------- |
| 1 | `src/storage.rs:22-44`     | ADR Decision Outcome は "downstream は `Connection::open` 前に一度だけ呼ぶ" と表現するが、実装は `OnceLock` で **idempotent** (`subsequent calls are no-ops`)。"一度だけ" は読者によっては "2 回呼ぶと失敗" と誤読され得る。Decision の意図 (`Connection::open` 前である必要、複数回呼んでも害なし) と整合する文言へ Note 追加が望ましい | adr-update | L        |

その他 symbol は ADR 通り:
- `pub fn ensure_sqlite_vec` (`src/storage.rs:22`) ✓
- `tracing::error!(rc, ...)` failure path (`src/storage.rs:29`) ✓
- `ensure_sqlite_vec_idempotent` test が idempotent property を pin (`src/storage.rs:47`) ✓
- README 案内 (L142-L145) と `Connection::open` 前順序 ✓

### ADR 0009: Adopt BUCKET_BOUNDS as cross-module forward-pass invariant

Status: Accepted (新規)

**No drift findings.**

- `BUCKET_BOUNDS = [128, 512, 2048, MAX_SEQ_LEN]` `pub(crate) const` (`src/model_io.rs:41`) ✓
- `assign_bucket` panic semantics (`src/model_io.rs:49-54`) ✓
- 3 caller (`src/embed/mlx.rs::forward_sub_batch` / `embed_query_truncated`、`src/reranker/mlx.rs::forward_sub_batch`) が `assign_bucket` 経由で round up している事実は import (`src/embed/mlx.rs:13`, `src/reranker/mlx.rs:4`) でも grep 確認 ✓
- `local_mask_cache` ≤4 entries (`src/modernbert/model.rs:204-212`) が ADR と整合 ✓
- `TOKEN_BUDGET = 256_000` + `compute_sub_batch_size` formula (`src/model_io.rs:277-285`) + `compute_sub_batch_size_matches_formula_per_bucket` test (`src/model_io.rs:444-462`) ✓

### ADR 0010: Adopt MASK_FILL constant to avoid Metal kernel NaN

Status: Accepted (新規)

**No drift findings.**

- `MASK_FILL = -1e9` (`src/modernbert/model.rs:384`) ✓
- `prepare_4d_attention_mask` で `MASK_FILL` 使用 (`src/modernbert/model.rs:423`) ✓
- `get_local_attention_mask` が `f32::NEG_INFINITY` を直接書き込み (`src/modernbert/model.rs:467`)、隣の comment (L465-466) で "mask 乗算経由しないため `0 * -inf` ハザード対象外" を明記 ✓
- ADR 0010 Pros and Cons table が 2 通りの fill 値 (production global mask vs local mask) の使い分けを文書化 ✓

`src/embed/mlx.rs:1023` の test fixture `vec![0.0, 1.0, 2.0, f32::NEG_INFINITY]` は `split_pooled` の non-finite 検出 test 用、production path 外。ADR 0010 違反ではない。

## External ADR Dependencies

| # | File:Line | External ADR ref | Recommended action |
| - | --------- | ---------------- | ------------------ |
| — | —         | (検出なし)       | —                  |

`ADR-NNNN` / `adr-nnnn` パターン scan で外部 ADR への ref は不検出。ADR 0001-0010 すべて local。

## Follow-up Issue Candidates

H-priority drift なし。L-priority 1 件は軽量 follow-up:

- [ ] **ADR 0008 L**: Decision Outcome / Confirmation の「`Connection::open` 前に一度だけ呼ぶ」表現を「`Connection::open` 前に呼ぶ (`OnceLock` で idempotent、再呼び出し可)」へ clarify する Note 追加。priority L、ADR 自体は immutable rule 通り Note で対応 (本文書き換えなし)

## Closure note for #158 / #159 audit cycle

本 rerun を以て #158 (ADR drift audit) と #159 (undocumented decisions audit) の audit cycle は実質クローズ。検出された 4 + 12 candidates のうち keep/downgrade verdict 受領分は全て Phase 1 (PR #170) + Phase 2 (PR #171) + bug fix (PR #169) + visibility narrow (PR #168) で実コードに反映済。残るは ADR 0008 wording の cosmetic L 1 件のみ。

audit → fix → re-audit のループは scout dogfood 同様、verbalize → verify cycle として機能した。memory `[[adr-as-verify-trigger]]` 通り、ADR は「決定の宣言」ではなく「verify ループの起点」。
