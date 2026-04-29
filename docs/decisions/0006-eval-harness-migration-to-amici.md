# ADR 0006: Migrate Search-Quality Evaluation Harness from `rurico` to `amici`

- Status: Accepted
- Date: 2026-04-27
- Confidence: high. The cyclic-dependency constraint that justified inlining the reference composition in ADR 0003 does not apply to `amici` (`amici/Cargo.toml:14` already pins `rurico`); the production wiring that the harness mirrors lives in `amici` today; embedder and reranker primitives remain in `rurico` and are called by `amici` unchanged, so the regenerated baseline is expected to match the existing `rurico/tests/fixtures/eval/baseline.json` bit-identically modulo the per-metric tolerance envelope already established in ADR 0003.

## Context

ADR 0003 placed an inline reference composition inside `rurico/src/eval/pipeline.rs` to evaluate `rurico` primitives in a `recall`-shape hybrid pipeline without forming a `rurico → recall` cyclic dependency. Since ADR 0003 was accepted, three signals shifted the calculus:

1. **`amici` is the production wiring host**. `amici/src/storage/fts.rs:16` owns `clean_for_trigram` (the FTS5 trigram-tokenizer adapter); `amici/Cargo.toml:6` describes itself as "Shared model-loading, storage helpers, and CLI utilities for the sae/yomu/recall toolchain". `recall`, `sae`, and `yomu` depend on `amici`. The reference composition that ADR 0003 mirrors *is* the wiring living in `amici` today.

2. **Mirroring drives Goodhart's-law drift**. `rurico/src/eval/pipeline.rs:381-382` carries the comment "Logic mirrors `amici::storage::fts::clean_for_trigram`; inlined here to avoid pulling `amici` into rurico's dependency graph." Two consequences follow: (a) `clean_for_trigram` exists twice with the `MAX_COMBOS` guard (issue #71, commit `7592a82`) present only in the `rurico` mirror — `amici/src/storage/fts.rs:20-43` lacks the guard (tracked in `amici#24`); (b) when `amici`'s production wiring evolves, the harness baseline measures the mirror, not the production code path that downstream users actually run.

3. **`rurico`'s charter is "ruri-v3 + storage primitives"**. `rurico/README.md:9` defines the crate as the shared `ruri-v3` inference + sqlite-vec primitive base; the harness is *evaluation infrastructure for downstream consumers*, not part of the primitives. Hosting the harness in `rurico` keeps two responsibilities entangled: ruri-v3 backend correctness (where ADR 0002 already lives) and end-to-end retrieval quality (where ADR 0003 lives but production wiring is elsewhere).

The cyclic-dependency constraint that justified inlining (ADR 0003 §Context, "`recall` already depends on `rurico`...") does not apply to `amici`: `amici → rurico` is the existing direction (`amici/Cargo.toml:14`), so moving the harness to `amici` keeps the graph acyclic without inlining.

## Decision

Migrate the search-quality evaluation harness from `rurico` to `amici`. Concretely:

1. **Move to `amici`**:
   - `rurico/src/eval/{pipeline,baseline,fixture,metrics,query_normalize}.rs` → `amici/src/eval/`
   - `rurico/src/bin/eval_harness.rs` → `amici/src/bin/eval_harness.rs`
   - `rurico/tests/fixtures/eval/*.json` → `amici/tests/fixtures/eval/`
   - `rurico/tests/eval_smoke.rs` → `amici/tests/eval_smoke.rs`
   - `eval-harness` feature gate is recreated in `amici/Cargo.toml`
   - `rurico/justfile` `=== 検索評価ハーネス ===` recipes (`eval-baseline`, `eval-reverse`, `eval-baseline-variant`, `eval-compare`, `eval-evaluate`, `eval-verify`) are recreated in a new `amici/justfile`

2. **Drop the mirror**: the harness in `amici` calls `crate::storage::fts::clean_for_trigram` directly (after `amici#24` ships the `MAX_COMBOS` guard). The mirrored copy in `rurico/src/eval/pipeline.rs:383-415` is deleted along with the rest of `src/eval/`.

3. **Stay in `rurico`**:
   - `src/storage/` primitives (`MatchFtsQuery`, `prepare_match_query`, `normalize_for_fts`, `rrf_merge`)
   - `src/bin/mlx_smoke.rs` and `tests/mlx_smoke.rs` — primitive-level speed and numerical-equivalence harness (ADR 0002), not retrieval quality.

4. **ADR continuity**:
   - This ADR is the migration record.
   - ADR 0003 is updated to `Status: Superseded by amici/docs/adr/0002-evaluation-methodology.md` once the `amici` ADR lands (work tracked in `rurico#86`).
   - The `amici` ADR re-states the four-part methodology from ADR 0003 (reference composition, fixture corpus, statistical contract, wiring validation) without behavioural change — the *location* of the harness moves; the methodology does not.

## Options Considered

### Option A: Status quo (keep the mirror in `rurico`)

Pros:

- zero migration cost.
- ADR 0003 stays valid as written.

Cons:

- two implementations of `clean_for_trigram`. The `MAX_COMBOS` guard from issue #71 (`rurico/src/eval/pipeline.rs:35-45`) has not been ported to `amici/src/storage/fts.rs:20-43`; production callers (`sae/src/storage/search.rs`) inherit the OOM risk until `amici#24` is fixed manually.
- the harness baseline keeps measuring the mirror. When production wiring in `amici` evolves (e.g. retrieval ordering, FTS distribution), the mirror must be hand-synced or the baseline silently legitimises a stale composition.
- Goodhart's law: the metric measures what the harness wires, not what users run.

### Option B: Promote `clean_for_trigram` to a `rurico` storage primitive

Pros:

- single source of truth for the FTS5-trigram adapter.
- no migration of `src/eval/`.

Cons:

- `rurico`'s charter (`README.md:9`) widens to include FTS5-trigram-tokenizer-specific knowledge. The trigram tokenizer is one consumer of `MatchFtsQuery`; baking adapter logic into the primitive layer constrains future tokenizer choices in downstream crates.
- does not address the larger duplication: `rurico/src/eval/pipeline.rs` *as a whole* is a wiring mirror, not just `clean_for_trigram`. Solving one symbol leaves the inlined pipeline in place, so the Goodhart's-law concern from §Context (2) survives.

### Option C (chosen): Migrate the entire eval-harness to `amici`

Pros:

- collapses both duplications: the wiring mirror and the `clean_for_trigram` mirror go away in one move.
- the harness measures production wiring directly. Goodhart's-law concern is structurally eliminated.
- `rurico` reaches the charter stated in `README.md:9` (ruri-v3 + storage primitives only).
- `amici` becomes the single place where end-to-end retrieval quality is governed, aligned with its existing role as the `sae`/`yomu`/`recall` toolchain base.
- no cyclic dependency: `amici → rurico` is the existing direction.

Cons:

- `amici`'s scope expands to host the harness, fixtures, and binary. Mitigation: the harness is feature-gated (`eval-harness`), so default `amici` builds are unaffected.
- ADR 0003 must be marked Superseded and the equivalent methodology re-stated in `amici`.
- the migration is a non-trivial multi-issue effort (`amici#24` → `amici#27` → `rurico#86`).

### Option D: Extract a third crate `rurico-eval` depending on both

Pros:

- preserves `rurico` and `amici` boundaries cleanly without expanding either crate's charter.

Cons:

- adds a crate purely for harness hosting; `amici` already exists for shared retrieval infrastructure used by `sae`/`yomu`/`recall`.
- two-hop dependency for a single purpose is heavier than the problem warrants. Reassessment Triggers list the condition under which Option D is reconsidered.

## Consequences

Positive:

- `rurico`'s public surface narrows to ruri-v3 inference + storage primitives. `[features] eval-harness` is removed; `[[bin]] eval_harness` is removed; `tests/fixtures/eval/` ships with `amici` instead.
- `clean_for_trigram` exists in exactly one place after `amici#24` lands. Future changes to the FTS5-trigram adapter touch one file.
- The committed baseline measures the production composition that `recall`/`sae`/`yomu` users actually run.
- `mlx_smoke` (ADR 0002) and `eval_harness` (this migration) live in the crate that owns each responsibility — primitive backend in `rurico`, end-to-end pipeline in `amici`.
- ADR 0003's Reassessment Trigger ("`recall/src/hybrid.rs` changes its public shape") becomes "`amici`'s retrieval-pipeline call sites change shape", which is structurally easier to detect because both files live in the same crate.

Negative:

- One ADR move and one ADR re-state are required. ADR 0003's content survives in `amici/docs/adr/0002-evaluation-methodology.md`; the `Superseded by` link in ADR 0003 ties the history.
- The migration involves four issues (`amici#24`, `amici#27`, `rurico#86`, `sae#77`). Sequencing matters: `amici#24` must merge before `amici#27` (the harness in `amici` calls the guarded `clean_for_trigram`), and `amici#27` must complete before `rurico#86` (the new `amici` rev must satisfy the `eval-verify` gate before the `rurico` mirror is deleted).
- Downstream baseline regeneration: the baseline is regenerated in `amici` with the same embedder/reranker primitives. The metric drift envelope (per `MetricSpec::tolerance` from ADR 0003) must hold across the regeneration; the migration's acceptance gate (`amici#27`) requires bit-identical mrr/ndcg/recall/p50/p95 against the existing `rurico/tests/fixtures/eval/baseline.json` before the move is considered successful.
- `rurico` loses the ability to detect end-to-end retrieval-quality regressions standalone. Any `rurico` change that affects pipeline metrics (e.g. embedding-output drift, RRF helper change) is detected via `amici`'s `eval-verify` after a `rurico` rev bump in `amici/Cargo.toml`. CI for `rurico` PRs covers primitive contracts only; pipeline-quality CI moves to `amici`.

## Migration Plan

The four-issue sequence is tracked on GitHub:

1. **`amici#24`** — port the `MAX_COMBOS` guard from `rurico/src/eval/pipeline.rs:35-45` (commit `7592a82`) into `amici/src/storage/fts.rs:20-43`. Required before `amici#27` so the migrated harness calls a guarded `clean_for_trigram`. Independently fixes the production OOM inherited by `sae/src/storage/search.rs`.
2. **`amici#27`** — receive the eval-harness in `amici`. Move `src/eval/`, `src/bin/eval_harness.rs`, `tests/fixtures/eval/`, `tests/eval_smoke.rs`; create `amici/justfile`; create `amici/docs/adr/0002-evaluation-methodology.md` (re-states ADR 0003); regenerate `baseline.json` and assert bit-identical metrics against the rurico-side baseline before the issue closes. Blocked by `amici#24`.
3. **`rurico#86`** — delete the migrated artefacts from `rurico` (`src/eval/`, `src/bin/eval_harness.rs`, `[[bin]] eval_harness`, `eval-harness` feature, `tests/eval_smoke.rs`, `tests/fixtures/eval/`, `justfile` `=== 検索評価ハーネス ===` section); update ADR 0003 status to `Superseded by amici/docs/adr/0002-evaluation-methodology.md`; clean up the `chunk-test` recipe's `--features eval-harness` dependency. Blocked by `amici#27`.
4. **`sae#77`** — bump `sae/Cargo.toml`'s `amici` rev to pick up `amici#24`'s guard. Independent of the migration in scope; tracked in the same family because the OOM-fix sequencing is part of this Decision's prerequisites.

## Reassessment Triggers

- Embedder or reranker primitive changes in `rurico` affect score determinism beyond the `MetricSpec::tolerance` envelope → re-evaluate whether `amici`'s `eval-verify` can absorb the drift, or whether per-metric tolerance needs widening in the `amici` ADR.
- `amici` becomes large enough that its dependents (`recall`, `sae`, `yomu`) want to depend on a smaller `amici-core` crate without the harness → split the harness into `amici-eval` (revisit Option D under new constraints). The current sequencing keeps the harness opt-in via the `eval-harness` feature so this is not yet warranted.
- A second downstream toolchain (not under `amici`) needs end-to-end retrieval-quality evaluation against `rurico` primitives → re-evaluate whether the harness should sit in a neutral crate (Option D) rather than in `amici`.
- ADR 0003's Reassessment Triggers re-fire (mlx-rs major bump, fixture-category extension, reverse-fixture lower-bound exceeded, etc.) → handled in the new location (`amici/docs/adr/0002`); this ADR is unaffected.

## References

- ADR 0002 (GPU-side pooling embed — primitive backend reproducibility, stays in `rurico`)
- ADR 0003 (Search Quality Evaluation Methodology — to be Superseded as part of `rurico#86`)
- ADR 0004 (Retrieval and rerank pipeline contract for rurico)
- `amici#24` (clean_for_trigram OOM guard port)
- `amici#27` (eval-harness intake)
- `rurico#86` (eval-harness removal + rurico purification)
- `sae#77` (amici rev bump)
- `rurico/src/eval/pipeline.rs:381-382` (existing mirror comment)
- `amici/src/storage/fts.rs:16` (production `clean_for_trigram`)
- `amici/Cargo.toml:6,14` (amici charter and rurico dependency direction)
- `rurico/README.md:9` (rurico charter)
