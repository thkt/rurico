# ADR 0004: Retrieval and Rerank Pipeline Contract for `rurico`

- Status: Proposed
- Date: 2026-04-26
- Confidence: medium. The 5-stage shape is empirically validated by `src/eval/pipeline.rs` (Phase 1c) and mirrors the production pipeline in `recall/src/search.rs`. The unknown is whether the Phase 3 aggregation step fits cleanly between merge and rerank for downstream crates whose hybrid wiring evolved independently (`sae`, `yomu`).

## Context

Issue #53 plans a six-phase retrieval-quality programme. Phase 1 (#65, ADR 0003) delivered the evaluation harness; Phases 3–6 will implement aggregation (#67), hybrid weights (#68), query normalization (#69), and prefix ensemble (#70). Issue #66 (this ADR) freezes the **interfaces** Phases 3–6 will plug into.

Three contract gaps must be resolved before Phase 3 can start:

1. **`rurico` ships primitives, not a pipeline.** `src/storage/search.rs:150,191` exposes `rrf_merge` and `prepare_match_query`; `src/embed/embedder.rs` exposes the `Embed` trait; `src/reranker.rs` exposes the `Rerank` trait. Each downstream (`recall`, `sae`, `yomu`) re-composes them by hand, so a high-recall/rerank-aware "standard pipeline" cannot be reused. Issue #53 Idea 6 names this explicitly: *reranker should be a first-class pipeline*.

2. **Two parallel implementations of the same shape already exist.** Phase 1c committed `src/eval/pipeline.rs` (ADR 0003 Option C: a recall-shaped reference composition gated behind `eval-harness`). Phase 1's commitments were scoped to evaluation, but the file implements precisely the FTS+vec+RRF+rerank shape Phase 2 needs to standardise. Without a stated relationship between this file and the production module, Phase 3 will either duplicate or diverge.

3. **No agreed plug-in point for the aggregation step (Phase 3).** Idea 4 (document-level aggregation) and Idea 10 (MMR) both need a hook between candidate merge and rerank. Without freezing the location and surface in this ADR, each Phase 3 PR will re-litigate the design.

The decision must be a **contract-level** ADR: Phase 2 ships no pipeline implementation. It commits the interfaces, the file location, the migration order, and the cross-phase hook shapes — leaving values (RRF k, weight vector, normalization rules, prefix set) to Phases 3–6.

## Decision

Adopt a five-stage public retrieval pipeline contract for `rurico`:

```
candidate retrieval → candidate merge → aggregation → rerank → final
   (Stage 1)            (Stage 2)        (Stage 3)    (Stage 4)  (Stage 5)
```

The contract is documented here; the implementation lands in Phase 3 by promoting `src/eval/pipeline.rs` to `src/retrieval.rs` and dropping the `eval-harness` feature gate from the production-shared types. Existing primitives (`prepare_match_query`, `rrf_merge`, `Embed`, `Rerank`, `RankedResult`) remain unchanged; the new API is additive.

### Pipeline contract

Each stage names the input/output types, the default behaviour, and the Phase-3+ extension hook. Type signatures are sketches, not final names — Phase 3 implementation may rename without superseding this ADR as long as the shape is preserved.

#### Stage 1: Candidate retrieval

- Input: `query: &str`, `top_n: usize`, optional `CandidateSource` filter.
- Output: `Vec<Candidate { source: CandidateSource, doc_id: String, score: f64, rank: usize }>` per source.
- `CandidateSource` is a closed enum: `Fts`, `Vector`. Closed-enum keeps misspelled labels a compile error (same posture as ADR 0003's `MetricSpec`). ~~Phase 6 may add `PrefixEnsemble` variants.~~ **Resolved 2026-04-27 (ADR 0005)**: Phase 6 (#70) ran the prefix-ensemble experiment, no variant cleared the locked adoption gate, and the closed enum stays at `{ Fts, Vector }`.
- The `top_n` semantics inherit recall's `limit * 3` heuristic: retrieve the best 3× the final cutoff per source so RRF has headroom. The multiplier is a default constant, not a contract guarantee — Phase 4 may override.

#### Stage 2: Candidate merge

- Input: `candidates: &[Candidate]` from multiple sources.
- Output: `Vec<MergedHit { doc_id: String, score: f64, source_scores: HashMap<CandidateSource, f64> }>`.
- Default: RRF with `k = 60.0` (current `rrf_merge` constant).
- Phase 4 hook: `MergeStrategy` trait OR `MergeConfig { rrf_k: f64, weights: HashMap<CandidateSource, f64> }`. Phase 2 fixes only the **shape** — the trait-vs-config decision is deferred to Phase 4 (#68) so it can be made against measured data.
- `source_scores` is preserved through Stage 2 onward so downstream UIs can display score breakdown. RRF currently discards this — the new wrapper preserves it.

#### Stage 3: Aggregation (Phase 3 implements)

- Position: between Stage 2 (merge) and Stage 4 (rerank). Locking the position now lets Phase 4–6 wire around it without re-arranging.
- Input: `Vec<MergedHit>` at pre-aggregation granularity (whatever Stage 1 emits).
- Output: `Vec<MergedHit>` at the granularity Stage 4 expects.
- Phase 3 hook: a single `Aggregator` trait with `fn aggregate(&self, hits: &[MergedHit]) -> Vec<MergedHit>`. Default impl is the **identity** (no aggregation, mirrors current Phase 1 behaviour where one `EvalDocument` = one `Hit`).
- Granularity decision deferred to Phase 3: whether Stage 1 retrieves chunks (`Candidate.doc_id` carries `chunk_id`) or whole documents is itself a Phase 3 design decision (#67 / Idea 3 parent-child retrieval). This ADR commits only to *where* aggregation happens, not *what* it aggregates over. If Phase 3 chooses chunk-level retrieval, `Candidate` gains a `chunk_id: Option<String>` field — non-breaking because the field is optional.
- Specific strategies (max/avg/RRF-by-chunk/MMR/dedupe) are out of scope for Phase 2 — Issue #67 picks them.
- Trait > closed enum: aggregation strategies are user-extensible (downstream can add domain-specific dedupers); the merge-strategy decision is deferred because RRF-vs-weighted is a closed conceptual set.

#### Stage 4: Rerank

- Input: `query: &str`, `hits: &[MergedHit]`, `corpus: impl Fn(&str) -> Option<&str>` for body lookup.
- Output: `Vec<RerankedHit { doc_id: String, score: f64, source_scores: HashMap<CandidateSource, f64>, reranker_score: Option<f32>, original_rank: usize }>`.
- The existing `Rerank::rerank(&self, &str, &[&str]) -> Vec<RankedResult>` trait stays untouched. The wrapper preserves `source_scores` from Stage 2 and adds `reranker_score` + `original_rank` for downstream presentation.
- `reranker: Option<&dyn Rerank>` — when `None`, Stage 4 is the identity (passes Stage 3 output through). Mirrors `eval/pipeline.rs::evaluate`'s current signature.
- `RankedResult` itself is **not** extended in Phase 2: extending the existing struct risks breaking downstream pattern-matching across `recall`/`sae`/`yomu`. The new `RerankedHit` wrapper carries metadata at the pipeline level; the underlying trait stays narrow.

#### Stage 5: Final

- Input: `Vec<RerankedHit>`.
- Output: `Vec<FinalHit>` after `top_k` cutoff.
- `FinalHit` is `RerankedHit` + presentation-only fields (e.g. snippet hooks). Implementation detail of Phase 3; Phase 2 only commits that the type exists and carries `source_scores` + `reranker_score`.

### Backward compatibility

| Existing API                 | Phase 2 commitment          |
| ---------------------------- | --------------------------- |
| `prepare_match_query`        | Unchanged                   |
| `rrf_merge<K>`               | Unchanged (used internally) |
| `Embed` trait + impls        | Unchanged                   |
| `Rerank` trait + impls       | Unchanged                   |
| `RankedResult { index, score }` | Unchanged (no field added)  |
| `recency_decay` (rurico, `src/storage/search.rs:11`) | Unchanged (primitive stays in rurico) |
| `apply_recency_boost` (recall, `recall/src/hybrid.rs`) | Unchanged (boost composition stays downstream-owned) |

Recency boost stays in `recall::hybrid` (downstream-owned) for now. Lifting it into the rurico pipeline is a candidate for Phase 4 once weight tuning produces evidence; doing it in Phase 2 would commit to a recency-included shape before measuring whether all downstream consumers want it.

### Cyclic-dependency posture

Inherited from ADR 0003: `recall` is the structural reference for pipeline shape; `rurico → recall` edges are forbidden because `recall → rurico` already exists (`recall/Cargo.toml:19`). Phase 3 implementation cross-checks against `recall/src/search.rs` manually, not through a code dependency. `recall/src/hybrid.rs` shape changes go on the Phase-3+ PR description's review checklist.

## Options Considered

### Option A: Stay primitives-only (status quo)

Continue exposing `prepare_match_query`, `rrf_merge`, `Embed`, `Rerank` and let downstream re-compose.

Pros:

- Zero code change in Phase 2
- No breaking-change risk
- Each downstream picks its own pipeline shape

Cons:

- Issue #53 Idea 6 explicitly names "first-class pipeline" — this option contradicts it
- The existing `src/eval/pipeline.rs` becomes an orphan reference whose relationship to production is undefined
- Phase 3 has nowhere to put the aggregation hook; each PR redefines it
- recall/sae/yomu maintainers continue duplicating the FTS+vec+RRF+rerank wiring

### Option B: New `rurico::retrieval` module parallel to `src/eval/pipeline.rs`

Create a fresh production module mirroring the eval pipeline shape; keep eval/pipeline.rs as-is behind its feature gate.

Pros:

- Clean separation between evaluation pipeline and production pipeline
- Eval-only types (e.g. `EvalDocument`, `EvalQuery`) stay isolated from the public API
- Phase 1 deliverables are immutable

Cons:

- Code duplication: the FTS+vec+RRF+rerank wiring exists twice
- Drift risk: a Phase 4 fix to one pipeline silently misses the other; the eval baseline becomes meaningless if the two diverge
- Test asset duplication: `src/eval/pipeline.rs` already has unit tests against `MockEmbedder` + `MockReranker` (T-011); rewriting them under a new module costs effort with no quality gain

### Option C (chosen): Promote `src/eval/pipeline.rs` to `src/retrieval.rs`

Phase 3 renames the file, drops the `eval-harness` feature gate from production-shared items, and extends the API per the contract above. Eval-specific types (`EvalDocument`, `EvalQuery`, fixture loaders) stay in `src/eval/`.

Pros:

- Zero implementation duplication: production and eval share the same code path
- Eval baseline meaningfully measures the production pipeline (rather than a parallel construction)
- Existing Phase 1 tests (T-011 + smoke fixtures) carry over without rewriting
- Removes the orphan-module problem in Option A

Cons:

- Production code temporarily lives under `src/eval/pipeline.rs` until Phase 3 renames it — one ADR cycle of awkward naming
- Cross-repo work (recall/sae/yomu adopting the new API) becomes follow-up issues, per the existing cross-repo issue-routing rule

### Option D: Lift the entire `recall` hybrid pipeline (recency + filters + snippets) into `rurico`

Phase 2 freezes a contract that includes recency boost, project/source filters, and snippet extraction.

Pros:

- One pipeline call covers the full UX recall provides today
- Maximum reuse for sae/yomu

Cons:

- Filters are application-specific (`recall` filters by `sessions.project`/`source`; `sae`/`yomu` have different schemas). Lifting them creates a leaky abstraction
- Snippet extraction depends on FTS5 `snippet()` and the column layout of the host schema — not generalisable
- Recency shape (half-life, weight) is a Phase 4 hyperparameter, not a Phase 2 contract concern
- Locks in decisions Phase 4–6 should make against measured data

## Consequences

Positive:

- Phase 3 (#67) starts with the aggregation hook position frozen; aggregation strategy choice is the only open question.
- Phase 4 (#68) starts with merge configuration's shape committed; only weight values need to be picked.
- Phase 5 (#69) and Phase 6 (#70) gain explicit insertion points (Stage 1 query normalization, Stage 1 source variants).
- Eval and production pipelines stop being parallel candidates — the eval baseline measures the same code Phase 3+ ships.
- Downstream consumers (recall/sae/yomu) have a single migration target rather than per-Phase API churn.

Negative:

- Phase 3's first PR carries the rename + feature-gate-drop work in addition to aggregation implementation. If that bundle becomes unwieldy, Phase 3 splits into 3a (rename/gate-drop) and 3b (aggregation).
- The `RerankedHit` / `MergedHit` / `FinalHit` type triplet adds weight to the public API surface. Phase 3 may collapse them to a single `Hit { stage: Stage, score: f64, source_scores: ... }` if the triplet feels redundant in code review — the contract permits this collapse as long as `source_scores` and `reranker_score` are preserved.
- Cross-repo adoption is gated by individual repo issues (per existing rule). Adoption velocity depends on those repos' throughput.

## Migration Plan

1. **Phase 2 (this ADR)**: commit `adr/0004-retrieval-and-rerank-pipeline-contract-for-rurico.md`, append the row to `adr/README.md`. No changes to `src/`. **Done 2026-04-26 (PR #74).**
2. **Phase 3** (Issue #67, **implemented 2026-04-26**): introduce `MergedHit` and the `Aggregator` trait at the position fixed in Stage 3, plus four impls (`IdentityAggregator`, `MaxChunkAggregator`, `DedupeAggregator`, `TopKAverageAggregator`). New surface lives in `src/retrieval.rs` with no feature gate. `eval/pipeline.rs::evaluate` takes an `&impl Aggregator` parameter; the harness dispatches by `aggregation=identity|max-chunk|dedupe|topk-average` argv. `BaselineSnapshot` gains `aggregation: String` (serde-default `"identity"` so pre-Phase-3 files round-trip).
   - **Pipeline-shape finding**: `rrf_merge` (`src/storage/search.rs:154`) fuses on `doc_id` via `HashMap`, so its output is already unique. Combined with one-chunk-per-`EvalDocument` indexing, every non-identity aggregator is **structurally identical** to identity on the current eval baseline. Strategy correctness is validated via synthetic multi-hit unit tests in `src/retrieval.rs`; non-vacuous baseline evaluation arrives once chunk-level retrieval lands (item 7 below).
   - **Deferred from this ADR's wording**: full `eval/pipeline.rs` → `src/retrieval.rs` rename. The Aggregator surface is what Phase 3+ needs; the reference composition stays in `eval/pipeline.rs` until a downstream consumer needs to share more of the wiring.
3. **Phase 4** (Issue #68, **implemented 2026-04-27**): expand Stage 2 with both `MergeStrategy` trait *and* `HybridSearchConfig` closed config. `WeightedRrf` is the default impl; `merge_with_recency` extends it for the recency boost (caller injects `Fn(&str) -> Option<f64>` for `updated_at` lookup, so storage-side metadata stays downstream-owned). `MergedHit` gains `source_scores: HashMap<CandidateSource, f64>` (serde-default empty), `Candidate { source, doc_id, score, rank }` lands as the Stage 1 boundary type, and `BaselineSnapshot.merge_config` round-trips capture-time config to verify-baseline. `eval_harness` accepts `rrf_k=`, `fts_weight=`, `vector_weight=` kvs and ships a new `compare-baselines paths=...` subcommand that emits a markdown matrix of variant metrics.
   - **Trait + config hybrid**: ADR Phase 2 left "trait OR config" deferred. Phase 4 implements both — the trait abstraction is the plug-in point for future learned-weight strategies, while `HybridSearchConfig` keeps weight tuning ergonomic. Avoids the reassessment trigger entirely (closed-config-too-narrow case).
   - **Naming**: Phase 2's placeholder `MergeConfig` (Stage 2 contract sketch) became `HybridSearchConfig` in implementation to align with downstream `recall::hybrid` terminology. The shape (`rrf_k: f64`, `source_weights: HashMap<CandidateSource, f64>`) matches Phase 2's contract; only the type name changed.
   - **Recency posture**: Recency stays *opt-in* via `merge_with_recency` and an injected age accessor, matching the pre-Phase-4 "downstream-owned" stance. `RecencyConfig` lives in `src/retrieval.rs` so all crates can share the math, but the rurico pipeline does not query metadata directly. Lifting recency into `evaluate()` is gated on the chunk-level retrieval follow-up (item 7) supplying `updated_at` in the eval fixture without invalidating the committed `baseline.json` fixture_hash.
   - **Deferred from this Migration Plan's wording**: variant baseline fixtures (fts-heavy, vec-heavy, k-low, k-high). Generation requires MLX (Apple Silicon) and is a one-time developer run via `capture-baseline rrf_k=... fts_weight=... output=...`; results compare via the new `compare-baselines` command. Recommended-default JSON is committed alongside the variant baselines once measurements exist.
4. **Phase 5** (Issue #69): insert query normalization at Stage 1 entry. Default off; opt-in via config to preserve current behaviour.
5. ~~**Phase 6** (Issue #70): add `CandidateSource::PrefixEnsemble` variants and the embedding-side fan-out logic.~~ **Resolved 2026-04-27 (ADR 0005)**: Phase 6 (#70) experiment compared four prefix combinations against baseline; no variant cleared the `+0.005 ndcg@10` gate (the cross-encoder reranker put `mrr@10` at ceiling on every variant, leaving no headroom for Stage 1 retrieval to surface). No public API added; `CandidateSource` stays closed at `{ Fts, Vector }`.
6. **Cross-repo follow-ups**: file individual issues against `recall`, `sae`, `yomu` after the first `rurico` bump that exposes `src/retrieval.rs`. Each downstream adopts at its own pace; `rurico` does not change `recall`/`sae`/`yomu` directly (cross-repo issue rule, ADR 0001 migration pattern).
7. **Chunk-level retrieval follow-up** (Issue #76, prerequisite for non-vacuous aggregation evaluation): extend `ChunkedEmbedding` with `chunk_id` metadata, switch the reference pipeline to chunk-level indexing, add the parent-child helper, and capture per-strategy baselines that reflect actual ranking changes. Touches `recall`/`sae`/`yomu` schema, so it ships under its own issue rather than #67.

## Reassessment Triggers

- Phase 3 finds that aggregation belongs **before** merge (chunk-level dedupe pre-RRF) or **after** rerank (post-rank diversity) instead of between them → supersede this ADR with the corrected stage order.
- ~~Phase 4 weight tuning shows that `MergeConfig` (closed) cannot express the configurations that move metrics → expand to `MergeStrategy` trait and document the trigger in a follow-up ADR.~~ **Resolved 2026-04-27**: Phase 4 ships both — `MergeStrategy` trait (extension point) plus `WeightedRrf { config: HybridSearchConfig }` (default closed config). The trigger no longer needs to fire because the trait already exists.
- A second downstream (`sae` or `yomu`) wants recency boost in the rurico pipeline → revisit Option D's "lift recency" question with measured demand.
- `recall/src/search.rs` introduces a stage absent from this contract (e.g. learned weights, multi-tier rerank) → manual diff at next Phase 3+ PR; if the new stage looks generalisable, supersede.
- The `MergedHit` / `RerankedHit` / `FinalHit` triplet feels redundant in Phase 3 implementation review → collapse to a single `Hit { stage, score, source_scores, reranker_score }` (permitted under "shape preserved" clause above; record the collapse in the Phase 3 PR description).
- A breaking change to `Rerank` or `Embed` becomes necessary (e.g. async rerank for batched GPU pools) → new ADR; this ADR's "primitives unchanged" clause is the explicit constraint that decision violates.

## References

- Parent issue #53 (search-quality programme, Phase 1–6)
- Phase 2 issue #66 (this ADR's deliverable)
- Phase 3 issue #67 (aggregation), Phase 4 #68 (weights), Phase 5 #69 (normalization), Phase 6 #70 (prefix ensemble)
- ADR 0001 (`adr/0001-typed-fts-query-contract.md`) — typed FTS query primitive
- ADR 0003 (`adr/0003-evaluation-methodology.md`) — Phase 1 reference composition + cyclic-dep posture
- `src/eval/pipeline.rs` — Phase 1c reference composition (Phase 3 promotion target)
- `src/storage/search.rs:146-198` — `rrf_merge`, `prepare_match_query` primitives
- `src/embed/embedder.rs` — `Embed` trait surface
- `src/reranker.rs:202-219` — `Rerank` trait + `RankedResult`
- `recall/src/search.rs:352-429` — downstream hybrid pipeline (structural reference)
- `recall/src/hybrid.rs` — recency boost helper (downstream-owned for now)
- `recall/Cargo.toml:19` — `recall → rurico` git-ref dependency (cyclic constraint)
