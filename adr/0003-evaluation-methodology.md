# ADR 0003: Search Quality Evaluation Methodology for `rurico`

- Status: Accepted
- Date: 2026-04-25
- Confidence: medium-high. Reference composition pattern is empirically established in `recall/src/hybrid.rs`; statistical significance via bootstrap CI on 140+ query fixtures is a well-known IR convention; the unknown is whether mlx inference f32 drift across machines / mlx-rs versions stays inside the regeneration tolerance already adopted by ADR 0002.

## Context

Issue #53 splits search-quality improvement into Phase 1〜6, where Phase 1 builds the evaluation harness used by all later phases to detect regressions and validate improvements. Issue #65 is the Phase 1 implementation issue.

Three contract gaps must be resolved before implementation:

1. **`rurico` does not ship a hybrid search**. `src/storage/search.rs:150,191` exposes `rrf_merge` and `prepare_match_query` as **primitives only**. The actual hybrid pipeline (schema, vec0 SQL, FTS↔vec join, rerank wiring) lives in downstream crates: `recall/src/search.rs`, `recall/src/hybrid.rs`, `sae`, `yomu`. A baseline taken on a harness-internal pipeline is not a baseline of `rurico` itself unless the relationship between the two is named explicitly.

2. **Per-category breakdown on small per-category sample sizes is not informative**. Bootstrap CI on graded relevance `nDCG@10` with 5 queries per category sits around `±0.25〜0.30`, wider than realistic per-Phase improvements. A breakdown table that always overlaps the gate threshold cannot drive decisions.

3. **Mutation testing via output shuffle alone is weak**. Shuffle confirms only that the metric responds to ranking change; it does not catch wiring bugs such as "query embedding swapped with document embedding", "RRF arguments reversed", or "k cutoff off-by-one". A passing shuffle test in the presence of these bugs would silently legitimise an incorrect baseline.

`recall` (the most-developed downstream of `rurico`) is the natural reference for the hybrid pipeline shape, but `recall` already depends on `rurico` (`recall/Cargo.toml:19`), so `rurico → recall` would form a cyclic dependency.

## Decision

Adopt a four-part evaluation methodology and document it as the contract under which Phase 1 lands and Phase 3〜6 operate:

1. **Reference composition**: a `recall`-inspired hybrid pipeline implemented inline inside `src/bin/eval_harness.rs`. Pipeline shape mirrors `recall/src/hybrid.rs` + `recall/src/search.rs` (FTS5 with `prepare_match_query`, vec0 cosine retrieval, RRF merge with `rrf_merge`, optional rerank with `Rerank::rerank`). It is **not** `recall` itself — the harness owns its schema, indexer, and orchestration. Phase 3〜6 results on this composition predict but do not guarantee `recall` outcomes.

2. **Fixture corpus and queries shipped in-repo**: `tests/fixtures/eval/` holds documents (≥50), queries (≥140, distributed across 7 categories at ≥20 each), and known-answer fixtures. Documents are sourced from openly licensed material so the repo can carry them. No live API access at evaluation time.

3. **Statistical contract**: every reported metric (Recall@k, MRR@k, nDCG@k for k ∈ {5, 10, 50}) carries a 95% bootstrap confidence interval computed over n=1000 resamples with a fixed RNG seed. Per-category breakdowns are emitted for exploratory inspection but the regression gate uses **global metrics only**; per-category numbers below the CI-decision threshold are flagged `uninformative` in `baseline.json`.

4. **Wiring validation via known-answer fixtures**: alongside the production fixture, three deterministic micro-fixtures verify that the harness pipeline is wired correctly:
   - `identity_ranker.jsonl` — every query's relevance map points at a single doc whose body is a verbatim copy of the query; expected `nDCG@10 = 1.0` and `Recall@1 = 1.0`.
   - `reverse_ranker.jsonl` — relevance is inverse-correlated with corpus order; pipeline output should produce nDCG near the lower bound for the fixture (see _Reverse fixture lower-bound protocol_ below).
   - `single_doc.jsonl` — corpus has one document and the relevance map matches it; expected `Recall@1 = MRR = 1.0`.

   These three are evaluated alongside the shuffle mutation test in `tests/eval_smoke.rs`. Together they cover query/doc mix-ups, RRF argument order, and k-cutoff off-by-one.

### Reverse fixture lower-bound protocol

The reverse fixture cannot use a hand-picked threshold because the true lower bound depends on `nDCG@10` semantics over the specific relevance distribution of the committed `reverse_ranker.jsonl`. The protocol pins this:

- During Phase 1e, run `eval_harness capture-reverse-baseline` once. The harness ranks the corpus in reverse order (worst-relevance-first) using only the relevance map, computes `nDCG@10`, and writes the value to `tests/fixtures/eval/reverse_baseline.json` as `observed_lower_bound`.
- That JSON file is committed to git and treated as immutable. Subsequent runs of the smoke assertion (T-014) load `observed_lower_bound` from the file and assert that the pipeline-produced `nDCG@10` does not exceed `observed_lower_bound × 1.05`.
- The 5% slack absorbs mlx f32 reduction drift (bounded by NFR-001) and the small numerical difference between the harness pipeline (with embedder + reranker) and the pure-relevance reverse ranking.
- If the smoke assertion ever exceeds the slack on a clean run, that is a Reassessment Trigger: the reverse fixture is regenerated and `reverse_baseline.json` is overwritten in a dedicated PR; the trigger condition is logged in the PR description so the protocol stays auditable.

The reproducibility contract follows ADR 0002:

- mlx inference f32 drift is bounded by `cosine_similarity ≥ 0.99999 ∧ max_abs_diff ≤ 1e-5` between regeneration runs.
- `eval_harness capture-baseline` emits the canonical baseline; `eval_harness verify-baseline` re-runs and asserts the tolerance.
- `baseline.json` carries `model_id`, `model_revision`, `mlx_rs_version`, `fixture_hash` to detect drift drivers explicitly.
- `fixture_hash` is FNV-1a 64-bit over the three JSONL files (`documents.jsonl`, `queries.jsonl`, `known_answers.jsonl`); SHA-256 is intentionally avoided to keep the dependency graph small. Hash collisions are acceptable since the hash is a fixture-changed signal, not a security primitive.
- The fixture corpus is synthetic paraphrase of publicly licensed source documentation (MIT / Apache 2.0 / BSD / MPL 2.0 / CC0 / CC-BY / W3C Software Notice / IETF Trust / Python license / PostgreSQL license / public domain). AS-005's enumeration of MIT / Apache / CC0 / CC-BY is read as a non-exhaustive permissive whitelist; equivalent permissive licenses are accepted under the same intent. `tests/fixtures/eval/LICENSES.md` records the per-source attribution and acknowledges share-alike upstream sources whose topics were authored independently.

## Options Considered

### Option A: Per-component evaluation only (no hybrid composition)

Evaluate `Embedder::embed_query` retrieval (cosine top-k) and `Reranker::rerank` separately; never compose.

Pros:

- evaluates `rurico`'s actual surface area
- simpler harness, no SQLite or RRF involvement

Cons:

- Phase 4 (RRF weight tuning, parent issue Idea 5) cannot be measured because RRF is part of the composition, not the primitives
- Phase 6 (prefix ensemble, parent issue Idea 9) requires query-side composition to test
- the user-perceived metric (hybrid retrieval quality) is what Phase 3〜6 try to improve

### Option B: Add `recall` as a pinned git dependency, evaluate end-to-end

Pros:

- zero pipeline duplication
- regression detection covers actual `recall` behaviour

Cons:

- `recall` already depends on `rurico` → cyclic dependency
- breaks `rurico` build standalone

### Option C (chosen): Reference composition inline + in-repo fixture

Pros:

- avoids the cyclic dependency
- harness is self-contained: clone `rurico`, run `eval_harness`, get baseline
- pipeline shape stays close to `recall` so Phase 3〜6 results are predictive
- shipping the fixture in-repo guarantees reproducibility across machines and over time

Cons:

- requires manual maintenance when `recall/src/hybrid.rs` changes shape
- baseline measures the harness composition, not `recall` itself — this must be communicated whenever the baseline is cited

### Option D: Standalone reference, no downstream relationship at all

Same as C but without naming `recall` as the structural reference.

Pros:

- simpler ADR
- no dependency on `recall` evolution

Cons:

- the baseline becomes a fully synthetic benchmark — Phase 3〜6 improvements have no defensible link to user-facing search quality
- review feedback "what does this baseline measure?" has no convincing answer

## Consequences

Positive:

- Phase 1 produces a baseline whose meaning is explicit: `rurico` primitives composed in a `recall`-shape pipeline against a fixed in-repo fixture.
- Phase 3〜6 changes are gated on a statistically meaningful global metric improvement (CI non-overlap with baseline), not on point-estimate movement.
- Wiring bugs in the harness are caught by the three known-answer fixtures before they corrupt the committed baseline values.
- Reproducibility across machines is bounded by the same f32 tolerance ADR 0002 already uses.
- Per-category breakdown remains available for exploratory analysis without polluting the gate decision.

Negative:

- `recall/src/hybrid.rs` evolution is not auto-tracked. A manual diff is required at each Phase 3〜6 cycle, recorded in the relevant phase's PR description.
- Fixture authoring (≥140 queries, ≥50 documents, all categories balanced) is human-effort-intensive. Single-annotator subjectivity is accepted for Phase 1 with a Reassessment Trigger to introduce a second annotator if Phase 4 disagreement appears.
- The harness binary adds ~1500-2500 lines of Rust that must be lint-clean under `rurico`'s strict pedantic clippy profile.
- `baseline.json` cannot literally claim "this is `recall`'s search quality"; downstream users of the baseline must read the methodology note in `docs/eval/baseline.md`.

## Migration Plan

1. **Phase 1a** (commit b8c761e): implement `src/eval/metrics.rs` (Recall@k, MRR@k, nDCG@k, bootstrap CI). Pure functions, unit-tested without mlx.
2. **Phase 1b** (commit 3ad0766): implement `src/eval/fixture.rs` types and JSONL loaders. Author the in-repo fixture: 60 documents (7 domains), 147 queries (7 IR task categories × 21), 3 known-answer micro-fixtures, `LICENSES.md`.
3. **Phase 1c** (commit 9088dbd): implement `src/eval/pipeline.rs` (recall-inspired composition) and `src/eval/baseline.rs` (JSON + markdown serialisation, `UNINFORMATIVE_HALF_WIDTH = 0.10`).
4. **Phase 1d** (commit f2b3ce3): implement `src/bin/eval_harness.rs` (mlx_smoke pattern, modes: `evaluate`, `capture-baseline`, `capture-reverse-baseline`, `verify-baseline`) and `tests/eval_smoke.rs` (subprocess-driven assertions, T-013/014/015/016/017/019). Gate behind `[features] eval-harness = []` so default `cargo test` skips the evaluation module entirely (FR-018); opt-in via `cargo test --workspace --features eval-harness -- --ignored` (FR-019).
5. **Phase 1e**: run `eval_harness capture-baseline` and `eval_harness capture-reverse-baseline` on Apple Silicon with `ruri-v3-310m` cached; commit `tests/fixtures/eval/baseline.json`, `tests/fixtures/eval/reverse_baseline.json`, and `docs/eval/baseline.md` (`## Methodology`, `## Per-category breakdown`, `## Reproducibility`).
6. **Phase 1f**: finalise this ADR with implementation-confirmed details and add the corresponding row to `adr/README.md`.

## Reassessment Triggers

- `recall/src/hybrid.rs` changes its public shape (signature change of `hybrid_search` / equivalent) → re-evaluate whether the inline composition still tracks recall closely enough.
- Phase 4 weight tuning produces statistically significant improvement on the harness composition but no perceived improvement when downstream users (`recall`, `sae`, `yomu`) deploy a `rurico` bump → introduce per-component evaluation alongside the composition baseline.
- Disagreement on any individual fixture's relevance map between two annotators exceeds 20% (kappa < 0.6) → re-author the fixture with explicit relevance-judgment guidelines and add a second annotator for new queries.
- mlx-rs major version bump → regenerate baseline; if `cosine_similarity < 0.99999` the tolerance must be re-evaluated.
- A new fixture category becomes necessary (e.g. multilingual, code-only) and existing 7 categories no longer cover Phase 3〜6 evaluation needs → add via fixture extension; bump `fixture_hash` to invalidate prior baselines.
- The reverse fixture smoke assertion (T-014) exceeds `observed_lower_bound × 1.05` on a clean run → regenerate the reverse fixture and overwrite `reverse_baseline.json` in a dedicated PR; log the cause (algorithm change, fixture rebalance, etc.) in the PR description.

## References

- Parent issue #53 (Phase 1〜6 search quality improvement)
- Phase 1 issue #65 (this ADR's deliverable)
- `recall/src/hybrid.rs` (reference pipeline shape)
- `recall/src/search.rs:363,393` (RRF + scoring composition)
- `rurico/src/storage/search.rs:150,191` (primitives `rrf_merge`, `prepare_match_query`)
- `rurico/src/embed/embedder.rs` (`Embed` trait surface)
- `rurico/tests/mlx_smoke.rs` (subprocess test pattern, reused)
- `rurico/adr/0002-gpu-side-pooling-embed.md` (regeneration tolerance and probe-bin pattern, reused)
