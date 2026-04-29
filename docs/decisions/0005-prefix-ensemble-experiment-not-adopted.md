# ADR 0005: Phase 6 Prefix-Ensemble Experiment — Not Adopted

- Status: Accepted
- Date: 2026-04-27
- Confidence: high. Outcome falsified the hypothesis on a fixed fixture / model / mlx-rs revision; the *experimental* nature of the issue means a single empirical run against the locked adoption gate is the contract for closure (#70).

## Context

Issue #70 (Phase 6 in the parent #53 search-quality programme) asked whether ruri-v3's 1+3 prefix scheme can improve `rurico` retrieval when more than one query-side prefix is used. The status quo embeds queries with a single prefix (`QUERY_PREFIX = "検索クエリ: "`); ruri-v3 also exposes `SEMANTIC_PREFIX = ""` (semantic meaning) and `TOPIC_PREFIX = "トピック: "` (classification / clustering). A *prefix ensemble* embeds the same query under multiple prefixes, retrieves top-N independently, and fuses the rankings via RRF. This is conceptually attractive on heterogeneous query distributions (factoid + concept + notation queries mixed), but issue #70 was scoped explicitly as an **experiment Issue**: only adopted as a public API if measured improvements clear an adoption gate.

The constraints from #70 are:

- Use the existing Phase 1 eval harness (`src/eval/pipeline.rs` + `tests/fixtures/eval/`) — no separate benchmark infrastructure.
- Compare at least two prefix combinations against the single-prefix baseline.
- Record per-category metrics (the harness already groups by IR task category) and the structural embedding-inference cost per query (`= len(query_prefixes)`).
- Keep the experiment behind the existing `eval-harness` Cargo feature so no new public API is exported until adoption.
- Lock the adoption threshold *before* any number is captured so post-hoc bias cannot move the bar.

This ADR records the experiment, the result, and the decision so a future maintainer reading the parent issue does not need to re-run the experiment to learn the outcome.

## Decision

**The prefix-ensemble retrieval option is not adopted.** No new public API is exported from `rurico` for prefix variants, and no new `CandidateSource` enum variant is added. The closed-set `CandidateSource = { Fts, Vector }` posture frozen by ADR 0004 is retained.

The experimental implementation was committed transiently behind the `eval-harness` feature gate, the four comparison baselines were captured against the canonical fixture, and the implementation was reverted after the result invalidated the hypothesis. This ADR is the durable record; the throwaway code is not retained because the implementation cost on a future re-run (one private helper plus an argv flag, ~80 LOC of pipeline code excluding tests) is small relative to the noise of carrying dead experimental paths.

### Locked adoption gate (recorded before measurement)

A prefix-ensemble variant would have been **adopted** only when **all** of the following held against the baseline capture on the same fixture / model / mlx-rs revision:

| Axis | Threshold |
| --- | --- |
| `ndcg@10` | improvement ≥ +0.005 absolute (point estimate) |
| `recall@10` | non-regression: ≥ baseline − 0.005 |
| `mrr@10` | non-regression: ≥ baseline − 0.005 |
| `latency_p95_ms` | ≤ 2.5× baseline |
| Per-category | no category drops more than 0.02 on `ndcg@10` |

`+0.005 ndcg@10` was chosen as the smallest improvement the harness's bootstrap CI can resolve as more than noise on this fixture (cf. ADR 0003 § Reproducibility — half-width on per-category metrics already exceeds `0.10`, but the global metric stabilises around `0.005`). The 2.5× latency cap reflects "M=3 prefixes triple the embedding-inference count, but reranker dominates wall-clock so the multiplier should still land below 3×". The 0.02 per-category floor blocks variants that win on average while crippling a single IR task.

### Methodology

The harness ran four captures against `tests/fixtures/eval/{documents,queries,known_answers}.jsonl` under `cl-nagoya/ruri-v3-310m` (commit `18b60fb8`) + `cl-nagoya/ruri-v3-reranker-310m`, mlx-rs `0.25`:

```text
baseline               — single QUERY_PREFIX (production behaviour as of Phase 5)
query+semantic         — QUERY_PREFIX + SEMANTIC_PREFIX
query+topic            — QUERY_PREFIX + TOPIC_PREFIX
query+semantic+topic   — all three
```

Each variant fans out the query side: `M` prefixes ⇒ `M` independent `Embed::embed_text` calls ⇒ `M` ranked vec-similarity lists. An *inner-RRF* (same `rrf_k = 60.0` Stage 2 uses) collapses the variants into a single `Vec<Candidate>` tagged `CandidateSource::Vector` before Stage 2 fuses with FTS. Inner-RRF is necessary because emitting `M` separate `Vector` candidates per prefix would multiply the Vector source contribution by `M` and break the FTS / Vector `source_weights` semantics frozen by ADR 0004 + Phase 4 (#68). Document-side embedding cost is unchanged across variants — the corpus is encoded once at index time regardless of `query_prefixes`.

The reranker (`cl-nagoya/ruri-v3-reranker-310m`) was kept enabled across all variants — bypassing it for "pure Stage 1" comparison would be the Goodhart trap recorded in the project memory `feedback_no_artifact_metrics.md` (production-disagreeing metrics are not informative). Aggregation, query normalization, and merge config were held at their post-Phase-5 production defaults.

### Results

Global metrics (point estimates, 95% bootstrap CI omitted for readability — full snapshots were captured but not retained):

| variant | recall@5 | recall@10 | mrr@10 | **ndcg@10** | p50 ms | p95 ms | embed/q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6300 | 0.7415 | 1.0000 | **0.8928** | 387.0 | 493.0 | 1 |
| query+semantic | 0.6290 | 0.7361 | 1.0000 | 0.8922 | 462.0 | 536.0 | 2 |
| query+topic | 0.6154 | 0.7316 | 1.0000 | 0.8881 | 420.0 | 545.0 | 2 |
| query+semantic+topic | 0.6243 | 0.7379 | 1.0000 | 0.8909 | 432.0 | 471.0 | 3 |

Δ versus baseline:

| variant | Δ recall@5 | Δ recall@10 | Δ mrr@10 | **Δ ndcg@10** |
| --- | --- | --- | --- | --- |
| query+semantic | -0.0010 | -0.0054 | 0.0000 | **-0.0006** |
| query+topic | -0.0146 | -0.0099 | 0.0000 | **-0.0047** |
| query+semantic+topic | -0.0057 | -0.0036 | 0.0000 | **-0.0019** |

Per-category `ndcg@10` drops, signed:

| category | baseline | query+semantic | query+topic | all3 |
| --- | --- | --- | --- | --- |
| comparative | 0.8983 | -0.0040 | -0.0047 | -0.0024 |
| conceptual | 0.8697 | +0.0065 | -0.0061 | -0.0040 |
| definitional | 0.8886 | +0.0014 | -0.0058 | -0.0008 |
| factoid | 0.9185 | -0.0015 | -0.0011 | -0.0030 |
| howto | 0.8508 | -0.0012 | +0.0007 | 0.0000 |
| listing | 0.8763 | -0.0001 | -0.0019 | +0.0004 |
| troubleshooting | 0.9042 | -0.0062 | -0.0088 | -0.0050 |
| variant_notation | 0.9363 | +0.0005 | -0.0104 | -0.0002 |

The largest single-category drop is `query+topic` on `variant_notation` at -0.0104, well within the 0.02 floor.

### Why the experiment cannot succeed on this fixture

`mrr@10 = 1.0000` on every variant including baseline. The cross-encoder reranker places the first relevant document at rank 1 on every fixture query, so MRR is at the metric ceiling and Stage 1 retrieval improvements have no headroom to surface as MRR gains. `ndcg@10` differences come from rank-2..10 ordering only, which the reranker also re-decides — so the prefix ensemble's signal is washed out by the reranker's final scoring pass.

This pattern matches a prior project finding recorded in memory `project_reranker_saves_fts_recall.md`: *"FTS path threshold 100 vs 10k で metric 同じ。reranker 込みなら FTS cap は積極的に効かせて OK"* — once the cross-encoder is in the pipeline, retrieval-side knobs do not visibly move the eval baseline. Phase 6 inherits the same headroom problem.

### Latency observations (gate satisfied, but irrelevant)

`p95_ms` rose by 8.7% (`query+semantic`) and 10.5% (`query+topic`) versus baseline — well below the 2.5× cap. The `all3` variant ran 4.5% *faster* than baseline, dominated by run-to-run noise rather than structural cost. The 310m cross-encoder reranker so heavily dominates wall-clock that the 1×→3× embedding multiplier is invisible. The latency gate would have passed if the metric gate had passed — but no variant cleared the metric gate, so the latency observation only confirms the experiment is cheap to re-run on a new fixture.

## Options Considered

### Option A: Adopt prefix ensemble as a public API

Promote the experimental code to a public API (`pub fn embed_query_variants`, `CandidateSource::PrefixEnsemble{Semantic,Topic}` variants, exported `HybridSearchConfig` field for prefix list).

Pros:

- Aligns with the parent issue #53's *Idea 9* once a positive signal is observed.
- Lets downstream consumers (`recall`, `sae`, `yomu`) opt in for query distributions where the cross-encoder is bypassed or unavailable.

Cons (decisive):

- The committed adoption gate fails on the locked fixture. Adopting against a measured negative result would silently re-introduce post-hoc bias.
- `CandidateSource::PrefixEnsemble*` adds enum variants whose downstream-visible cost (pattern-match exhaustiveness across `recall`/`sae`/`yomu`) is unjustified by zero evidence of benefit.
- Adoption commits the inner-RRF parameter (`rrf_k`) shape into a public API, making future "decouple inner-RRF from Stage 2" refactors a breaking change.

### Option B (chosen): Not adopted; ADR records the experiment

Revert the experimental code to keep the production path single-prefix. Document the experiment, methodology, gate, results, and verdict in this ADR. Keep ADR 0004's closed `CandidateSource = { Fts, Vector }` enum unchanged.

Pros:

- Honours the locked adoption gate. The next maintainer who proposes prefix ensemble has a measured null result to argue against, not a reflexive "we never tried it".
- No breaking-change risk in `rurico`'s public surface.
- ADR carries the methodology + gate + numbers so a future re-run (different fixture, different reranker, no reranker) is a clean copy of the harness invocation, not a re-derivation of the experiment design.

Cons:

- Re-running the experiment on a different fixture or model requires re-implementing the inner-RRF helper (≈80 LOC of pipeline code excluding tests). The ADR's inline pseudocode and the per-prefix call shape make this a low-friction copy-paste, but it is not free.
- Discards the `tests/fixtures/eval/phase6/*.json` baseline files that recorded the comparison. The summarised tables in this ADR replace them; raw bootstrap CI is not retained.

### Option C: Keep the experimental code behind a Cargo feature, ungated by a `pub use`

Leave the implementation in `src/eval/pipeline.rs` behind the `eval-harness` feature, with no public API. Future re-runs would have a cheaper resurrection cost.

Pros:

- Re-run on a different fixture is a one-line argv change rather than a re-implementation.
- The single-prefix-equivalence test pins the routing contract so the inner code path stays correct without active maintenance.

Cons (decisive):

- Dead-experiment code accumulates if every Phase-N experiment lands ungated. The CLAUDE.md *"Don't add features ... beyond what the task requires"* posture applies — the experiment is closed, not on hold.
- The `eval-harness` feature is exercised by CI (`cargo test --features eval-harness`) so the dead path eats CI time forever.
- A second maintainer reading the code has to determine "is this live or experimental?" — the ADR plus a deleted path is unambiguous; a feature-gated path is not.

### Option D: Run a second experiment with the reranker bypassed

Re-run the four variants with the reranker disabled, on the assumption that prefix ensemble might help recall at Stage 1 even if the reranker eats the gain on this fixture.

Pros:

- Would isolate the prefix ensemble's effect from the reranker's washing-out behaviour.

Cons (decisive):

- The project memory `feedback_no_artifact_metrics.md` calls out exactly this anti-pattern: *"stage 単独 eval で見えない effect を visible にしようとするのは Goodhart's law、production と乖離した metric は意味なし"* — measuring a stage in isolation does not predict production behaviour because production runs the reranker.
- A bypass-mode improvement that does not survive the rerank pass cannot be promoted to a public API anyway, because the public pipeline (ADR 0004 Stage 4) includes rerank.

## Consequences

Positive:

- The prefix-ensemble option is closed for this fixture / reranker combination — no future PR re-litigates the design without supplying a different fixture or a different reranker shape.
- ADR 0004's frozen `CandidateSource = { Fts, Vector }` enum stays a closed set; downstream pattern-matches do not break.
- The project's "experiment, then decide" posture (#70 was an experiment Issue) gains a worked example: gate locked → run → record → close.

Negative:

- Future maintainers who want to re-test this on a different fixture must re-implement the inner-RRF helper. Mitigation: this ADR's *Methodology* section gives the algorithmic shape; the issue body of #70 also has the call shape.
- The four `tests/fixtures/eval/phase6/*.json` baseline files are not retained, so cross-checking the numbers in this ADR against a regenerated run requires re-running the captures (≈70s × 4 variants on Apple Silicon with the model cached).

## Reassessment Triggers

- The eval fixture gains queries where `mrr@10` falls below `1.0` (e.g. variants of the same canonical question, multiple equally-relevant docs without graded relevance hints) — the ceiling that masked the experiment goes away. Re-run with the same gate.
- The pipeline's reranker is removed or replaced with a less-dominant scorer (e.g. a smaller cross-encoder, a bi-encoder reranker) — Stage 1 retrieval gains headroom to surface in the final metric. Re-run.
- A multilingual or non-Japanese fixture lands where `SEMANTIC_PREFIX = ""` may behave qualitatively differently from `QUERY_PREFIX = "検索クエリ: "` (the prefix string is Japanese-anchored). Re-run with the new fixture; this ADR does not generalise to that case.
- A downstream (`recall` / `sae` / `yomu`) reports a query distribution where the production reranker visibly fails on a class of queries that prefix ensemble might rescue (e.g. very short / very long queries the reranker mis-scores). Re-run scoped to that class.

## References

- Parent issue #53 (search-quality programme, Phase 1–6)
- This issue #70 (Phase 6 prefix ensemble — experiment scope)
- ADR 0003 (`docs/decisions/0003-evaluation-methodology.md`) — eval harness methodology, bootstrap CI tolerance, fixture posture
- ADR 0004 (`docs/decisions/0004-retrieval-and-rerank-pipeline-contract-for-rurico.md`) — `CandidateSource` closed-enum frozen at `{ Fts, Vector }`; Stage 1 source-variant placeholder for Phase 6 (now closed by this ADR)
- Memory `project_reranker_saves_fts_recall.md` — prior observation that retrieval-side knobs are washed out by the cross-encoder
- Memory `feedback_no_artifact_metrics.md` — reranker-bypass evaluation as Goodhart-law anti-pattern
- ruri-v3 prefix scheme: <https://huggingface.co/collections/cl-nagoya/ruri-v3>
