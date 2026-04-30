# ADR 0007: Library Logging Boundary — `rurico` Emits Internal Recovery Warns, CLI Emits Caller Boundary Warns

- Status: Accepted
- Date: 2026-04-30
- Confidence: high. The boundary is observable from the existing call graph: `yomu/src/query.rs:487-493` already warns on `embed_query` failures (caller boundary); `rurico::embed::Embedder::forward` callers are the only consumers that decide whether a typed error becomes `degraded=true`, so a `rurico`-side warn at that point would duplicate. The internal-recovery sites (`split_pooled` non-finite output, `mlx_cache` lock poisoning, probe subprocess timeout, etc.) are decisions `rurico` makes that the caller cannot see, so a `rurico`-side warn is the only place where the diagnostic information exists.

## Context

Issue [#89](https://github.com/thkt/rurico/issues/89) surfaced that operators of downstream CLIs (`yomu`, `sae`, `recall`) cannot diagnose why `degraded=true` is returned, because `rurico`'s `log::warn!` calls are silently dropped: downstream initialises `tracing_subscriber::fmt()` without `LogTracer`, so events from the `log` crate never reach the subscriber, and the EnvFilter (`yomu=warn` / `sae=info` / `recall=warn`) excludes the `rurico` target even after the `log → tracing` migration.

Once `rurico` migrates to `tracing` and downstream filters include `rurico=warn` (handled in `amici#36`, `yomu#149`, `sae#85`), the question becomes: at *which* sites should `rurico` emit `warn!`? Two failure modes need to be avoided:

1. **Silent drop (the original problem)** — internal recovery decisions invisible to the caller, so `degraded=true` cannot be triaged.
2. **Duplicate warn (the new risk)** — if `rurico` warns at every typed-error return site, callers that already warn at *their* boundary (e.g. `yomu/src/query.rs:487-493`) emit two log lines for one event, forcing operators to deduplicate.

The boundary that distinguishes the two is whether `rurico` is making an internal recovery decision the caller cannot observe.

## Decision

`rurico` warns when it makes an internal recovery decision; the caller (CLI) warns at its boundary when it converts a typed error into a degraded mode.

| Site type | Owner of `tracing::warn!` | Example |
|---|---|---|
| Internal recovery (rurico decides to truncate, recover from poison, give up after retries, reject corrupt output) | `rurico` | `embed/mlx.rs::split_pooled` rejecting non-finite output; `mlx_cache.rs` recovering from lock poisoning; `model_probe.rs` timing out and killing subprocess |
| Caller boundary (caller converts typed error into `degraded=true` / `DegradedReason::*` / fallback path) | CLI (yomu / sae / recall) | `yomu/src/query.rs:487-493` warning when `embed_query` fails before falling back to text search; `yomu/src/tools.rs:525-535` (yomu#131) warning when `embed_query` failure becomes `DegradedReason::ProbeFailed` |

Concrete classification for `rurico` 0.2.x:

### `rurico` warns (internal recovery)

- `embed.rs::truncate_for_query` — query exceeds `max_seq_len`, truncating
- `embed/metrics.rs::PhaseMetrics::log` — per-call debug summary (debug, not warn)
- `embed/mlx.rs::shrink_chunk_to_fit` — chunk cannot fit after adaptive shrink
- `embed/mlx.rs::split_pooled` — `BufferShapeMismatch` (corrupt readback) and `NonFiniteOutput` (kernel overflow / corrupt weights)
- `mlx_cache.rs::clear_inference_cache` — lock poison recovery, FFI cache-clear non-zero exit
- `modernbert/model.rs::forward` — `seq_len` exceeds `max_seq_len`, defense-in-depth truncation
- `model_probe.rs::wait_with_timeout` / `collect_pipe` — probe subprocess timeout, drain failure, channel disconnect
- `reranker/mlx.rs::truncate_pair` — pair exceeds `max_seq_len`, truncating
- `reranker/mlx.rs::score_batch` — output shape mismatch, non-finite scores

### `rurico` does not warn (caller boundary or noise)

- `embed/mlx.rs:178,352` `forward(...).map_err(EmbedError::inference)?` — caller (`yomu/src/query.rs:487-493`) already warns; `rurico` warn would duplicate.
- `EmbedInitError::backend` — init-time failure surfacing as binary error; the binary's error handler prints it.
- `artifacts.rs::WrongModelKind` / `MissingFile` / `InvalidConfig` / `InvalidTokenizer` — caller decides whether the absence is fatal or fallback (`DegradedReason::ProbeFailed` in yomu#131); caller responsibility.
- `model_probe.rs::ProbeError::*` returned from `probe_via_subprocess` — caller decides whether the probe failure becomes `degraded=true`; caller responsibility.

### Field discipline

`tracing::warn!(error = %e, "<context>")` when wrapping an `Error` value. Otherwise, name structured fields after the relevant local (snake_case); prefer `expected` / `actual` for shape mismatches and a `*_secs` suffix for durations.

## Options Considered

### Option A (chosen): Sharp library/CLI boundary by recovery semantics

Pros:

- Each warn line maps to exactly one event. Operators can grep without deduplication.
- Compatible with the existing degraded-reason machinery in CLIs (`amici::model::embedder::DegradedReason`); rurico does not replicate the reason mapping, so the abstraction stays one-directional.
- The classification is decidable per call site by asking "does the caller know?" — internal recovery is invisible to the caller; typed errors are visible.

Cons:

- Requires reviewers to remember the boundary on new code. Mitigated by this ADR being the explicit reference and by the `rurico=warn` directive in `amici::logging::init_subscriber` (amici#36) making the boundary observable in tests.

### Option B: `rurico` warns at every typed-error return site

Pros:

- Mechanical rule: any `?`/`map_err` over an internal error gets a warn.

Cons:

- Duplicate warns at every caller that already wraps the error (`yomu/src/query.rs:487-493`, future `sae` / `recall` callers). Operators see N+1 lines per event.
- "Library is too noisy" leads operators to filter `rurico=error` (or worse, suppress `rurico` entirely), defeating the point of the migration.
- Conflates init-time failures (which the binary's error handler already surfaces) with runtime degradation.

### Option C: `rurico` never warns; CLIs must reconstruct context from typed errors

Pros:

- Minimal coupling between rurico and CLIs — rurico is "pure".

Cons:

- The original silent-drop problem returns: internal recovery decisions (e.g. lock poison recovery, probe timeout kill, non-finite output rejection) are not exposed via the typed error chain. The decision is invisible.
- Forces every CLI to instrument the same internal recovery sites (e.g. wrap every `embed_query` call with telemetry that re-derives "the kernel produced a non-finite output"). High duplication in CLI code.
- The N+1 `yomu/sae/recall` problem from issue #89 reasserts: each downstream re-implements the diagnostic.

## Consequences

Positive:

- The `rurico` warn surface is documented and bounded. New rurico contributors can decide warn placement without re-deriving the boundary.
- Downstream filter directives (`amici::logging::init_subscriber` adds `rurico=warn` per amici#36) collect exactly the diagnostic events a `degraded=true` operator needs, without per-call-site instrumentation.
- yomu#131's caller-boundary warns and rurico#89's internal-recovery warns compose without duplication.

Negative:

- The boundary requires judgement on borderline cases (e.g. `EmbedError::Inference` returned from a tokenizer encoding error in `plan_long_document`: is the tokenizer error caller-observable? In this codebase yes, so `rurico` does not warn). Reviewers must apply the rule.
- `tracing` becomes a public-API-shape concern for `rurico`: removing or renaming a structured field is a downstream-grep-breaking change. Mitigated by the structured-field naming convention listed under Decision §Field discipline.
- `rurico` Cargo.toml gains a `tracing` runtime dependency and a `tracing-subscriber` dependency for the `mlx_smoke` binary. The library code itself never initialises a subscriber (LANG.md compliance); only the smoke binary does.

## Reassessment Triggers

- A new caller pattern emerges where typed errors do *not* propagate to a place the caller can warn (e.g. async runtime that drops errors). The boundary may need to shift toward Option B for that subsystem only.
- Downstream `tracing` filtering at `rurico=warn` produces audibly noisy logs in production (more than ~1 line per `degraded=true` event). The internal-recovery surface is too broad and needs trimming, or the field discipline needs tightening (e.g. spans instead of repeated warns).
- A future `rurico-eval` or other consumer crate sits between `rurico` and the CLIs. The boundary applies recursively: the consumer crate becomes a "caller" relative to `rurico` and an "internal" relative to its CLI; each level applies the same rule.

## References

- Issue [thkt/rurico#89](https://github.com/thkt/rurico/issues/89) — originating issue with full Fix-site classification (A/B/C/D scope)
- Issue [thkt/yomu#131](https://github.com/thkt/yomu/issues/131) — caller-boundary warns in yomu (RC-001 — SIL-001/002/008 + OPS-001)
- Issue [thkt/amici#36](https://github.com/thkt/amici/issues/36) — `init_subscriber` adds `rurico=warn` directive
- Issue [thkt/yomu#149](https://github.com/thkt/yomu/issues/149) — yomu migrates to `amici::logging::init_subscriber`
- Issue [thkt/sae#85](https://github.com/thkt/sae/issues/85) — sae migrates to `amici::logging::init_subscriber`
- LANG.md `Logging` section — `tracing` over `log`, structured fields, no subscriber init in library code
- ADR 0006 — eval-harness migration to `amici` (similar boundary concern: who owns what)
