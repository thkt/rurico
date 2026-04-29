# ADR 0001: Adopt a Typed FTS Query Contract in `rurico`

- Status: Proposed
- Date: 2026-03-28
- Confidence: high - The current API contract is ambiguous, and the migration path across downstream crates is clear.

## Context

`rurico::storage::sanitize_fts_query` currently returns a plain `String`. That leaves three contract gaps:

1. Callers cannot tell whether the returned string is always safe to pass to FTS5 `MATCH`.
2. Empty or fully stripped queries can still surface as runtime FTS syntax errors in downstream crates.
3. `fts_expand_short_terms` still accepts raw `&str`, so sanitization is not enforced by the type system.

Recent review found concrete failure modes:

- operator-like terms such as `NOT` may be interpreted inconsistently across callers
- malformed `NEAR(...)` input can collapse to an unusable query
- different downstream crates (`recall`, `sae`, `yomu`) handle the edge cases inconsistently

## Decision

Adopt a typed FTS query contract in `rurico` and make `rurico` responsible for producing only validated query values for FTS5 `MATCH`.

The storage API will move toward these types:

- `SanitizedFtsQuery`
- `MatchFtsQuery`
- `SanitizeError`

The first step is:

1. Change `sanitize_fts_query` from `String` to `Result<SanitizedFtsQuery, SanitizeError>`.
2. Define `SanitizeError` with two variants:
   `EmptyInput`
   `NoSearchableTerms`
3. Preserve operator-like keywords (`AND`, `OR`, `NOT`) as literal terms during sanitization.
4. Change `fts_expand_short_terms` to accept `&SanitizedFtsQuery` instead of raw `&str`.
5. Return a typed value for final `MATCH` usage, either via `MatchFtsQuery` or a helper such as `build_match_fts_query`.

The intended responsibility split is:

- `rurico`: sanitize and build safe FTS query values
- downstream crates: decide whether to skip search, return empty results, or show a user-facing message

## Options Considered

### Option A: Keep `String` and document caller-side empty checks

Pros:

- no breaking API change
- smallest local diff

Cons:

- keeps the contract ambiguous
- repeats empty checks in every downstream crate
- still allows raw unsafe strings to reach `MATCH`

### Option B: Return `Option<String>`

Pros:

- prevents `MATCH ''`
- smaller API change than a richer error model

Cons:

- loses why sanitization failed
- does not distinguish empty input from no searchable terms
- still leaves the final query as an untyped `String`

### Option C: Return `Result<SanitizedFtsQuery, SanitizeError>` and type the pipeline

Pros:

- makes the contract explicit
- gives downstream crates enough signal to handle failures consistently
- allows `fts_expand_short_terms` and later helpers to require typed inputs

Cons:

- breaking API change
- requires coordinated updates in downstream crates

## Consequences

Positive:

- `rurico` becomes the single source of truth for FTS query safety
- downstream crates stop relying on duplicated local sanitizers
- runtime FTS syntax errors from empty or stripped queries are reduced
- literal `AND` / `OR` / `NOT` remain searchable across downstream crates

Negative:

- `recall`, `sae`, and `yomu` must be updated in lockstep
- this introduces a short migration window across dependent crates

## Migration Plan

1. Update `rurico` storage API and tests.
2. Migrate `recall` from local sanitization to `rurico`.
3. Migrate `sae` from local sanitization to `rurico`.
4. Add sanitization to `yomu` before short-term expansion.
5. Run `cargo test` and `cargo clippy --all-targets --all-features -- -D warnings` in all affected crates.

## Reassessment Triggers

- A downstream crate needs more detailed failure causes than `EmptyInput` and `NoSearchableTerms`
- FTS query construction grows beyond sanitize plus expansion into a true parser/validator
- another storage backend with different query semantics is added
