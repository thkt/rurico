# Error Documentation Convention

This repository documents fallible public APIs in rustdoc close to the code.

## Rule

- Public functions and trait methods that return `Result` must include a `# Errors` section.
- Public functions that may terminate the process or otherwise have important non-`Err`
  control flow must document that in `# Process Behavior`.

## What `# Errors` should say

- Name the public error type when one exists, for example `EmbedError` or `SanitizeError`.
- Explain the failure conditions callers can act on.
- Say when the returned message is opaque or backend-generated and therefore not stable.
- Say when failures are intentionally downgraded to logging or fallback behavior instead
  of surfacing as `Err`.

## Scope

- Apply this convention to crate public API.
- Internal helpers do not need `# Errors` unless they are unusually subtle or widely reused.

## Stability

- Structured public error enums are part of the API contract.
- Opaque `String` and backend `Exception` messages are not part of the stable contract unless
  explicitly stated otherwise.
