# Threat Model

`rurico` is a library for ruri-v3 (ModernBERT) model loading and inference via
the MLX backend on Apple silicon. This document records the threat model for
the `model_probe` subprocess pathway and enumerates the risks that the
project explicitly accepts.

## Scope

The probe pathway spawns a child process via `Command` (fork+exec) to
validate model artifacts before main-process load. The child receives a
filtered environment (see `FORWARD` in `src/model_probe.rs:301`) plus the
`__RURICO_PROBE_*` triple naming the model / config / tokenizer paths. The
child canonicalizes the paths and verifies they are component-wise
descendants of the HuggingFace cache root, then exits with a status code
that the parent maps back into `ProbeStatus` or `ProbeError`.

## In-scope threats

| Threat                                                     | Mitigation                                                                                                            |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Environment injection bypassing `__RURICO_PROBE_*`         | `Command::env_clear` plus explicit `FORWARD` allowlist (`src/model_probe.rs:301-343`)                                 |
| Path traversal via probe env (e.g. `../../../etc/passwd`)  | `validate_probe_paths_with_cache` canonicalizes and rejects paths outside the cache root (`src/model_probe.rs:87-101`) |
| Symlink redirection to attacker-controlled targets         | Canonicalize-then-component-wise `starts_with` check rejects targets resolving outside the cache root                  |
| Unbounded probe runtime                                    | 30s timeout (`PROBE_TIMEOUT`) followed by SIGKILL via `wait_with_timeout`                                              |

## Out-of-scope / Accepted Risks

The risks below are visible in the codebase but are intentionally not
mitigated. They are listed here so that future maintainers do not silently
"fix" them without first revisiting the trade-off.

### SEC-003: Probe exit code as filesystem oracle

`validate_probe_paths_with_cache` (`src/model_probe.rs:87-101`) returns
distinct setup-phase exit codes:

| Code | Constant                          | Meaning                                                |
| ---- | --------------------------------- | ------------------------------------------------------ |
| 3    | `PROBE_EXIT_ENV_INCOMPLETE`       | Required `__RURICO_PROBE_*` triple incomplete          |
| 4    | `PROBE_EXIT_CANONICALIZE_FAILED`  | A candidate path does not resolve (typically nonexistent) |
| 5    | `PROBE_EXIT_PATH_OUTSIDE_CACHE`   | A candidate path resolves outside the cache root       |
| 6    | `PROBE_EXIT_CACHE_ROOT_INVALID`   | The cache root itself cannot be canonicalized          |

An adversary that can invoke the probe child can therefore observe whether a
given absolute path exists (code 4 vs 5) and whether it lives under the HF
cache root (code 5 vs 0). This is a filesystem-existence oracle.

**Accepted because:**

- Invoking the probe child requires the ability to launch the same binary
  with controlled env vars; an attacker with that capability already has
  local code execution and can call `stat(2)` / `readdir(2)` directly.
- Collapsing the codes into a single "rejected" status would erase
  observability that is load-bearing for incident triage and test
  diagnostics (each code maps to a distinct `setup_label` in
  `src/model_probe.rs:164-172`).
- No additional information is leaked beyond what the OS already exposes
  to a local attacker.

### SEC-004: DYLD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH in `FORWARD`

The probe child inherits `DYLD_LIBRARY_PATH` and
`DYLD_FALLBACK_LIBRARY_PATH` from the parent process via the `FORWARD`
allowlist (`src/model_probe.rs:328-330`). A local attacker who can set
these env vars in the parent process before a probe is spawned can redirect
dynamic library lookups in the child.

**Accepted because:**

- MLX resolves its Metal kernel and accelerator dylibs through the macOS
  dynamic linker. Stripping `DYLD_LIBRARY_PATH` /
  `DYLD_FALLBACK_LIBRARY_PATH` breaks development workflows (`cargo test`,
  local MLX builds) and packaged distributions that ship MLX as a
  side-loaded framework.
- The threat already requires the attacker to control the parent process
  environment. In that scenario the entire process tree is compromised
  regardless of how the probe child handles env forwarding — `LD_PRELOAD`,
  `DYLD_INSERT_LIBRARIES`, and arbitrary code execution in the parent are
  all available without going through the probe.
- The `FORWARD` allowlist intentionally narrows the inherited surface
  compared to the parent's full env. High-risk injection vectors not
  required by MLX (notably `DYLD_INSERT_LIBRARIES`) are excluded.

## Wire format commitments

The probe pathway exposes a small surface that downstream tooling (incident
triage, log aggregation, monitoring) is allowed to grep. These are pinned
contracts — changes are visible to consumers and need a deprecation cycle.

### SEC-002: Probe setup-rejection stderr message format

When the probe child rejects a path / env-vars during the setup phase, it
writes `"probe setup rejected: <label>"` to stderr and exits with the
matching `PROBE_EXIT_*` code. `<label>` is sourced from
`SetupReason::label()` (`src/model_probe.rs:100-108`) and is one of:

| Label                       | Exit code                          |
| --------------------------- | ---------------------------------- |
| `env incomplete`            | `PROBE_EXIT_ENV_INCOMPLETE` (3)    |
| `path canonicalize failed`  | `PROBE_EXIT_CANONICALIZE_FAILED` (4) |
| `path outside cache`        | `PROBE_EXIT_PATH_OUTSIDE_CACHE` (5)  |
| `cache root invalid`        | `PROBE_EXIT_CACHE_ROOT_INVALID` (6)  |

**Pinned by:** `setup_reason_label_is_stable_per_variant` (T-115-D,
`src/model_probe/tests.rs:597`) — label drift breaks the test.

**Why a commitment, not an accepted risk:** the stderr text is the only
mechanism for an external observer to distinguish the four setup-phase
rejection causes (the exit code triple SEC-003 already accepts is the
oracle for "rejected vs. ok"; this anchor lets it be triaged further).
A future maintainer who renames a label must update both the label table
above and any downstream log-grep tooling — the rename is a breaking
change to the wire.

> **Note (2026-05-14, audit clarification)**: the
> `docs/audit/2026-05-14-undocumented-decisions.md` report referred to
> "SEC-001 / SEC-002 enumeration" but `SEC-001` is not a real identifier —
> SEC-002 is the only orphaned anchor. This section closes the orphan.

## Reporting Vulnerabilities

For potentially sensitive issues, contact the maintainers privately rather
than opening a public issue. Otherwise, file a GitHub issue with the
`security` label.
