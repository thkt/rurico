#!/usr/bin/env bash
set -euo pipefail

# Comparisons require a known Cargo invocation. Do not forward arbitrary args
# (notably inline --config settings that build.rs cannot observe).
if [[ $# -ne 0 ]]; then
  echo 'build-smoke: no arguments accepted; comparison build is locked release + smoke' >&2
  exit 1
fi
for wrapper in RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER CARGO_BUILD_RUSTC_WRAPPER CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER; do
  if [[ -n "${!wrapper:-}" ]]; then
    echo "build-smoke: $wrapper is unsupported for comparison provenance" >&2
    exit 1
  fi
done
cd "$(dirname "$0")/.."
RURICO_SMOKE_BUILD_INVOCATION=locked-release-smoke-v1 \
  cargo build --locked --release --features smoke --bin mlx_smoke
