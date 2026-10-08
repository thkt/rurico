#!/usr/bin/env bash
set -euo pipefail

# Filters could omit required smoke tests.
if (( $# != 0 )); then
    echo 'scripts/test.sh does not accept test filters or other arguments' >&2
    exit 2
fi

# Reuse built binaries so selection validation covers what runs.
metadata=$(mktemp -d)
trap 'rm -rf "$metadata"' EXIT
cargo metadata --locked --format-version 1 --features test-support,test-mlx,smoke > "$metadata/cargo.json"
cargo nextest list --locked --workspace --features test-support,test-mlx,smoke \
    --cargo-metadata "$metadata/cargo.json" \
    --profile ci --run-ignored default --list-type binaries-only --message-format json > "$metadata/binaries.json"
selection=(--cargo-metadata "$metadata/cargo.json" --binaries-metadata "$metadata/binaries.json" --profile ci --run-ignored default)
cargo nextest list "${selection[@]}" --message-format json > "$metadata/listing.json"
python3 scripts/verify-smoke-tests.py "$metadata/listing.json"
cargo nextest run "${selection[@]}" --no-tests fail --status-level all --final-status-level all
