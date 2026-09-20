#!/usr/bin/env bash
set -euo pipefail

cargo check --locked --workspace --features test-support,test-mlx
cargo nextest run --locked --workspace --features test-support,test-mlx --profile ci
cargo test --locked --doc --workspace --features test-support,test-mlx
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo fmt -- --check
