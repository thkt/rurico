#!/usr/bin/env bash
set -euo pipefail

cargo check --locked --workspace --features test-support,test-mlx,smoke
python3 -m unittest discover -b -s scripts -p 'test_*.py'
bash scripts/test.sh
cargo test --locked --doc --workspace --features test-support,test-mlx,smoke
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo fmt -- --check
