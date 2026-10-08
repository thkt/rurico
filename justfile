# rurico harness shortcuts. Run `just` (or `just --list`) for the menu.
#
# Sections: Development checks (check/test/lint/fmt), chunk-level retrieval
# tests, mlx_smoke (embed speed + numerical parity, ADR 0002), and probe smoke
# (subprocess probe contract).

default:
    @just --list

# === Development ===

# Standard Rust check + smoke selection checks + clippy + fmt --check
check:
    bash scripts/check.sh

test:
    python3 -m unittest discover -b -s scripts -p 'test_*.py'
    bash scripts/test.sh
    cargo test --locked --doc --workspace --features test-support,test-mlx,smoke

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt-check:
    cargo fmt -- --check

fmt:
    cargo fmt

# Run only the chunk-level retrieval tests (T-076-001..008, MLX-free)
chunk-test:
    cargo nextest run --lib chunk

# === embed スピード + 数値同等性ハーネス (mlx_smoke, ADR 0002) ===

# Capture embed fixture → tests/fixtures/embed/{w1,w2,w3}.bin
embed-capture:
    cargo run --bin mlx_smoke --features smoke --release -- capture-fixture

# Measure embed speed / padding / R² baseline (stderr diagnostics)
embed-baseline:
    cargo run --bin mlx_smoke --features smoke --release -- measure-baseline

# Verify embed fixture parity (cosine_min ≥ 0.99999, max_abs_diff ≤ 1e-5)
embed-verify:
    cargo run --bin mlx_smoke --features smoke --release -- verify-fixture

# === Probe smoke (subprocess probe contract) ===

probe-embed:
    cargo run --bin probe_embed_smoke --release

probe-reranker:
    cargo run --bin probe_reranker_smoke --release

# Both probes in sequence
probe: probe-embed probe-reranker
