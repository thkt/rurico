default:
    @just --list

# Standard check, including model-free smoke tests
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

# Chunk retrieval tests without model inference (MLX build required)
chunk-test:
    cargo nextest run --lib chunk

# Capture embed fixtures in tests/fixtures/phase2_baseline
embed-capture:
    cargo run --bin mlx_smoke --features smoke --release -- capture-fixture

# Measure embed speed / padding / R² baseline (stderr diagnostics)
embed-baseline:
    cargo run --bin mlx_smoke --features smoke --release -- measure-baseline

# Verify embed fixture numerical parity
embed-verify:
    cargo run --bin mlx_smoke --features smoke --release -- verify-fixture

probe-embed:
    cargo run --bin probe_embed_smoke --release

probe-reranker:
    cargo run --bin probe_reranker_smoke --release

# Both probes in sequence
probe: probe-embed probe-reranker
