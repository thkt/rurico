# rurico harness shortcuts. Run `just` (or `just --list`) for the menu.
#
# Two harness families live in this crate:
# - 検索評価 (eval_harness, ADR 0003 / Issue #65): retrieval quality metrics.
# - embed スピード + 数値同等性 (mlx_smoke, ADR 0002): MLX inference SLA + fixture parity.
#
# All recipes are read-only against `tests/fixtures/eval/*.json` unless they
# end in `*-baseline` / `*-reverse` (those overwrite committed fixtures).

default:
    @just --list

# === Development ===

# Full local CI: cargo test + clippy + fmt --check
check: test lint fmt-check

test:
    cargo test --workspace --all-features

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt-check:
    cargo fmt -- --check

fmt:
    cargo fmt

# Run only the chunk-level retrieval tests (T-076-001..008, MLX-free)
chunk-test:
    cargo test --lib --features eval-harness chunk

# === 検索評価ハーネス (eval_harness, ADR 0003) ===

# Recapture identity baseline → tests/fixtures/eval/baseline.json (MLX required)
eval-baseline:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-baseline aggregation=identity \
      output=tests/fixtures/eval/baseline.json

# Recapture reverse baseline → tests/fixtures/eval/reverse_baseline.json
eval-reverse:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-reverse-baseline output=tests/fixtures/eval/reverse_baseline.json

# Capture a variant baseline to /tmp (agg = identity / max-chunk / dedupe / topk-average)
eval-baseline-variant agg:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-baseline aggregation={{agg}} \
      output=/tmp/baseline-{{agg}}.json

# Capture all 4 strategies + compare-baselines markdown table
eval-compare:
    just eval-baseline-variant identity
    just eval-baseline-variant max-chunk
    just eval-baseline-variant dedupe
    just eval-baseline-variant topk-average
    cargo run --bin eval_harness --features eval-harness --release -- \
      compare-baselines paths=/tmp/baseline-identity.json,/tmp/baseline-max-chunk.json,/tmp/baseline-dedupe.json,/tmp/baseline-topk-average.json

# Evaluate (kind = full / shuffled / identity / reverse / single_doc)
eval-evaluate kind="full":
    cargo run --bin eval_harness --features eval-harness --release -- \
      evaluate kind={{kind}}

# Verify committed baseline.json against the current pipeline (FR-017 gate)
eval-verify:
    cargo run --bin eval_harness --features eval-harness --release -- \
      verify-baseline baseline=tests/fixtures/eval/baseline.json

# === embed スピード + 数値同等性ハーネス (mlx_smoke, ADR 0002) ===

# Capture embed fixture → tests/fixtures/embed/{w1,w2,w3}.bin
embed-capture:
    cargo run --bin mlx_smoke --release -- capture-fixture

# Measure embed speed / padding / R² baseline (stderr diagnostics)
embed-baseline:
    cargo run --bin mlx_smoke --release -- measure-baseline

# Verify embed fixture parity (cosine_min ≥ 0.99999, max_abs_diff ≤ 1e-5)
embed-verify:
    cargo run --bin mlx_smoke --release -- verify-fixture

# === Probe smoke (subprocess probe contract) ===

probe-embed:
    cargo run --bin probe_embed_smoke --release

probe-reranker:
    cargo run --bin probe_reranker_smoke --release

# Both probes in sequence
probe: probe-embed probe-reranker
