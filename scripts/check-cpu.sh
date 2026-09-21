#!/usr/bin/env bash
# Metal- and model-free verification. Keep the default MLX checks in check.sh.
set -euo pipefail

# Check each package independently as well as workspace feature unification.
# Cargo.lock intentionally retains optional MLX packages; inspect active edges.
for selection in rurico rurico-ffi workspace; do
  if [[ "$selection" == workspace ]]; then
    packages=(--workspace)
  else
    packages=(-p "$selection")
  fi
  dependencies=$(cargo tree --locked "${packages[@]}" --no-default-features --edges normal,build,dev --prefix none)
  if grep -E '^mlx-(rs|sys) v' <<< "$dependencies"; then
    echo "MLX dependency leaked into CPU configuration: $selection" >&2
    exit 1
  fi
done

# test-support must not silently re-enable the native backend either.
dependencies=$(cargo tree --locked --workspace --no-default-features --features test-support --edges normal,build,dev --prefix none)
if grep -E '^mlx-(rs|sys) v' <<< "$dependencies"; then
  echo 'MLX dependency leaked through test-support' >&2
  exit 1
fi

# Aliases must re-enable MLX even when defaults are explicitly disabled.
for feature in mlx test-mlx smoke; do
  dependencies=$(cargo tree --locked -p rurico --no-default-features --features "$feature" --edges normal,build,dev --prefix none)
  for package in mlx-rs mlx-sys; do
    if ! grep -E "^${package} v" <<< "$dependencies" > /dev/null; then
      echo "$feature did not enable $package" >&2
      exit 1
    fi
  done
done

cargo check --locked -p rurico --no-default-features
cargo test --locked -p rurico-ffi --no-default-features
cargo nextest run --locked --workspace --no-default-features --features test-support --profile ci
cargo test --locked --doc --workspace --no-default-features --features test-support
cargo clippy --locked --workspace --all-targets --no-default-features --features test-support -- -D warnings
