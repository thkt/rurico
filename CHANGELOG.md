# Changelog

## [Unreleased]

### Breaking Changes

- **Remove `mlx` Cargo feature.** `mlx-rs` is now a regular (non-optional)
  dependency. Downstream crates referencing `rurico/mlx` in their `Cargo.toml`
  must remove that feature reference. The crate is now explicitly MLX-backend
  only.
- **Remove `EmbedError::BackendUnavailable` and `EmbedError::ModelNotAvailable`
  variants.** Neither variant was constructed within the crate.
  `ProbeStatus::BackendUnavailable` remains available for probe-based backend
  detection.
