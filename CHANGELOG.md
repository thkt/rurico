# Changelog

## [Unreleased]

### Breaking Changes

- **`storage::rrf_merge` and `RRF_K` constant removed.** The free function and
  its hardcoded `RRF_K = 60.0` are deleted; `retrieval::WeightedRrf` is now
  the canonical RRF primitive (configurable `rrf_k` via `HybridSearchConfig`,
  multi-source weights, recency support). For drop-in replacement of the
  legacy behavior, use `WeightedRrf::default().merge(&candidates)` — bit-equal
  to the removed `rrf_merge` (default `rrf_k = 60.0`, weight = 1.0, including
  the `doc_id` ascending tie-break). Callers that fed `(K, f64)` tuples must
  convert to `Vec<Candidate>` (see `rurico::retrieval::Candidate`).
- **`EmbedInitError` and `RerankerInitError` removed; replaced by single
  `ModelInitError` (`rurico::model_init::ModelInitError`).** Variant set
  (`Backend(String)` and `ModelCorrupt { reason: String }`) is unchanged;
  callers that previously matched on the per-kind enum must rename the type
  only. `embed::Embedder::new` / `embed::Embedder::probe` /
  `reranker::Reranker::new` / `reranker::Reranker::probe` now return
  `Result<_, ModelInitError>`.
- **`download_model` and `cached_artifacts` signatures changed.** Both
  functions are now generic over `ModelArtifact` and live in
  `rurico::model_lifecycle`: `pub fn download_model<Id: ModelArtifact>(model: Id)
-> Result<VerifiedArtifacts<Id::Kind>, ArtifactError>` and
  `pub fn cached_artifacts<Id: ModelArtifact>(model: Id)
-> Result<Option<VerifiedArtifacts<Id::Kind>>, ArtifactError>`.
  `embed::download_model` / `embed::cached_artifacts` /
  `reranker::download_model` / `reranker::cached_artifacts` remain as
  `pub use` re-exports of the canonical functions, so existing call
  sites continue to compile unchanged.
- **Internal: `ModelArtifact` trait gains an associated type `Kind: ModelKind`.**
  `ModelArtifact` is a `pub trait` inside the `pub(crate) mod model_io`,
  so it has no external implementors and this is not a caller-visible
  change. The associated type binds each model identifier to its kind
  marker (`ModelId::Kind = EmbedKind`, `RerankerModelId::Kind =
RerankerKind`), which lets `download_model(model)` return
  `VerifiedArtifacts<Id::Kind>` without an explicit kind type parameter.
  Wrong-kind combinations (e.g. constructing a candidate for the embed
  kind from a `RerankerModelId`) are no longer expressible at the call
  site.
- **`CandidateArtifacts` is now generic over a kind marker.**
  `rurico::artifacts::CandidateArtifacts<K>` is the canonical definition;
  `embed::CandidateArtifacts` and `reranker::CandidateArtifacts` are now
  type aliases for `CandidateArtifacts<EmbedKind>` /
  `CandidateArtifacts<RerankerKind>` respectively.
- **`embed_document` returns `ChunkedEmbedding` instead of `Vec<f32>`.** Long
  documents are split into overlapping chunks. Short documents return
  `chunks().len() == 1` with an identical embedding value.
- **`embed_documents_batch` returns `Vec<ChunkedEmbedding>` instead of
  `Vec<Vec<f32>>`.** Each element maps 1:1 to an input document.
- **Remove `mlx` Cargo feature.** `mlx-rs` is now a regular (non-optional)
  dependency. The crate is explicitly MLX-backend only and compiles
  unconditionally with MLX. Downstream migration: remove
  `features = ["mlx"]` from any `rurico` dependency entry, and remove
  `#[cfg(feature = "rurico/mlx")]` guards from downstream code (those
  paths are now always active).
- **Remove `EmbedError::BackendUnavailable` and `EmbedError::ModelNotAvailable`
  variants.** Neither variant was ever constructed within the crate.
  Migration: callers that pattern-match on either variant must remove
  the corresponding match arm. `ProbeStatus::BackendUnavailable` is the
  replacement for backend-availability detection.
- **`ModelId::default()` removed.** Use the explicit `ModelId::DEFAULT`
  constant when callers want the product-chosen default embedding model.
- **`ModelId::repo_id()` inherent helper removed.** Model artifact metadata is
  now owned by the internal `ModelArtifact` implementation, avoiding a method
  name collision between the inherent helper and trait method.
- **`ChunkedEmbedding::new` replaced by `ChunkedEmbedding::try_new`.** Empty
  chunk lists now return `EmptyChunksError`; `chunks` / `chunk_ids` fields are
  now private to keep the invariant enforceable. Use `chunks()` and
  `chunk_ids()` for read access.
- **`Embed::embed_documents_batch` no longer has a default implementation.**
  Implementors must explicitly choose a per-document fallback or a fused batch
  implementation.
- **Public error enums are now `#[non_exhaustive]`.** Downstream `match`
  expressions over `ModelInitError`, `EmbedError`, or `ArtifactError` must
  include a wildcard arm. The crate-internal `ModelIoError` is also annotated.
- **`EmbedError::Inference` and `EmbedError::Tokenizer` are struct variants
  with source chains.** Match on `message` / `source` fields instead of the old
  tuple payload.
- **MSRV bump to 1.96.** `rust-version` in `Cargo.toml` is now `1.96`
  (`assert_matches!`, stabilized in 1.96, is now used in tests).
  Downstream crates pinning an older toolchain must upgrade.

### Added

- **Chunked embedding for long documents.** Documents exceeding the model's
  `MAX_SEQ_LEN` (8192 tokens) are split into overlapping chunks
  (`CHUNK_OVERLAP_TOKENS = 2048`). Each chunk retains the document prefix per
  the upstream pooling contract (prefix tokens are always included in the
  attended sequence).
- **Query truncation.** `embed_query` truncates oversize queries to
  `MAX_SEQ_LEN` with a `log::warn` instead of failing. BOS/EOS
  (beginning- and end-of-sequence tokens) are preserved.
- **Defense-in-depth in `model.forward()`.** Oversize inputs are truncated
  with a warning instead of returning an error. Public API paths
  (`embed_query` / `embed_document`) already truncate before calling
  `forward()`, so this fallback only fires when callers invoke
  `forward()` directly with un-truncated inputs.
- **Prefix-aware chunking.** Long documents re-tokenize each chunk
  with the document prefix to avoid token boundary mismatch from prefix
  merging. Chunk boundaries are adjusted to fit within `MAX_SEQ_LEN` without
  truncation.

### Fixed

- **`WeightedRrf` no longer emits `rrf_k`-induced non-finite fused scores.** A
  `HybridSearchConfig` with `rrf_k <= 0`, NaN, or ±inf could drive the RRF
  denominator (`rrf_k + rank`) non-positive or non-finite, producing `inf` or
  NaN `MergedHit.score` values that corrupt Stage 3/4 ranking and JSON output.
  Such contributions are now dropped (same handling as a zero-weight source),
  and the valid range for `rrf_k` is documented on `HybridSearchConfig::rrf_k`.
  Default `rrf_k = 60.0` is unaffected (bit-equal).
- **`WeightedRrf` no longer emits weight- or recency-induced non-finite fused
  scores.** Follow-up to the `rrf_k` guard above: non-finite (NaN, ±inf)
  weights slipped past the `== 0.0` guards under IEEE 754, and a non-finite
  decay (NaN `half_life_days`, or `+inf` half-life with a `+inf` age from the
  caller's `age_days_for` closure) leaked through the boost arithmetic — all
  contaminating `MergedHit.score` with values that corrupt Stage 3/4 ranking
  and JSON output. The guards now close each path:
  - a non-finite `HybridSearchConfig::source_weights` entry has its source's
    contribution dropped (same handling as a zero weight);
  - a non-finite `RecencyConfig::weight` disables the recency boost;
  - a per-hit non-finite boost (`weight × decay`) is skipped, retaining the
    RRF score.
    `half_life_days = +inf` with a finite age keeps its full boost (decay
    exactly `1.0`), and all-finite configurations are unaffected (bit-equal).
