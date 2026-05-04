# Changelog

## [Unreleased]

### Breaking Changes

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
  `reranker::download_model` / `reranker::cached_artifacts` are kept as
  re-exports so existing call sites continue to compile.
- **`ModelArtifact` trait gains an `Kind: ModelKind` associated type.**
  Implementors must now specify `type Kind = EmbedKind;` (or `RerankerKind`).
  This binds the model identifier to its kind so wrong-kind combinations
  (e.g. `download_model::<EmbedKind, _>(RerankerModelId::default())`) are
  no longer expressible at the call site.
- **`CandidateArtifacts` is now generic over a kind marker.**
  `rurico::artifacts::CandidateArtifacts<K>` is the canonical definition;
  `embed::CandidateArtifacts` and `reranker::CandidateArtifacts` are now
  type aliases for `CandidateArtifacts<EmbedKind>` /
  `CandidateArtifacts<RerankerKind>` respectively.
- **`embed_document` returns `ChunkedEmbedding` instead of `Vec<f32>`.** Long
  documents are split into overlapping chunks. Short documents return
  `chunks.len() == 1` with an identical embedding value.
- **`embed_documents_batch` returns `Vec<ChunkedEmbedding>` instead of
  `Vec<Vec<f32>>`.** Each element maps 1:1 to an input document.
- **Remove `mlx` Cargo feature.** `mlx-rs` is now a regular (non-optional)
  dependency. Downstream crates referencing `rurico/mlx` in their `Cargo.toml`
  must remove that feature reference. The crate is now explicitly MLX-backend
  only.
- **Remove `EmbedError::BackendUnavailable` and `EmbedError::ModelNotAvailable`
  variants.** Neither variant was constructed within the crate.
  `ProbeStatus::BackendUnavailable` remains available for probe-based backend
  detection.
- **MSRV bump to 1.95.** `rust-version` in `Cargo.toml` is now `1.95`.
  Downstream crates pinning an older toolchain must upgrade.

### Added

- **Chunked embedding for long documents.** Documents exceeding the model's
  `MAX_SEQ_LEN` (8192 tokens) are split into overlapping chunks
  (`CHUNK_OVERLAP_TOKENS = 2048`). Each chunk retains the document prefix per
  the upstream pooling contract (prefix tokens are always included in the
  attended sequence).
- **Query truncation.** `embed_query` truncates oversize queries to
  `MAX_SEQ_LEN` with a `log::warn` instead of failing. BOS/EOS are preserved.
- **Defense-in-depth in `model.forward()`.** Oversize inputs are truncated with
  a warning instead of returning an error. Normal API paths do not depend on
  this fallback.
- **Prefix-aware chunking (IG-001).** Long documents re-tokenize each chunk
  with the document prefix to avoid token boundary mismatch from prefix
  merging. Chunk boundaries are adjusted to fit within `MAX_SEQ_LEN` without
  truncation.
