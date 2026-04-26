//! Local semantic search over text — embedding, indexing, and retrieval.
#![warn(missing_docs)]

/// Typed artifact verification: [`CandidateArtifacts`](embed::CandidateArtifacts) → [`VerifiedArtifacts`](artifacts::VerifiedArtifacts).
pub mod artifacts;
/// Embedding models and the [`Embed`](embed::Embed) trait.
pub mod embed;
/// Search evaluation harness — metrics, fixtures, pipeline (Phase 1, Issue #65).
#[cfg(feature = "eval-harness")]
pub mod eval;
/// Process-global MLX cache lock shared by embed and reranker.
pub(crate) mod mlx_cache;
/// Shared model I/O utilities (config, tokenizer, constants).
pub(crate) mod model_io;
/// Shared subprocess probe infrastructure for model loading verification.
pub mod model_probe;
/// ModernBERT transformer implementation on MLX.
pub mod modernbert;
/// Cross-encoder reranker for query-document relevance scoring.
pub mod reranker;
/// Retrieval pipeline contract: Stage 3 [`Aggregator`](retrieval::Aggregator) hook (ADR 0004, Issue #67).
pub mod retrieval;
/// Codex seatbelt sandbox detection for MLX runtime gating.
///
/// Internal support API used by smoke binaries and integration tests. Exposed
/// as `pub` for cross-target reuse within this crate — not part of the public
/// semantic-search surface for downstream consumers.
#[doc(hidden)]
pub mod sandbox;
/// SQLite-backed vector + FTS5 hybrid storage.
pub mod storage;
#[cfg(test)]
pub(crate) mod test_support;
/// Text chunking utilities.
pub mod text;
