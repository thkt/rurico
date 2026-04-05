//! Local semantic search over text — embedding, indexing, and retrieval.
#![warn(missing_docs)]

/// Typed artifact verification: [`CandidateArtifacts`](embed::CandidateArtifacts) → [`VerifiedArtifacts`](artifacts::VerifiedArtifacts).
pub mod artifacts;
/// Embedding models and the [`Embed`](embed::Embed) trait.
pub mod embed;
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
/// SQLite-backed vector + FTS5 hybrid storage.
pub mod storage;
#[cfg(test)]
pub(crate) mod test_support;
/// Text chunking utilities.
pub mod text;
