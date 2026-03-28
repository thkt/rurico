//! Local semantic search over text — embedding, indexing, and retrieval.
#![warn(missing_docs)]

/// Embedding models and the [`Embed`](embed::Embed) trait.
pub mod embed;
/// ModernBERT transformer implementation on MLX.
pub mod modernbert;
/// SQLite-backed vector + FTS5 hybrid storage.
pub mod storage;
/// Text chunking utilities.
pub mod text;
