//! Local semantic search over text — embedding, indexing, and retrieval.
#![warn(missing_docs)]

/// Typed artifact verification: [`CandidateArtifacts`](embed::CandidateArtifacts) → [`VerifiedArtifacts`](artifacts::VerifiedArtifacts).
pub mod artifacts;
/// Top-level probe dispatch wiring embed and reranker domains.
pub mod dispatch;
/// Embedding models and the [`Embed`](embed::Embed) trait.
pub mod embed;
/// Process-global MLX cache lock shared by embed and reranker.
pub(crate) mod mlx_cache;
/// Unified [`ModelInitError`](model_init::ModelInitError) for embed and reranker constructors.
pub mod model_init;
/// Shared model I/O utilities (config, tokenizer, constants).
pub(crate) mod model_io;
/// Generic model lifecycle entry points: download / cache lookup / probe-env resolution.
pub mod model_lifecycle;
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
/// Public API for downstream consumers that drive MLX through rurico and need
/// to skip MLX-touching paths inside Codex Desktop's seatbelt sandbox. See
/// the [module-level docs](sandbox) for the operational contract.
pub mod sandbox;
/// SQLite-backed vector + FTS5 hybrid storage.
pub mod storage;
#[cfg(test)]
pub(crate) mod test_support;
/// Text chunking utilities.
pub mod text;

pub use dispatch::{handle_probe_if_needed, handle_probe_if_needed_with};
