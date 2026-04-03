use std::sync::atomic::{AtomicU32, Ordering};

use super::{ChunkedEmbedding, EMBEDDING_DIMS, Embed, EmbedError};

fn one_hot(index: usize, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dims];
    v[index % dims] = 1.0;
    v
}

/// Returns deterministic one-hot vectors for all inputs.
///
/// Default dimension is [`EMBEDDING_DIMS`] (768, matching `ruri-v3-310m`).
/// Use [`MockEmbedder::with_dims`] to test other model sizes.
pub struct MockEmbedder {
    dims: usize,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self {
            dims: EMBEDDING_DIMS,
        }
    }
}

impl MockEmbedder {
    /// Create with a custom embedding dimension.
    pub fn with_dims(dims: usize) -> Self {
        Self { dims }
    }
}

impl Embed for MockEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0, self.dims))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: vec![one_hot(0, self.dims)],
        })
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| ChunkedEmbedding {
                chunks: vec![one_hot(i, self.dims)],
            })
            .collect())
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
}

/// Embedder that returns errors (configurable per method).
pub struct FailingEmbedder {
    message: &'static str,
    docs_fail: bool,
    dims: usize,
}

impl FailingEmbedder {
    /// Both `embed_query` and `embed_document` return errors.
    pub fn all_fail(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: true,
            dims: EMBEDDING_DIMS,
        }
    }

    /// Only `embed_query` returns errors; documents succeed.
    pub fn query_only(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: false,
            dims: EMBEDDING_DIMS,
        }
    }
}

impl Embed for FailingEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Inference(self.message.into()))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        if self.docs_fail {
            Err(EmbedError::Inference(self.message.into()))
        } else {
            Ok(ChunkedEmbedding {
                chunks: vec![one_hot(0, self.dims)],
            })
        }
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
}

/// Batch returns fewer vectors than inputs (triggers mismatch errors).
pub struct MismatchEmbedder;

impl Embed for MismatchEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0, EMBEDDING_DIMS))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: vec![one_hot(0, EMBEDDING_DIMS)],
        })
    }

    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(vec![ChunkedEmbedding {
            chunks: vec![one_hot(0, EMBEDDING_DIMS)],
        }])
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
}

/// Returns multi-chunk embeddings for documents.
///
/// `embed_document` returns `chunks_per_doc` chunks per document.
/// `embed_query` returns a single vector as usual.
///
/// Default dimension is [`EMBEDDING_DIMS`] (768). Use [`MockChunkedEmbedder::with_dims`]
/// to test other model sizes.
pub struct MockChunkedEmbedder {
    chunks_per_doc: usize,
    dims: usize,
}

impl MockChunkedEmbedder {
    /// Create with the given number of chunks per document (default dimension).
    pub fn new(chunks_per_doc: usize) -> Self {
        Self {
            chunks_per_doc,
            dims: EMBEDDING_DIMS,
        }
    }

    /// Create with the given number of chunks per document and a custom embedding dimension.
    pub fn with_dims(chunks_per_doc: usize, dims: usize) -> Self {
        Self {
            chunks_per_doc,
            dims,
        }
    }
}

impl Embed for MockChunkedEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0, self.dims))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: (0..self.chunks_per_doc)
                .map(|i| one_hot(i, self.dims))
                .collect(),
        })
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| ChunkedEmbedding {
                chunks: (0..self.chunks_per_doc)
                    .map(|j| one_hot(i * self.chunks_per_doc + j, self.dims))
                    .collect(),
            })
            .collect())
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
}

/// Alternates between success and failure on `embed_document`.
/// The first call (call_count=0) fails; odd-numbered calls succeed.
pub struct AlternatingEmbedder {
    call_count: AtomicU32,
    dims: usize,
}

impl AlternatingEmbedder {
    /// Create with call counter at zero (first `embed_document` fails).
    pub fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
            dims: EMBEDDING_DIMS,
        }
    }
}

impl Default for AlternatingEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embed for AlternatingEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0, self.dims))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n.is_multiple_of(2) {
            Err(EmbedError::Inference("alternating failure".into()))
        } else {
            Ok(ChunkedEmbedding {
                chunks: vec![one_hot(0, self.dims)],
            })
        }
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_query(text)
    }
}
