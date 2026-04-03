use std::sync::atomic::{AtomicU32, Ordering};

use super::{ChunkedEmbedding, EMBEDDING_DIMS, Embed, EmbedError};

fn one_hot(index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
    v[index % EMBEDDING_DIMS as usize] = 1.0;
    v
}

/// Returns deterministic one-hot vectors for all inputs.
pub struct MockEmbedder;

impl Embed for MockEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: vec![one_hot(0)],
        })
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| ChunkedEmbedding {
                chunks: vec![one_hot(i)],
            })
            .collect())
    }
}

/// Embedder that returns errors (configurable per method).
pub struct FailingEmbedder {
    message: &'static str,
    docs_fail: bool,
}

impl FailingEmbedder {
    /// Both `embed_query` and `embed_document` return errors.
    pub fn all_fail(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: true,
        }
    }

    /// Only `embed_query` returns errors; documents succeed.
    pub fn query_only(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: false,
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
                chunks: vec![one_hot(0)],
            })
        }
    }
}

/// Batch returns fewer vectors than inputs (triggers mismatch errors).
pub struct MismatchEmbedder;

impl Embed for MismatchEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: vec![one_hot(0)],
        })
    }

    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(vec![ChunkedEmbedding {
            chunks: vec![one_hot(0)],
        }])
    }
}

/// Returns multi-chunk embeddings for documents.
///
/// `embed_document` returns `chunks_per_doc` chunks per document.
/// `embed_query` returns a single vector as usual.
pub struct MockChunkedEmbedder {
    chunks_per_doc: usize,
}

impl MockChunkedEmbedder {
    /// Create with the given number of chunks per document.
    pub fn new(chunks_per_doc: usize) -> Self {
        Self { chunks_per_doc }
    }
}

impl Embed for MockChunkedEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(ChunkedEmbedding {
            chunks: (0..self.chunks_per_doc).map(one_hot).collect(),
        })
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| ChunkedEmbedding {
                chunks: (0..self.chunks_per_doc)
                    .map(|j| one_hot(i * self.chunks_per_doc + j))
                    .collect(),
            })
            .collect())
    }
}

/// Alternates between success and failure on `embed_document`.
/// The first call (call_count=0) fails; odd-numbered calls succeed.
pub struct AlternatingEmbedder {
    call_count: AtomicU32,
}

impl AlternatingEmbedder {
    /// Create with call counter at zero (first `embed_document` fails).
    pub fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
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
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n.is_multiple_of(2) {
            Err(EmbedError::Inference("alternating failure".into()))
        } else {
            Ok(ChunkedEmbedding {
                chunks: vec![one_hot(0)],
            })
        }
    }
}
