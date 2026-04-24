use std::sync::atomic::{AtomicU32, Ordering};

use super::{ChunkedEmbedding, EMBEDDING_DIMS, Embed, EmbedError};

// ── Phase 2 reference workloads ──────────────────────────────────────────────
//
// Deterministic text generators for benchmarking `embed_documents_batch_chunked`.
// Phase 1 (issue #52) established `padding_ratio=1.405` and `forward_eval_ms=16,896`
// on the long-document mix that is now encoded as [`workload_w1`]. Workloads W2 and
// W3 explore other shapes of chunk-length distribution to validate that bucket
// batching improves the long-document case without degrading short-text throughput.

/// W1: long-document mix. 2 texts, both chunked into multiple overlapping chunks.
/// Matches the Phase 1 smoke's long-doc assertion (~7991 + ~3669 tokens).
pub fn workload_w1() -> Vec<String> {
    vec![
        "apple pie is a traditional dessert enjoyed around the world. ".repeat(800),
        "the rain in Spain falls mainly on the plain. ".repeat(500),
    ]
}

/// W2: short-text-heavy batch. 100 texts, each ~50-70 characters
/// (well below a single chunk boundary).
pub fn workload_w2() -> Vec<String> {
    (0..100)
        .map(|i| format!("short text number {i} for benchmarking W2 workload"))
        .collect()
}

/// W3: long/short alternating. 10 texts: 5 long (~4000-5600 tokens each)
/// interleaved with 5 short (~100 bytes each). Stresses the long×short
/// padding-waste case that bucket batching is designed to fix.
pub fn workload_w3() -> Vec<String> {
    (0..5)
        .flat_map(|i| {
            vec![
                "benchmarking long text for W3 workload. ".repeat(100 + i * 10),
                format!("short text {i}"),
            ]
        })
        .collect()
}

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

#[cfg(test)]
mod workload_tests {
    use super::*;

    // W1: 2 long texts, both containing their base phrase
    #[test]
    fn workload_w1_has_two_long_texts() {
        let w = workload_w1();
        assert_eq!(w.len(), 2);
        assert!(w[0].contains("apple pie"));
        assert!(w[1].contains("Spain"));
        assert!(
            w[0].len() > 40000,
            "w1[0] should be long, got {}",
            w[0].len()
        );
        assert!(
            w[1].len() > 20000,
            "w1[1] should be long, got {}",
            w[1].len()
        );
    }

    // W2: exactly 100 short texts, each under 80 bytes
    #[test]
    fn workload_w2_has_hundred_short_texts() {
        let w = workload_w2();
        assert_eq!(w.len(), 100);
        assert!(
            w.iter().all(|t| t.len() < 80),
            "all W2 texts should be short"
        );
        assert!(w[0].contains("number 0"));
        assert!(w[99].contains("number 99"));
    }

    // W3: 10 texts, alternating long/short
    #[test]
    fn workload_w3_alternates_long_and_short() {
        let w = workload_w3();
        assert_eq!(w.len(), 10);
        for (i, text) in w.iter().enumerate() {
            if i.is_multiple_of(2) {
                assert!(
                    text.len() > 3000,
                    "w3[{i}] should be long, got {}",
                    text.len()
                );
            } else {
                assert!(
                    text.len() < 50,
                    "w3[{i}] should be short, got {}",
                    text.len()
                );
            }
        }
    }

    // Determinism: same function returns equal output across calls
    #[test]
    fn workloads_are_deterministic() {
        assert_eq!(workload_w1(), workload_w1());
        assert_eq!(workload_w2(), workload_w2());
        assert_eq!(workload_w3(), workload_w3());
    }
}
