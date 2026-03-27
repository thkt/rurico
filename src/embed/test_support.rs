use std::sync::atomic::{AtomicU32, Ordering};

use super::{EMBEDDING_DIMS, Embed, EmbedError};

fn one_hot(index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
    v[index % EMBEDDING_DIMS as usize] = 1.0;
    v
}

pub struct MockEmbedder;

impl Embed for MockEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().enumerate().map(|(i, _)| one_hot(i)).collect())
    }
}

pub struct FailingEmbedder {
    message: &'static str,
    docs_fail: bool,
}

impl FailingEmbedder {
    pub fn all_fail(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: true,
        }
    }

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

    fn embed_document(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        if self.docs_fail {
            Err(EmbedError::Inference(self.message.into()))
        } else {
            Ok(one_hot(0))
        }
    }
}

pub struct MismatchEmbedder;

impl Embed for MismatchEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![one_hot(0)])
    }
}

pub struct AlternatingEmbedder {
    call_count: AtomicU32,
}

impl AlternatingEmbedder {
    pub fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }
}

impl Embed for AlternatingEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(one_hot(0))
    }

    fn embed_document(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n % 2 == 0 {
            Err(EmbedError::Inference("alternating failure".into()))
        } else {
            Ok(one_hot(0))
        }
    }
}
