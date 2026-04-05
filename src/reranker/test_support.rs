use super::{RankedResult, Rerank, RerankerError};

/// Returns a fixed score for all (query, document) pairs.
///
/// Default score is `0.5`. Use [`MockReranker::with_score`] to customize.
/// Useful for testing code that depends on [`Rerank`] without a live model.
pub struct MockReranker {
    score: f32,
}

impl Default for MockReranker {
    fn default() -> Self {
        Self { score: 0.5 }
    }
}

impl MockReranker {
    /// Create with a fixed score returned for every pair.
    pub fn with_score(score: f32) -> Self {
        Self { score }
    }
}

impl Rerank for MockReranker {
    fn score(&self, _query: &str, _document: &str) -> Result<f32, RerankerError> {
        Ok(self.score)
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        Ok(vec![self.score; pairs.len()])
    }

    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
        let scores = vec![self.score; documents.len()];
        Ok(super::sort_results(&scores))
    }
}
