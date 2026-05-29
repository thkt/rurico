use std::fmt::{self, Debug, Formatter};
use std::sync::OnceLock;

use super::{RankedResult, Rerank, RerankerError};

type InitFn<R> = dyn Fn() -> Result<R, String> + Send + Sync;

/// Defers wrapped reranker construction until the first [`Rerank`] call.
///
/// Wraps any [`Rerank`] implementation so model artifacts, tokenizer, and MLX
/// state are only allocated when actually needed. Useful for replay-first
/// retrieval paths (e.g. `replay-first-search`,
/// `verify-baseline kind=first_search_replay`) that may skip reranking
/// entirely — without [`LazyReranker`] those paths still pay the load cost
/// of an unused reranker.
///
/// The init closure runs at most once: [`OnceLock`] synchronises concurrent
/// callers and both success and failure are cached. A failed init is **not**
/// retried — model-load errors are deterministic in-process (missing
/// artifacts, corrupted cache, malformed config) so re-running the closure
/// would burn IO without changing the outcome.
///
/// # Examples
///
/// ```no_run
/// # use rurico::reranker::{RankedResult, Rerank, RerankerError};
/// # struct Stub;
/// # impl Rerank for Stub {
/// #     fn score(&self, _q: &str, _d: &str) -> Result<f32, RerankerError> { Ok(0.5) }
/// #     fn score_batch(&self, p: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
/// #         Ok(vec![0.5; p.len()])
/// #     }
/// #     fn rerank(&self, _q: &str, d: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
/// #         Ok(d.iter().enumerate().map(|(i, _)| RankedResult { index: i, score: 0.5 }).collect())
/// #     }
/// # }
/// use rurico::reranker::LazyReranker;
///
/// let lazy = LazyReranker::new(|| Ok(Stub));
/// // No init has happened yet — closure runs only on the first call below.
/// let score = lazy.score("query", "document")?;
/// # Ok::<(), rurico::reranker::RerankerError>(())
/// ```
pub struct LazyReranker<R: Rerank> {
    cell: OnceLock<Result<R, String>>,
    init: Box<InitFn<R>>,
}

impl<R: Rerank> LazyReranker<R> {
    /// Wrap a fallible init closure. The closure runs on the first
    /// [`Rerank`] method call and at most once.
    pub fn new<F>(init: F) -> Self
    where
        F: Fn() -> Result<R, String> + Send + Sync + 'static,
    {
        Self {
            cell: OnceLock::new(),
            init: Box::new(init),
        }
    }

    /// Borrow the wrapped reranker, lazily running the init closure on the
    /// first call. Subsequent calls return the cached outcome (success or
    /// failure) without re-running the closure.
    fn inner(&self) -> Result<&R, RerankerError> {
        match self.cell.get_or_init(|| (self.init)()) {
            Ok(r) => Ok(r),
            Err(msg) => Err(RerankerError::InitFailed(msg.clone())),
        }
    }
}

impl<R: Rerank> Debug for LazyReranker<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyReranker")
            .field("initialized", &self.cell.get().is_some())
            .finish_non_exhaustive()
    }
}

impl<R: Rerank> Rerank for LazyReranker<R> {
    fn score(&self, query: &str, document: &str) -> Result<f32, RerankerError> {
        self.inner()?.score(query, document)
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        self.inner()?.score_batch(pairs)
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        self.inner()?.rerank(query, documents)
    }
}

#[cfg(test)]
mod tests;
