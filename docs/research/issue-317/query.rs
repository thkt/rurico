//! Unadopted, object-safe query batch proposal. Only compiled in lib tests.
use crate::embed::ForwardShape;
use crate::embed::{
    Embed, EmbedError, EmbedOptions, InferenceMetrics, MAX_SEQ_LEN, QUERY_PREFIX,
    tokenize_with_prefix, truncate_for_query,
};
use std::{cell::RefCell, time::Duration};
use tokenizers::Tokenizer;

pub(crate) struct Queries {
    pub(crate) values: Vec<Vec<f32>>,
    pub(crate) metrics: Option<InferenceMetrics>,
}

pub(crate) trait QueryBatch: Embed {
    // Existing implementors need no scheduling options or measurements.
    fn queries(&self, texts: &[&str]) -> Result<Queries, EmbedError> {
        Ok(Queries {
            values: texts
                .iter()
                .map(|text| self.embed_query(text))
                .collect::<Result<_, _>>()?,
            metrics: None,
        })
    }
}

pub(crate) fn prepare(tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Vec<u32>>, EmbedError> {
    texts
        .iter()
        .map(|text| {
            let tokens = tokenize_with_prefix(tokenizer, text, QUERY_PREFIX)?;
            let (ids, mask, _) =
                truncate_for_query(tokens.input_ids, tokens.attention_mask, MAX_SEQ_LEN);
            // Existing bucket padding constructs masks from lengths. This spike is
            // only eligible for the pinned unpadded tokenizer, never arbitrary masks.
            if mask.iter().any(|&value| value != 1) {
                return Err(EmbedError::inference_message(
                    "query batch requires unpadded encoding",
                ));
            }
            Ok(ids)
        })
        .collect()
}

pub(crate) fn default_options() -> EmbedOptions {
    EmbedOptions::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::ChunkedEmbedding;
    use std::sync::Mutex;
    #[derive(Default)]
    struct Spy(Mutex<Vec<String>>);
    impl Embed for Spy {
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            self.0.lock().unwrap().push(text.into());
            match text {
                "b" => Ok(vec![2.0, -2.0]),
                "a" => Ok(vec![1.0, -1.0]),
                "" => Ok(vec![3.0, -3.0]),
                _ => Err(EmbedError::NonFiniteOutput),
            }
        }
        fn embed_document(&self, _: &str) -> Result<ChunkedEmbedding, EmbedError> {
            unreachable!()
        }
        fn embed_documents_batch(&self, _: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            unreachable!()
        }
        fn embed_text(&self, _: &str, _: &str) -> Result<Vec<f32>, EmbedError> {
            unreachable!()
        }
    }
    impl QueryBatch for Spy {}
    #[test]
    fn default_dyn_delegates_values_order_empty_and_first_error() {
        let spy = Spy::default();
        let provider: &dyn QueryBatch = &spy;
        assert!(provider.queries(&[]).unwrap().values.is_empty());
        assert!(spy.0.lock().unwrap().is_empty());
        let result = provider.queries(&["b", "", "a", "b"]).unwrap();
        assert!(result.metrics.is_none());
        assert_eq!(
            result.values,
            [vec![2., -2.], vec![3., -3.], vec![1., -1.], vec![2., -2.]]
        );
        assert!(matches!(
            provider.queries(&["a", "fail", "b"]),
            Err(EmbedError::NonFiniteOutput)
        ));
        assert_eq!(*spy.0.lock().unwrap(), ["b", "", "a", "b", "a", "fail"]);
    }
}

// Test-only observations from the actual single-query path; disabled by default.
thread_local! {
    static SINGLE: RefCell<Option<Vec<(Duration, ForwardShape)>>> = const { RefCell::new(None) };
}
pub(crate) fn begin_singles() {
    SINGLE.with_borrow_mut(|records| *records = Some(Vec::new()));
}
pub(crate) fn observe_single(tokenize: Duration, sequence_length: usize) {
    SINGLE.with_borrow_mut(|records| {
        if let Some(records) = records {
            records.push((
                tokenize,
                ForwardShape {
                    batch_size: 1,
                    sequence_length,
                },
            ));
        }
    });
}
pub(crate) fn take_singles() -> Vec<(Duration, ForwardShape)> {
    SINGLE.with_borrow_mut(|records| records.take().unwrap_or_default())
}
