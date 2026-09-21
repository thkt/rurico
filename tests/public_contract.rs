//! External consumer contract, exercised with and without the MLX feature.
#![cfg(feature = "test-support")]

use std::error::Error;

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError, EmbedOptions, MockEmbedder};
use rurico::model_init::ModelInitError;
use rurico::reranker::{LazyReranker, MockReranker, Rerank, RerankerError};
use rurico::retrieval::{Candidate, CandidateSource, MergeStrategy, WeightedRrf};
use rurico::storage::ensure_sqlite_vec;
use rusqlite::Connection;

#[test]
fn consumer_can_store_mock_embeddings_and_rerank_retrieved_candidates() -> Result<(), Box<dyn Error>>
{
    let embedder: &dyn Embed = &MockEmbedder::with_dims(4);
    let embeddings: Vec<ChunkedEmbedding> = embedder
        .embed_documents_batch_with_options(&["文書A", "文書B"], &EmbedOptions::default())?;
    // Public error types remain available without any concrete backend.
    let _: Option<(EmbedError, ModelInitError, RerankerError)> = None;

    ensure_sqlite_vec()?;
    let conn = Connection::open_in_memory()?;
    conn.execute_batch("CREATE VIRTUAL TABLE vectors USING vec0(embedding float[4]);")?;
    for (i, embedding) in embeddings.iter().enumerate() {
        conn.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?1, ?2)",
            rusqlite::params![
                i64::try_from(i + 1)?,
                bytemuck::cast_slice::<f32, u8>(&embedding.chunks()[0])
            ],
        )?;
    }
    let query = embedder.embed_query("検索")?;
    let id: i64 = conn.query_row(
        "SELECT rowid FROM vectors WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
        [bytemuck::cast_slice::<f32, u8>(&query)],
        |row| row.get(0),
    )?;
    assert_eq!(id, 1);
    let merged = WeightedRrf::default().merge(&[Candidate {
        source: CandidateSource::Vector,
        doc_id: id.to_string(),
        chunk_id: Some(embeddings[0].chunk_ids()[0].clone()),
        score: 1.0,
        rank: 0,
    }]);
    assert_eq!(merged[0].doc_id, "1");
    let reranker = LazyReranker::new(|| Ok(MockReranker::default()));
    let provider: &dyn Rerank = &reranker;
    let ranked = provider.rerank("検索", &[&merged[0].doc_id])?;
    assert_eq!(ranked[0].index, 0);
    assert_eq!(ranked[0].score, 0.5);
    Ok(())
}
