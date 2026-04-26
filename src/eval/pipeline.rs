//! Reference search pipeline composition (FR-008..FR-010).
//!
//! Inline composition of `rurico` primitives — in-memory SQLite + FTS5 +
//! sqlite-vec + RRF + optional reranker — modeled after `recall`'s wiring
//! shape (ADR 0003). recall is *not* imported (cyclic dep avoided).
//!
//! Phase 1c RED — `evaluate` is a stub. Tests use `MockEmbedder` to drive
//! the wiring; Phase 1d wires the real mlx embedder via the harness binary.

use std::collections::HashMap;
use std::time::Instant;

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::embed::{EMBEDDING_DIMS, Embed, EmbedError};
use crate::eval::fixture::{EvalDocument, EvalQuery};
use crate::reranker::{Rerank, RerankerError};
use crate::retrieval::{
    Aggregator, Candidate, CandidateSource, MergeStrategy, MergedHit, WeightedRrf,
};
use crate::storage::{
    MatchFtsQuery, SanitizeError, ensure_sqlite_vec, f32_as_bytes, prepare_match_query,
};

/// FTS5 vocab table name used by [`prepare_match_query`].
const FTS_VOCAB_TABLE: &str = "docs_vocab";

/// FTS + vec retrievals each fetch this many candidates per query before
/// RRF; matches recall's `opts.limit * 3` heuristic.
const RRF_CANDIDATE_MULTIPLIER: usize = 3;

/// Upper bound on `cross_product` output size in [`clean_for_trigram`].
///
/// When `Π or_groups[i].len()` exceeds this threshold, the OR-distribution
/// path is skipped and only `fixed` terms are used. Prevents the O(Π m_i)
/// memory blowup observed in issue #71 (worst observed `8 × 25 × 25 × 12 ×
/// 20 × 25 = 30,000,000` combos in `queries.jsonl`).
///
/// At 100 combos the MATCH string stays under ~10 KB; the reranker recovers
/// any recall lost on the small fraction of queries that hit the fallback
/// (mrr / ndcg_at_10 unchanged versus a 10,000-combo cap).
const MAX_COMBOS: usize = 100;

/// One query's pipeline output: ordered hits + wall-clock latency.
///
/// `ranked_hits` reuses [`MergedHit`] — the same type passed across the Stage
/// 3 boundary — so the rerank tail no longer round-trips through a separate
/// `Hit` shape. Downstream callers should still treat the slice as sorted by
/// descending `score`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Identifier of the input [`EvalQuery`].
    pub query_id: String,
    /// Hits sorted by descending [`MergedHit::score`].
    pub ranked_hits: Vec<MergedHit>,
    /// Wall-clock latency for this single query in milliseconds.
    pub latency_ms: u64,
}

/// Tunable pipeline parameters.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Top-k cutoff after RRF merge.
    pub k: usize,
}

/// Errors surfaced by the reference pipeline.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum PipelineError {
    /// SQLite storage failure (schema build, FTS index, vec table, etc.).
    #[error("pipeline sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Embedder failed during query or document encoding.
    #[error("pipeline embed error: {0}")]
    Embed(#[from] EmbedError),
    /// Optional reranker failed during scoring.
    #[error("pipeline rerank error: {0}")]
    Rerank(#[from] RerankerError),
    /// FTS5 query sanitization rejected the surface query.
    #[error("pipeline FTS sanitize error: {0}")]
    Sanitize(#[from] SanitizeError),
    /// `sqlite_vec` extension registration failed at process level.
    #[error("pipeline sqlite-vec load failed: {0}")]
    SqliteVec(String),
    /// Embedder returned a chunk count that does not match the corpus.
    #[error(
        "pipeline embed batch size mismatch: corpus has {corpus} docs, embedder returned {chunked}"
    )]
    ChunkCountMismatch {
        /// Number of documents in the corpus passed to the embedder.
        corpus: usize,
        /// Number of [`crate::embed::ChunkedEmbedding`] entries returned.
        chunked: usize,
    },
    /// Embedder produced zero chunks for a corpus document.
    #[error("pipeline empty embedding for doc {doc_id:?}")]
    EmptyEmbedding {
        /// Identifier of the corpus document with no chunks.
        doc_id: String,
    },
}

/// Run the reference pipeline on `corpus` for every query in `queries`.
///
/// Indexes `corpus` into an in-memory SQLite store with FTS5 and sqlite-vec
/// virtual tables, encodes each query with `embedder`, retrieves top hits via
/// FTS5 and vector search, merges with RRF, and optionally rescores with
/// `reranker`. Output ranked-hit count is bounded by [`PipelineConfig::k`].
///
/// # Errors
///
/// See [`PipelineError`] variants. Sqlite, embed, rerank, and sanitize errors
/// each surface their respective source via `#[from]`; sqlite-vec load
/// failure surfaces as [`PipelineError::SqliteVec`].
pub fn evaluate<E, R, A>(
    corpus: &[EvalDocument],
    queries: &[EvalQuery],
    embedder: &E,
    reranker: Option<&R>,
    aggregator: &A,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, PipelineError>
where
    E: Embed,
    R: Rerank,
    A: Aggregator,
{
    ensure_sqlite_vec().map_err(PipelineError::SqliteVec)?;
    let conn = Connection::open_in_memory()?;
    create_schema(&conn)?;
    index_corpus(&conn, corpus, embedder)?;

    // Build (doc_id → body) lookup once before the query loop. apply_reranker
    // does O(1) hits against this map instead of an O(N) corpus.iter().find()
    // per merged hit per query.
    let corpus_index: HashMap<&str, &str> = corpus
        .iter()
        .map(|d| (d.id.as_str(), d.body.as_str()))
        .collect();

    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let started = Instant::now();
        let ranked_hits = run_single_query(
            &conn,
            query,
            embedder,
            reranker,
            aggregator,
            &corpus_index,
            config,
        )?;
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        results.push(QueryResult {
            query_id: query.id.clone(),
            ranked_hits,
            latency_ms,
        });
    }
    Ok(results)
}

/// Build the in-memory schema (documents + FTS5 + vec0 + fts5vocab).
fn create_schema(conn: &Connection) -> Result<(), PipelineError> {
    conn.execute_batch(&format!(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, body TEXT NOT NULL); \
         CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id UNINDEXED, body, tokenize='trigram'); \
         CREATE VIRTUAL TABLE vec_docs USING vec0(embedding FLOAT[{EMBEDDING_DIMS}], +doc_id TEXT); \
         CREATE VIRTUAL TABLE {FTS_VOCAB_TABLE} USING fts5vocab(docs_fts, row);"
    ))?;
    Ok(())
}

/// Encode each document body with `embedder` and insert into all three tables.
///
/// `EvalDocument` is treated as a single chunk; the first chunk vector from
/// [`Embed::embed_documents_batch`] is what lands in `vec_docs`. Returns
/// [`PipelineError::ChunkCountMismatch`] when the embedder's output length
/// does not match the corpus and [`PipelineError::EmptyEmbedding`] when a
/// document yields no chunks — silent truncation of the vec index would
/// degrade recall without a visible error.
fn index_corpus<E: Embed>(
    conn: &Connection,
    corpus: &[EvalDocument],
    embedder: &E,
) -> Result<(), PipelineError> {
    let bodies: Vec<&str> = corpus.iter().map(|d| d.body.as_str()).collect();
    let chunked = embedder.embed_documents_batch(&bodies)?;
    if chunked.len() != corpus.len() {
        return Err(PipelineError::ChunkCountMismatch {
            corpus: corpus.len(),
            chunked: chunked.len(),
        });
    }
    let mut insert_doc = conn.prepare_cached("INSERT INTO documents(id, body) VALUES (?, ?)")?;
    let mut insert_fts = conn.prepare_cached("INSERT INTO docs_fts(doc_id, body) VALUES (?, ?)")?;
    let mut insert_vec =
        conn.prepare_cached("INSERT INTO vec_docs(embedding, doc_id) VALUES (?, ?)")?;
    for (doc, chunked_embedding) in corpus.iter().zip(chunked.iter()) {
        insert_doc.execute(params![&doc.id, &doc.body])?;
        insert_fts.execute(params![&doc.id, &doc.body])?;
        let vector =
            chunked_embedding
                .chunks
                .first()
                .ok_or_else(|| PipelineError::EmptyEmbedding {
                    doc_id: doc.id.clone(),
                })?;
        insert_vec.execute(params![f32_as_bytes(vector), &doc.id])?;
    }
    Ok(())
}

/// Drive one `EvalQuery` through FTS + vec retrieval, RRF merge, Stage 3
/// aggregation, and (when supplied) reranker rescoring.
fn run_single_query<E, R, A>(
    conn: &Connection,
    query: &EvalQuery,
    embedder: &E,
    reranker: Option<&R>,
    aggregator: &A,
    corpus_index: &HashMap<&str, &str>,
    config: &PipelineConfig,
) -> Result<Vec<MergedHit>, PipelineError>
where
    E: Embed,
    R: Rerank,
    A: Aggregator,
{
    let candidate_limit = config.k * RRF_CANDIDATE_MULTIPLIER;
    let fts_hits = retrieve_fts(conn, &query.text, candidate_limit)?;
    let vec_hits = retrieve_vec(conn, embedder, &query.text, candidate_limit)?;
    let mut all_candidates = Vec::with_capacity(fts_hits.len() + vec_hits.len());
    all_candidates.extend(fts_hits);
    all_candidates.extend(vec_hits);
    let merged_hits = WeightedRrf::default().merge(&all_candidates);

    // Aggregator::aggregate guarantees score-descending output (see trait
    // doc), so truncate-then-rerank does not silently drop higher-scoring
    // hits.
    let mut aggregated = aggregator.aggregate(&merged_hits);
    aggregated.truncate(config.k);

    if let Some(reranker) = reranker {
        aggregated = apply_reranker(reranker, &query.text, aggregated, corpus_index)?;
    }

    Ok(aggregated)
}

/// FTS5 retrieval. Empty / unsanitisable queries return an empty hit list
/// (mirrors recall's early-return behavior on `SanitizeError::EmptyInput`).
///
/// Returns Stage 1 [`Candidate`]s tagged with [`CandidateSource::Fts`]. The
/// `score` field carries SQLite FTS5's negative BM25 (lower magnitude is
/// better) and `rank` is 0-based — fed verbatim to [`WeightedRrf`].
fn retrieve_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<Candidate>, PipelineError> {
    let matched = match prepare_match_query(conn, query, FTS_VOCAB_TABLE) {
        Ok(m) => m,
        Err(SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms) => {
            return Ok(Vec::new());
        }
        Err(e) => return Err(e.into()),
    };
    let Some(fts_query) = clean_for_trigram(&matched) else {
        return Ok(Vec::new());
    };
    let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut stmt = conn.prepare_cached(
        "SELECT doc_id, rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
    )?;
    let rows = stmt.query_map(params![fts_query, limit_i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;
    let mut hits = Vec::new();
    for (rank, row) in rows.enumerate() {
        let (doc_id, score) = row?;
        hits.push(Candidate {
            source: CandidateSource::Fts,
            doc_id,
            score,
            rank,
        });
    }
    Ok(hits)
}

/// Adapt a [`MatchFtsQuery`] for an FTS5 `trigram` tokenizer.
///
/// The trigram tokenizer rejects `("a" OR "b") "c"` (a parenthesized OR-group
/// followed by implicit AND). Distribute OR-groups into flat alternatives —
/// `(A OR B) C` → `A C OR B C` — and drop sub-trigram (<3 char) terms inside
/// OR-groups because the trigram tokenizer cannot index them. Returns `None`
/// when no indexable terms remain (callers treat this as "no results").
///
/// Logic mirrors `amici::storage::fts::clean_for_trigram`; inlined here to
/// avoid pulling `amici` into rurico's dependency graph.
fn clean_for_trigram(query: &MatchFtsQuery) -> Option<String> {
    let cleaned: String = query.as_str().chars().filter(|c| !c.is_control()).collect();
    let (fixed, or_groups) = parse_fts_segments(&cleaned);
    let fixed_only = || (!fixed.is_empty()).then(|| fixed.join(" "));

    if or_groups.is_empty() {
        return fixed_only();
    }

    // Estimate Π m_i without materializing cross_product. Saturates at
    // usize::MAX on overflow so the threshold check still fires.
    let estimated_combos = or_groups
        .iter()
        .map(Vec::len)
        .try_fold(1usize, usize::checked_mul)
        .unwrap_or(usize::MAX);
    if estimated_combos > MAX_COMBOS {
        return fixed_only();
    }

    let combos = cross_product(&or_groups);
    Some(
        combos
            .iter()
            .map(|combo| {
                let mut parts = combo.clone();
                parts.extend(fixed.iter().cloned());
                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

fn parse_fts_segments(cleaned: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut fixed: Vec<String> = Vec::new();
    let mut or_groups: Vec<Vec<String>> = Vec::new();
    let mut chars = cleaned.chars();

    while let Some(c) = chars.next() {
        if c == '(' {
            let mut group = String::new();
            for gc in chars.by_ref() {
                if gc == ')' {
                    break;
                }
                group.push(gc);
            }
            let terms: Vec<String> = group
                .split(" OR ")
                .filter(|t| t.trim().trim_matches('"').chars().count() >= 3)
                .map(|t| t.trim().to_owned())
                .collect();
            if !terms.is_empty() {
                or_groups.push(terms);
            }
        } else if c == '"' {
            let mut term = String::from('"');
            for tc in chars.by_ref() {
                term.push(tc);
                if tc == '"' {
                    break;
                }
            }
            if term.trim_matches('"').chars().count() >= 3 {
                fixed.push(term);
            }
        }
    }

    (fixed, or_groups)
}

fn cross_product(groups: &[Vec<String>]) -> Vec<Vec<String>> {
    if groups.is_empty() {
        return vec![vec![]];
    }
    let rest = cross_product(&groups[1..]);
    let mut result = Vec::new();
    for term in &groups[0] {
        for combo in &rest {
            let mut v = vec![term.clone()];
            v.extend(combo.iter().cloned());
            result.push(v);
        }
    }
    result
}

/// Vector retrieval via the `vec0` virtual table's KNN operator.
///
/// Uses sqlite-vec's official `AND k = ?` syntax; rows already arrive in
/// distance-ascending order, so no `ORDER BY` clause is needed.
///
/// Returns Stage 1 [`Candidate`]s tagged with [`CandidateSource::Vector`].
/// The `score` field carries the raw distance (lower is better) and `rank`
/// is 0-based — fed verbatim to [`WeightedRrf`].
fn retrieve_vec<E: Embed>(
    conn: &Connection,
    embedder: &E,
    query: &str,
    limit: usize,
) -> Result<Vec<Candidate>, PipelineError> {
    let embedding = embedder.embed_query(query)?;
    let bytes = f32_as_bytes(&embedding);
    let k_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut stmt = conn.prepare_cached(
        "SELECT doc_id, distance FROM vec_docs WHERE embedding MATCH ? AND k = ?",
    )?;
    let rows = stmt.query_map(params![bytes, k_i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;
    let mut hits = Vec::new();
    for (rank, row) in rows.enumerate() {
        let (doc_id, score) = row?;
        hits.push(Candidate {
            source: CandidateSource::Vector,
            doc_id,
            score,
            rank,
        });
    }
    Ok(hits)
}

/// Rescore `merged` via `reranker.rerank(query, doc_bodies)`.
///
/// Consumes `merged` and yields a fresh `Vec<MergedHit>` with rerank scores
/// in place of RRF scores. Filters by `corpus_index` membership before
/// scoring, so the reranker's returned `index` aligns with the same filtered
/// slice and no missing-from-corpus entry can misalign rerank output with
/// merged identity. Rerank score (`f32`) widens to `f64` to keep the merged
/// list type consistent.
fn apply_reranker<R: Rerank>(
    reranker: &R,
    query: &str,
    merged: Vec<MergedHit>,
    corpus_index: &HashMap<&str, &str>,
) -> Result<Vec<MergedHit>, PipelineError> {
    let resolved: Vec<(String, &str, HashMap<CandidateSource, f64>)> = merged
        .into_iter()
        .filter_map(|h| {
            corpus_index
                .get(h.doc_id.as_str())
                .map(|body| (h.doc_id, *body, h.source_scores))
        })
        .collect();
    let bodies: Vec<&str> = resolved.iter().map(|(_, body, _)| *body).collect();
    let ranked_results = reranker.rerank(query, &bodies)?;
    let mut reranked = Vec::with_capacity(ranked_results.len());
    for r in ranked_results {
        if let Some((doc_id, _, source_scores)) = resolved.get(r.index) {
            reranked.push(MergedHit {
                doc_id: doc_id.clone(),
                score: f64::from(r.score),
                source_scores: source_scores.clone(),
            });
        }
    }
    Ok(reranked)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{PipelineConfig, evaluate};
    use crate::embed::MockEmbedder;
    use crate::eval::fixture::{EvalDocument, EvalQuery};
    use crate::reranker::MockReranker;
    use crate::retrieval::IdentityAggregator;

    /// Build an [`EvalDocument`] with stub title / source. The body carries
    /// the surface text the FTS5 + vec wiring will see.
    fn make_document(id: &str, body: &str) -> EvalDocument {
        EvalDocument {
            id: id.to_owned(),
            title: format!("title for {id}"),
            body: body.to_owned(),
            category_hint: None,
            source: "test fixture".to_owned(),
        }
    }

    /// Build an [`EvalQuery`] with a single-doc relevance map and stub
    /// category / annotation. Distribution validation is not exercised here.
    fn make_query(id: &str, text: &str, relevant_doc: &str) -> EvalQuery {
        let mut relevance_map = HashMap::new();
        relevance_map.insert(relevant_doc.to_owned(), 1u8);
        EvalQuery {
            id: id.to_owned(),
            text: text.to_owned(),
            category: "C1".to_owned(),
            relevance_map,
            annotation: "test query".to_owned(),
        }
    }

    // T-011: evaluate_with_mock_embedder_returns_one_result_with_hits
    // FR-008: 5-doc corpus + 1 query + MockEmbedder + no reranker →
    //         single QueryResult whose ranked_hits is non-empty.
    // FR-009: pipeline must compose `prepare_match_query`, `rrf_merge`,
    //         `Embed::embed_query`, optional `Rerank::rerank` (verified by
    //         CI grep against this file in Phase 1d).
    #[test]
    fn evaluate_with_mock_embedder_returns_one_result_with_hits() {
        let corpus = vec![
            make_document("d1", "alpha document about retrieval"),
            make_document("d2", "beta document about ranking"),
            make_document("d3", "gamma document about indexing"),
            make_document("d4", "delta document about scoring"),
            make_document("d5", "epsilon document about evaluation"),
        ];
        let queries = vec![make_query("q1", "alpha retrieval", "d1")];
        let embedder = MockEmbedder::default();
        let reranker: Option<&MockReranker> = None;
        let aggregator = IdentityAggregator;
        let config = PipelineConfig { k: 5 };

        let result = evaluate(&corpus, &queries, &embedder, reranker, &aggregator, &config)
            .expect("pipeline must succeed with MockEmbedder + no reranker");

        assert_eq!(
            result.len(),
            1,
            "FR-008: one query in → exactly one QueryResult out, got {} results",
            result.len()
        );
        assert!(
            !result[0].ranked_hits.is_empty(),
            "FR-008: indexed corpus + valid query → ranked_hits must be non-empty, got empty"
        );
    }
}
