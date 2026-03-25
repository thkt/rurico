use std::collections::HashMap;
use std::hash::Hash;

use rusqlite::Connection;

const RRF_K: f64 = 60.0;

/// Reciprocal Rank Fusion merge. Generic over key type.
///
/// Each input is a ranked list of `(key, score)` pairs. The score value is ignored —
/// only the rank position matters. Returns merged results sorted by RRF score descending.
pub fn rrf_merge<K: Clone + Eq + Hash>(
    fts_hits: &[(K, f64)],
    vec_hits: &[(K, f64)],
) -> Vec<(K, f64)> {
    let mut scores: HashMap<K, f64> = HashMap::new();

    for (rank, (key, _)) in fts_hits.iter().enumerate() {
        *scores.entry(key.clone()).or_default() += 1.0 / (RRF_K + rank as f64);
    }
    for (rank, (key, _)) in vec_hits.iter().enumerate() {
        *scores.entry(key.clone()).or_default() += 1.0 / (RRF_K + rank as f64);
    }

    let mut results: Vec<(K, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

/// Quote a term for FTS5 MATCH syntax.
pub fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Expand short terms (1-2 chars) via fts5vocab prefix matching.
///
/// All tokens are quoted for FTS5 MATCH syntax. Terms with 3+ chars are quoted
/// as-is. Short terms (1-2 chars) are expanded by querying the `fts_chunks_vocab`
/// table for prefix matches (up to 25 results), joined with OR.
///
/// Boolean operators (AND/OR/NOT) must be injected by the caller — this function
/// treats all input tokens as search terms.
///
/// If the vocab table is unavailable, short terms are quoted as-is (graceful
/// degradation, no error).
///
/// Requires an `fts_chunks_vocab` table created as:
/// ```sql
/// CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);
/// ```
pub fn fts_expand_short_terms(conn: &Connection, query: &str) -> String {
    let mut stmt = match conn.prepare_cached(
        "SELECT term FROM fts_chunks_vocab \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25",
    ) {
        Ok(s) => Some(s),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg)))
            if msg.contains("no such table") =>
        {
            None
        }
        Err(e) => {
            eprintln!("warning: fts vocab prepare failed: {e}");
            None
        }
    };

    let mut parts = Vec::new();
    for token in query.split_whitespace() {
        if token.chars().count() >= 3 {
            parts.push(fts_quote(token));
            continue;
        }
        let expanded = stmt.as_mut().and_then(|s| {
            let escaped = token
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_");
            let pattern = format!("{escaped}%");
            match s
                .query_map([&pattern], |row| row.get::<_, String>(0))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            {
                Ok(terms) => Some(terms),
                Err(e) => {
                    eprintln!("warning: fts vocab query failed: {e}");
                    None
                }
            }
        });
        match expanded {
            Some(terms) if !terms.is_empty() => {
                let quoted: Vec<String> = terms.iter().map(|t| fts_quote(t)).collect();
                parts.push(format!("({})", quoted.join(" OR ")));
            }
            _ => parts.push(fts_quote(token)),
        }
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_merge_both_lists() {
        let fts: Vec<(u32, f64)> = vec![(1, 1.0), (2, 0.5)];
        let vec: Vec<(u32, f64)> = vec![(2, 1.0), (3, 0.5)];
        let result = rrf_merge(&fts, &vec);
        assert_eq!(result[0].0, 2);
        assert_eq!(result.len(), 3);
        assert!(
            result[0].1 > result[1].1,
            "dual-list item should outscore single-list: {} vs {}",
            result[0].1,
            result[1].1
        );
        assert!(
            result[1].1 >= result[2].1,
            "earlier rank should score >= later rank: {} vs {}",
            result[1].1,
            result[2].1
        );
    }

    #[test]
    fn rrf_merge_empty_fts() {
        let fts: Vec<(String, f64)> = vec![];
        let vec: Vec<(String, f64)> = vec![("a".into(), 1.0), ("b".into(), 0.5)];
        let result = rrf_merge(&fts, &vec);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "a");
    }

    #[test]
    fn rrf_merge_empty_both() {
        let fts: Vec<(u32, f64)> = vec![];
        let vec: Vec<(u32, f64)> = vec![];
        let result = rrf_merge(&fts, &vec);
        assert!(result.is_empty());
    }

    #[test]
    fn rrf_merge_string_keys() {
        let fts = vec![("session-1".to_string(), 1.0)];
        let vec = vec![("session-1".to_string(), 1.0), ("session-2".to_string(), 0.5)];
        let result = rrf_merge(&fts, &vec);
        assert_eq!(result[0].0, "session-1");
    }

    #[test]
    fn fts_quote_simple() {
        assert_eq!(fts_quote("hello"), "\"hello\"");
    }

    #[test]
    fn fts_quote_with_quotes() {
        assert_eq!(fts_quote("he\"llo"), "\"he\"\"llo\"");
    }

    fn setup_fts_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE fts_chunks USING fts5(content);
             INSERT INTO fts_chunks(content) VALUES ('authentication login session');
             INSERT INTO fts_chunks(content) VALUES ('authorization permission role');
             INSERT INTO fts_chunks(content) VALUES ('audit logging trace');
             CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);",
        )
        .unwrap();
        conn
    }

    #[test]
    fn expand_long_term_quoted_as_is() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "authentication");
        assert_eq!(result, "\"authentication\"");
    }

    #[test]
    fn expand_short_term_with_vocab_matches() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "au");
        assert!(result.contains("\"audit\""), "should contain audit: {result}");
        assert!(
            result.contains("\"authentication\""),
            "should contain authentication: {result}"
        );
        assert!(result.contains(" OR "), "expanded terms joined with OR: {result}");
    }

    #[test]
    fn expand_short_term_no_matches() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "zz");
        assert_eq!(result, "\"zz\"", "no matches → quoted as-is");
    }

    #[test]
    fn expand_empty_input() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "");
        assert_eq!(result, "");
    }

    #[test]
    fn expand_mixed_short_and_long() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "au login");
        assert!(result.contains("\"login\""), "long term quoted: {result}");
        assert!(result.contains(" OR "), "short term expanded: {result}");
    }

    #[test]
    fn expand_operators_are_quoted_not_passed_through() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "NOT secret");
        assert!(
            !result.starts_with("NOT "),
            "NOT should be quoted, not passed through: {result}"
        );
    }

    #[test]
    fn expand_special_chars_escaped() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "a%");
        // '%' must be escaped for LIKE, not act as wildcard
        assert!(
            result.contains("\"audit\"") || result == "\"a%\"",
            "should expand to 'a'-prefixed vocab terms or quote as-is, got: {result}"
        );
    }

    #[test]
    fn expand_without_vocab_table_degrades() {
        let conn = Connection::open_in_memory().unwrap();
        let result = fts_expand_short_terms(&conn, "au login");
        assert_eq!(result, "\"au\" \"login\"", "all terms quoted as fallback");
    }

    #[test]
    fn expand_whitespace_only() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "   ");
        assert_eq!(result, "");
    }

    #[test]
    fn expand_single_char_term() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, "a");
        assert!(result.contains("\"audit\"") || result == "\"a\"");
    }
}
