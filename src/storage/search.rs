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
/// Terms with 3+ chars are quoted as-is. Short terms are expanded by querying
/// the `fts_chunks_vocab` table for prefix matches (up to 25 results), joined with OR.
///
/// Requires an `fts_chunks_vocab` table created as:
/// ```sql
/// CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, instance);
/// ```
pub fn fts_expand_short_terms(conn: &Connection, sanitized: &str) -> String {
    let mut stmt = match conn.prepare(
        "SELECT term FROM fts_chunks_vocab \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25",
    ) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("warning: fts vocab expansion unavailable: {e}");
            None
        }
    };

    let mut parts = Vec::new();
    for token in sanitized.split_whitespace() {
        let upper = token.to_ascii_uppercase();
        if matches!(upper.as_str(), "AND" | "OR" | "NOT") {
            parts.push(token.to_string());
            continue;
        }
        if token.is_empty() {
            continue;
        }
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
            s.query_map([&pattern], |row| row.get::<_, String>(0))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                .ok()
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
        // Item 2 appears in both → highest score
        assert_eq!(result[0].0, 2);
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
}
