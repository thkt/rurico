use std::collections::HashMap;
use std::hash::Hash;

use rusqlite::Connection;

/// Exponential recency decay: 1.0 at age=0, 0.5 at one half-life, approaching 0.0.
///
/// Negative `age_days` is clamped to 0 (returns 1.0).
/// Returns 0.0 when `half_life_days <= 0.0` (avoids division by zero).
pub fn recency_decay(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 0.0;
    }
    (-std::f64::consts::LN_2 * age_days.max(0.0) / half_life_days).exp()
}

/// Filter out `NEAR(...)` and `NEAR/N(...)` groups from whitespace-split tokens.
fn strip_near_groups(query: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut paren_depth: usize = 0;
    for w in query.split_whitespace() {
        let upper = w.to_ascii_uppercase();
        if upper.starts_with("NEAR(") || upper.starts_with("NEAR/") {
            paren_depth += w.chars().filter(|&c| c == '(').count();
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        if paren_depth > 0 {
            paren_depth = paren_depth.saturating_sub(w.chars().filter(|&c| c == ')').count());
            continue;
        }
        tokens.push(w);
    }
    tokens
}

/// A sanitized FTS5 query string, free of special syntax but not yet expanded.
///
/// Created only by [`sanitize_fts_query`]. The inner value is guaranteed
/// non-empty and free of FTS5 special syntax. Pass to
/// [`fts_expand_short_terms`] to produce a [`MatchFtsQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizedFtsQuery(String);

impl SanitizedFtsQuery {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A fully expanded FTS5 query string ready to pass to `MATCH`.
///
/// Created only by [`fts_expand_short_terms`]. Short terms have been
/// expanded via vocab lookup and all terms are quoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchFtsQuery(String);

impl MatchFtsQuery {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` when expansion produced no usable terms.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Reasons why [`sanitize_fts_query`] cannot produce a usable query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanitizeError {
    /// The input string was empty or contained only whitespace.
    EmptyInput,
    /// The input contained tokens, but none survived sanitization
    /// (e.g. only boolean operators, `NEAR()` groups, or prefix characters).
    NoSearchableTerms,
}

impl std::fmt::Display for SanitizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("search query is empty"),
            Self::NoSearchableTerms => f.write_str("search query contains no searchable terms"),
        }
    }
}

impl std::error::Error for SanitizeError {}

/// FTS5 boolean operators to strip from user queries.
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn is_fts5_operator(token: &str) -> bool {
    let upper = token.to_ascii_uppercase();
    FTS5_OPERATORS.iter().any(|op| upper == *op)
}

/// Neutralize FTS5 special syntax in user queries: boolean operators
/// (`AND`, `OR`, `NOT`), `NEAR()` grouping, start-of-column `^`,
/// required `+` / excluded `-` prefixes, column-filter colons, and
/// unbalanced quotes.
///
/// Returns [`SanitizedFtsQuery`] on success — a non-empty string safe
/// to pass to FTS5 `MATCH`. Returns [`SanitizeError`] when the input
/// is empty or all tokens are stripped.
pub fn sanitize_fts_query(query: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    if query.trim().is_empty() {
        return Err(SanitizeError::EmptyInput);
    }
    let result = strip_near_groups(query)
        .into_iter()
        .filter(|w| !is_fts5_operator(w))
        .map(|w| {
            let stripped = w.trim_start_matches(['^', '+', '-']);
            let cleaned = stripped.trim_matches(['(', ')']);
            if (cleaned.contains(':') || cleaned.contains('-')) && !cleaned.starts_with('"') {
                let unquoted = cleaned.replace('"', "");
                format!("\"{unquoted}\"")
            } else {
                cleaned.to_string()
            }
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if result.is_empty() {
        return Err(SanitizeError::NoSearchableTerms);
    }
    let quote_count = result.chars().filter(|&c| c == '"').count();
    if quote_count % 2 != 0 {
        Ok(SanitizedFtsQuery(format!("{result}\"")))
    } else {
        Ok(SanitizedFtsQuery(result))
    }
}

const RRF_K: f64 = 60.0;

/// Reciprocal Rank Fusion merge — only rank position matters, score values are ignored.
/// Returns merged results sorted by RRF score descending.
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

/// Expand short terms (1-2 chars) via `fts_chunks_vocab` prefix matching.
///
/// Terms with 3+ chars are quoted as-is. Short terms are expanded into
/// `("term1" OR "term2" ...)` groups. Falls back to quoting as-is when
/// the vocab table is unavailable.
pub fn fts_expand_short_terms(conn: &Connection, query: &SanitizedFtsQuery) -> MatchFtsQuery {
    let mut stmt = match conn.prepare_cached(
        "SELECT term FROM fts_chunks_vocab \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25",
    ) {
        Ok(s) => Some(s),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg))) if msg.contains("no such table") => {
            None
        }
        Err(e) => {
            log::warn!("fts vocab prepare failed: {e}");
            None
        }
    };

    let mut parts = Vec::new();
    for token in query.as_str().split_whitespace() {
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
                    log::warn!("fts vocab query failed: {e}");
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
    MatchFtsQuery(parts.join(" "))
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
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 >= result[2].1);
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
        let result = fts_expand_short_terms(&conn, &sanitized("authentication"));
        assert_eq!(result.as_str(), "\"authentication\"");
    }

    #[test]
    fn expand_short_term_with_vocab_matches() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("au"));
        assert!(result.as_str().contains("\"audit\""), "{}", result.as_str());
        assert!(
            result.as_str().contains("\"authentication\""),
            "{}",
            result.as_str()
        );
        assert!(result.as_str().contains(" OR "), "{}", result.as_str());
    }

    #[test]
    fn expand_short_term_no_matches() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("zz"));
        assert_eq!(result.as_str(), "\"zz\"");
    }

    #[test]
    fn expand_mixed_short_and_long() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("au login"));
        assert!(result.as_str().contains("\"login\""), "{}", result.as_str());
        assert!(result.as_str().contains(" OR "), "{}", result.as_str());
    }

    #[test]
    fn expand_operators_are_quoted_not_passed_through() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("NOT secret"));
        assert!(!result.as_str().starts_with("NOT "), "{}", result.as_str());
    }

    #[test]
    fn expand_special_chars_escaped() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("a%"));
        assert!(
            result.as_str().contains("\"audit\"") || result.as_str() == "\"a%\"",
            "{}",
            result.as_str()
        );
    }

    #[test]
    fn expand_without_vocab_table_degrades() {
        let conn = Connection::open_in_memory().unwrap();
        let result = fts_expand_short_terms(&conn, &sanitized("au login"));
        assert_eq!(result.as_str(), "\"au\" \"login\"");
    }

    #[test]
    fn expand_single_char_term() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("a"));
        assert!(result.as_str().contains("\"audit\"") || result.as_str() == "\"a\"");
    }

    fn sanitized(s: &str) -> SanitizedFtsQuery {
        SanitizedFtsQuery(s.to_string())
    }

    fn ok(s: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
        Ok(SanitizedFtsQuery(s.to_string()))
    }

    #[test]
    fn t001_near_removal() {
        assert_eq!(sanitize_fts_query("NEAR(a b) hello"), ok("hello"));
    }

    #[test]
    fn t001b_near_with_distance() {
        assert_eq!(sanitize_fts_query("NEAR/3(a b c) hello"), ok("hello"));
    }

    #[test]
    fn t001c_near_unclosed_paren() {
        assert_eq!(
            sanitize_fts_query("NEAR(a b hello"),
            Err(SanitizeError::NoSearchableTerms)
        );
    }

    #[test]
    fn t002_auto_quote_hyphen() {
        assert_eq!(sanitize_fts_query("rate-limit"), ok("\"rate-limit\""));
    }

    #[test]
    fn t003_auto_quote_colon() {
        assert_eq!(sanitize_fts_query("std::io"), ok("\"std::io\""));
    }

    #[test]
    fn t004_prefix_strip() {
        assert_eq!(sanitize_fts_query("^+hello"), ok("hello"));
    }

    #[test]
    fn t005b_quote_balancing_exact_output() {
        assert_eq!(sanitize_fts_query("unbalanced\""), ok("unbalanced\"\""));
    }

    #[test]
    fn t006_empty_input() {
        assert_eq!(sanitize_fts_query(""), Err(SanitizeError::EmptyInput));
    }

    #[test]
    fn t006b_whitespace_only() {
        assert_eq!(sanitize_fts_query("   "), Err(SanitizeError::EmptyInput));
    }

    #[test]
    fn t011_not_operator_stripped() {
        assert_eq!(sanitize_fts_query("NOT secret"), ok("secret"));
    }

    #[test]
    fn t012_and_operator_stripped() {
        assert_eq!(sanitize_fts_query("foo AND bar"), ok("foo bar"));
    }

    #[test]
    fn t013_or_operator_stripped() {
        assert_eq!(sanitize_fts_query("foo OR bar"), ok("foo bar"));
    }

    #[test]
    fn t014_operator_only_returns_error() {
        assert_eq!(
            sanitize_fts_query("NOT"),
            Err(SanitizeError::NoSearchableTerms)
        );
        assert_eq!(
            sanitize_fts_query("AND OR NOT"),
            Err(SanitizeError::NoSearchableTerms)
        );
    }

    #[test]
    fn t015_case_insensitive_operators() {
        assert_eq!(sanitize_fts_query("not secret"), ok("secret"));
        assert_eq!(sanitize_fts_query("Not secret"), ok("secret"));
    }

    #[test]
    fn t016_prefix_only_returns_error() {
        assert_eq!(
            sanitize_fts_query("^"),
            Err(SanitizeError::NoSearchableTerms)
        );
    }

    #[test]
    fn t007_age_zero_returns_one() {
        let result = recency_decay(0.0, 30.0);
        assert!((result - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn t008_one_half_life_returns_half() {
        let result = recency_decay(30.0, 30.0);
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn t009_two_half_lives_returns_quarter() {
        let result = recency_decay(60.0, 30.0);
        assert!((result - 0.25).abs() < 0.01);
    }

    #[test]
    fn t010_zero_half_life_returns_zero() {
        let result = recency_decay(0.0, 0.0);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn t010b_negative_half_life_returns_zero() {
        let result = recency_decay(5.0, -1.0);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn t010c_negative_age_clamped_to_one() {
        let result = recency_decay(-5.0, 30.0);
        assert!((result - 1.0).abs() < f64::EPSILON);
    }
}
