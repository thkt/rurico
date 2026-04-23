use std::collections::HashMap;
use std::f64::consts::LN_2;
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
    (-LN_2 * age_days.max(0.0) / half_life_days).exp()
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

/// FTS5 operator-like keywords that must not be vocab-expanded, and are dropped
/// when dangling (no non-operator neighbour on both sides).
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn is_fts5_operator(token: &str) -> bool {
    FTS5_OPERATORS
        .iter()
        .any(|op| token.eq_ignore_ascii_case(op))
}

/// Remove operator-like keywords that lack a non-operator neighbour on both
/// sides. Keeps operators sandwiched between real terms.
fn drop_dangling_operators(tokens: &[String]) -> Vec<&str> {
    tokens
        .iter()
        .enumerate()
        .filter(|&(i, t)| {
            if !is_fts5_operator(t) {
                return true;
            }
            let has_left = i > 0 && !is_fts5_operator(&tokens[i - 1]);
            let has_right = i + 1 < tokens.len() && !is_fts5_operator(&tokens[i + 1]);
            has_left && has_right
        })
        .map(|(_, t)| t.as_str())
        .collect()
}

/// A pre-processed FTS5 query string — intermediate representation between
/// raw user input and [`MatchFtsQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedFtsQuery(String);

impl SanitizedFtsQuery {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// A fully expanded and quoted FTS5 query string, safe to pass to `MATCH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchFtsQuery(String);

impl MatchFtsQuery {
    /// Borrow the inner query string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner `String`.
    pub fn into_string(self) -> String {
        self.0
    }
}

/// Reasons why [`prepare_match_query`] cannot produce a usable query.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SanitizeError {
    /// The input string was empty or contained only whitespace.
    #[error("search query is empty")]
    EmptyInput,
    /// The input contained tokens, but none survived sanitization
    /// (e.g. only `NEAR()` groups or prefix characters).
    #[error("search query contains no searchable terms")]
    NoSearchableTerms,
    /// The supplied vocab table name is not a safe SQL identifier.
    #[error("invalid vocab table name: {0:?}")]
    InvalidVocabTable(String),
    /// SQLite failed while consulting the vocab table for short-term expansion.
    #[error("fts vocab lookup failed: {0}")]
    VocabLookupFailed(String),
}

/// Neutralize FTS5 special syntax in user queries: `NEAR()` grouping,
/// start-of-column `^`, required `+` / excluded `-` prefixes, column-filter
/// colons, and unbalanced quotes. Operator-like keywords (`AND`, `OR`, `NOT`)
/// sandwiched between non-operator terms are preserved (unquoted); dangling
/// operators (e.g. leading `NOT`, trailing `OR` after NEAR removal) are dropped.
pub(crate) fn sanitize_fts_query(query: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    if query.trim().is_empty() {
        return Err(SanitizeError::EmptyInput);
    }
    let tokens: Vec<String> = strip_near_groups(query)
        .into_iter()
        .map(|w| {
            let stripped = w.trim_start_matches(['^', '+', '-']);
            let cleaned = stripped.trim_matches(['(', ')']);
            if (cleaned.contains(':') || cleaned.contains('-')) && !cleaned.starts_with('"') {
                let unquoted = cleaned.replace('"', "");
                format!("\"{unquoted}\"")
            } else {
                cleaned.to_owned()
            }
        })
        .filter(|w| !w.is_empty())
        .collect();
    let result = drop_dangling_operators(&tokens).join(" ");
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
/// Returns merged results sorted by RRF score descending, ties broken by key ascending.
pub fn rrf_merge<K: Clone + Eq + Hash + Ord>(
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
    results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    results
}

/// Wrap `s` in double quotes for FTS5 MATCH syntax, escaping internal `"` as `""`.
pub fn fts_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Sanitize user input and expand short terms into a query safe for FTS5 `MATCH`.
///
/// Combines `sanitize_fts_query` and `fts_expand_short_terms` into a single call.
///
/// `vocab_table` names the `fts5vocab` virtual table to consult for short-term
/// expansion (e.g. `"fts_chunks_vocab"` or `"messages_vocab"`). Must be created
/// with the `row` or `col` vocabulary type — the `instance` type lacks the
/// `cnt` column this query relies on. The value is interpolated into the SQL
/// unescaped.
///
/// # Errors
///
/// Returns:
/// - [`SanitizeError::EmptyInput`] if `query` is empty or whitespace-only
/// - [`SanitizeError::NoSearchableTerms`] if sanitization strips all searchable terms
///
/// SQLite failures while consulting the vocab table surface as `Err`, except
/// for the specific "no such table" case which degrades to "no expansion".
pub fn prepare_match_query(
    conn: &Connection,
    query: &str,
    vocab_table: &str,
) -> Result<MatchFtsQuery, SanitizeError> {
    let sanitized = sanitize_fts_query(query)?;
    fts_expand_short_terms(conn, &sanitized, vocab_table)
}

/// Expand a single short token via vocab prefix matching.
///
/// Returns `Some(expanded_group)` when matches are found, `None` when the
/// vocab lookup returns no results or the vocab table disappeared.
fn expand_token_via_vocab(
    stmt: &mut rusqlite::CachedStatement<'_>,
    token: &str,
) -> Result<Option<String>, SanitizeError> {
    let escaped = token
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    let pattern = format!("{escaped}%");
    match stmt
        .query_map([&pattern], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
    {
        Ok(terms) if !terms.is_empty() => {
            let quoted: Vec<String> = terms.iter().map(|t| fts_quote(t)).collect();
            Ok(Some(format!("({})", quoted.join(" OR "))))
        }
        Ok(_) => Ok(None),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg))) if msg.contains("no such table") => {
            Ok(None)
        }
        Err(e) => {
            Err(SanitizeError::VocabLookupFailed(e.to_string()))
        }
    }
}

fn is_valid_sql_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let head_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let tail_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    head_ok && tail_ok
}

/// Expand short terms (1-2 chars) via `fts5vocab` prefix matching.
/// Falls back to quoting as-is when the vocab table is unavailable.
///
/// `vocab_table` must be an `fts5vocab` of type `row` or `col` (the SQL
/// references the `term` and `cnt` columns). The name is interpolated into
/// the SQL unescaped.
///
pub(crate) fn fts_expand_short_terms(
    conn: &Connection,
    query: &SanitizedFtsQuery,
    vocab_table: &str,
) -> Result<MatchFtsQuery, SanitizeError> {
    if !is_valid_sql_identifier(vocab_table) {
        return Err(SanitizeError::InvalidVocabTable(vocab_table.to_owned()));
    }
    let sql = format!(
        "SELECT term FROM {vocab_table} \
         WHERE term LIKE ?1 ESCAPE '\\' \
         ORDER BY cnt DESC LIMIT 25"
    );
    let mut stmt = match conn.prepare_cached(&sql) {
        Ok(s) => Some(s),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg))) if msg.contains("no such table") => {
            None
        }
        Err(e) => {
            return Err(SanitizeError::VocabLookupFailed(e.to_string()));
        }
    };

    let mut parts = Vec::new();
    for token in query.as_str().split_whitespace() {
        // Length check covers AND/NOT (3 chars); operator guard adds OR (2 chars).
        if token.chars().count() >= 3 || is_fts5_operator(token) {
            parts.push(fts_quote(token));
            continue;
        }
        let expanded = match stmt.as_mut() {
            Some(s) => expand_token_via_vocab(s, token)?,
            None => None,
        };
        parts.push(expanded.unwrap_or_else(|| fts_quote(token)));
    }
    Ok(MatchFtsQuery(parts.join(" ")))
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
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 >= result[2].1);
    }

    #[test]
    fn rrf_merge_tied_scores_ordered_by_key() {
        let fts: Vec<(u32, f64)> = vec![(3, 1.0)];
        let vec: Vec<(u32, f64)> = vec![(1, 1.0)];
        let result = rrf_merge(&fts, &vec);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1, "lower key should come first on tie");
        assert_eq!(result[1].0, 3);
        assert_eq!(result[0].1, result[1].1, "scores must be equal");
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
        let result =
            fts_expand_short_terms(&conn, &sanitized("authentication"), "fts_chunks_vocab")
                .unwrap();
        assert_eq!(result.as_str(), "\"authentication\"");
    }

    #[test]
    fn expand_short_term_with_vocab_matches() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("au"), "fts_chunks_vocab").unwrap();
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
        let result = fts_expand_short_terms(&conn, &sanitized("zz"), "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"zz\"");
    }

    #[test]
    fn expand_mixed_short_and_long() {
        let conn = setup_fts_db();
        let result =
            fts_expand_short_terms(&conn, &sanitized("au login"), "fts_chunks_vocab").unwrap();
        assert!(result.as_str().contains("\"login\""), "{}", result.as_str());
        assert!(result.as_str().contains(" OR "), "{}", result.as_str());
    }

    #[test]
    fn expand_operator_like_terms_are_quoted_not_expanded() {
        let conn = setup_fts_db();
        // NOT (3 chars) and OR (2 chars) must both be quoted as-is,
        // not expanded via vocab (e.g. OR must not become "order" OR ...).
        let result =
            fts_expand_short_terms(&conn, &sanitized("NOT secret"), "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"NOT\" \"secret\"");

        let result = fts_expand_short_terms(&conn, &sanitized("OR"), "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"OR\"");

        let result =
            fts_expand_short_terms(&conn, &sanitized("foo or bar"), "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"foo\" \"or\" \"bar\"");
    }

    #[test]
    fn expand_special_chars_escaped() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("a%"), "fts_chunks_vocab").unwrap();
        assert!(
            result.as_str().contains("\"audit\"") || result.as_str() == "\"a%\"",
            "{}",
            result.as_str()
        );
    }

    #[test]
    fn expand_without_vocab_table_degrades() {
        let conn = Connection::open_in_memory().unwrap();
        let result =
            fts_expand_short_terms(&conn, &sanitized("au login"), "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"au\" \"login\"");
    }

    #[test]
    fn expand_single_char_term() {
        let conn = setup_fts_db();
        let result = fts_expand_short_terms(&conn, &sanitized("a"), "fts_chunks_vocab").unwrap();
        assert!(result.as_str().contains("\"audit\"") || result.as_str() == "\"a\"");
    }

    #[test]
    fn prepare_match_query_end_to_end() {
        let conn = setup_fts_db();
        let result = prepare_match_query(&conn, "au login", "fts_chunks_vocab").unwrap();
        assert!(result.as_str().contains("\"login\""), "{}", result.as_str());
        assert!(result.as_str().contains(" OR "), "{}", result.as_str());
    }

    #[test]
    fn prepare_match_query_empty_input() {
        let conn = setup_fts_db();
        assert_eq!(
            prepare_match_query(&conn, "", "fts_chunks_vocab"),
            Err(SanitizeError::EmptyInput)
        );
    }

    #[test]
    fn prepare_match_query_operators_are_quoted() {
        let conn = setup_fts_db();
        let result = prepare_match_query(&conn, "foo OR bar", "fts_chunks_vocab").unwrap();
        assert_eq!(result.as_str(), "\"foo\" \"OR\" \"bar\"");
    }

    #[test]
    fn prepare_match_query_rejects_invalid_vocab_table_name() {
        let conn = setup_fts_db();
        assert_eq!(
            prepare_match_query(&conn, "au", "1vocab"),
            Err(SanitizeError::InvalidVocabTable("1vocab".into()))
        );
    }

    #[test]
    fn prepare_match_query_surfaces_non_missing_vocab_errors() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE bad_vocab(term TEXT)", [])
            .unwrap();

        let result = prepare_match_query(&conn, "au", "bad_vocab");
        assert!(
            matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
            "expected vocab lookup failure, got {result:?}"
        );
    }

    #[test]
    fn expand_rejects_leading_digit_vocab_name() {
        let conn = setup_fts_db();
        assert_eq!(
            fts_expand_short_terms(&conn, &sanitized("au"), "1vocab"),
            Err(SanitizeError::InvalidVocabTable("1vocab".into()))
        );
    }

    #[test]
    fn expand_rejects_empty_vocab_name() {
        let conn = setup_fts_db();
        assert_eq!(
            fts_expand_short_terms(&conn, &sanitized("au"), ""),
            Err(SanitizeError::InvalidVocabTable("".into()))
        );
    }

    #[test]
    fn expand_surfaces_non_missing_vocab_errors() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE bad_vocab(term TEXT)", [])
            .unwrap();

        let result = fts_expand_short_terms(&conn, &sanitized("au"), "bad_vocab");
        assert!(
            matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
            "expected vocab lookup failure, got {result:?}"
        );
    }

    fn sanitized(s: &str) -> SanitizedFtsQuery {
        SanitizedFtsQuery(s.to_owned())
    }

    fn ok(s: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
        Ok(SanitizedFtsQuery(s.to_owned()))
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
    fn t011_sandwiched_operator_preserved() {
        assert_eq!(sanitize_fts_query("foo AND bar"), ok("foo AND bar"));
        assert_eq!(sanitize_fts_query("foo OR bar"), ok("foo OR bar"));
    }

    #[test]
    fn t012_dangling_operator_dropped() {
        // Leading/trailing operators without both neighbours are dropped.
        assert_eq!(sanitize_fts_query("NOT secret"), ok("secret"));
        assert_eq!(sanitize_fts_query("foo OR"), ok("foo"));
    }

    #[test]
    fn t012b_consecutive_operators_between_terms() {
        // Neither AND nor OR has a non-operator on both sides → both dropped.
        assert_eq!(sanitize_fts_query("foo AND OR bar"), ok("foo bar"));
        assert_eq!(sanitize_fts_query("NOT foo NOT"), ok("foo"));
    }

    #[test]
    fn t013_operator_only_returns_error() {
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
    fn t014_near_then_dangling_operator() {
        // "foo OR NEAR(bar baz)" → NEAR stripped → "foo OR" → OR dangling → "foo"
        assert_eq!(sanitize_fts_query("foo OR NEAR(bar baz)"), ok("foo"));
    }

    #[test]
    fn t015_case_insensitive_operators() {
        assert_eq!(sanitize_fts_query("foo or bar"), ok("foo or bar"));
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
