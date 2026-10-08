use super::*;

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
        fts_expand_short_terms(&conn, &sanitized("authentication"), "fts_chunks_vocab").unwrap();
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
fn expand_operator_like_terms_are_quoted_not_expanded() {
    let conn = setup_fts_db();
    // NOT (3 chars) and OR (2 chars) must both be quoted as-is,
    // not expanded via vocab (e.g. OR must not become "order" OR ...).
    let result =
        fts_expand_short_terms(&conn, &sanitized("NOT secret"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"NOT\" AND \"secret\"");

    let result = fts_expand_short_terms(&conn, &sanitized("OR"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"OR\"");

    let result =
        fts_expand_short_terms(&conn, &sanitized("foo or bar"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"foo\" AND \"or\" AND \"bar\"");
}

#[test]
fn expand_special_chars_escaped() {
    let conn = setup_fts_db();
    let result = fts_expand_short_terms(&conn, &sanitized("a%"), "fts_chunks_vocab").unwrap();
    assert_eq!(result.as_str(), "\"a%\"");
}

/// Tests below pin the sanitize / expand contract — pass `disabled()` so
/// changes to the Phase 5 normalization defaults can never alter what
/// these tests measure.
fn no_norm() -> QueryNormalizationConfig {
    QueryNormalizationConfig::disabled()
}

// #297: synthetic input, serialized MATCH, and exact hits are kept together.
// unicode61 discards punctuation; trigram exposes accidental extra quotes.
// Each pair includes a distractor so dropping punctuation is also detectable.
#[test]
fn prepare_match_query_literals_execute_with_both_tokenizers() {
    let cases = [
        (
            "rate-limit",
            r#""rate-limit""#,
            ["rate-limit", "rate limit"],
        ),
        ("std::io", r#""std::io""#, ["std::io", "std io"]),
        ("say\"hi", r#""say""hi""#, ["say\"hi", "say hi"]),
        ("a\"b-c", r#""a""b-c""#, ["a\"b-c", "a b c"]),
        (
            "unbalanced\"",
            r#""unbalanced""""#,
            ["unbalanced\"", "unbalanced"],
        ),
        (
            "\"rate-limit\"",
            r#""""rate-limit""""#,
            ["\"rate-limit\"", "rate-limit"],
        ),
        (
            "(rate-limit)",
            r#""rate-limit""#,
            ["rate-limit", "rate limit"],
        ),
        ("abc(def)", r#""abc(def""#, ["abc(def)", "abc def"]),
        ("abc*def%_", r#""abc*def%_""#, ["abc*def%_", "abc def"]),
        (
            " \trate-limit\nlogin　",
            r#""rate-limit" AND "login""#,
            ["rate-limit login", "rate limit login"],
        ),
    ];
    for tokenizer in ["unicode61", "trigram"] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}');
             CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row);"
        ))
        .unwrap();
        for (input, expected_match, bodies) in cases {
            conn.execute("DELETE FROM docs", []).unwrap();
            for (id, body) in [1, 2].into_iter().zip(bodies) {
                conn.execute("INSERT INTO docs(rowid, body) VALUES (?1, ?2)", (id, body))
                    .unwrap();
            }
            let query = prepare_match_query(&conn, input, "vocab", &no_norm()).unwrap();
            assert_eq!(query.as_str(), expected_match, "{tokenizer}: {input:?}");
            let expected_hits: &[i64] = if tokenizer == "unicode61" {
                &[1, 2]
            } else {
                &[1]
            };
            assert_eq!(
                match_ids(&conn, &query),
                expected_hits,
                "{tokenizer}: {input:?}"
            );
        }
    }
}

fn match_ids(conn: &Connection, query: &MatchFtsQuery) -> Vec<i64> {
    conn.prepare("SELECT rowid FROM docs WHERE docs MATCH ?1 ORDER BY rowid")
        .unwrap()
        .query_map([query.as_str()], |row| row.get(0))
        .unwrap_or_else(|e| panic!("MATCH rejected {:?}: {e}", query.as_str()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

// #314 research uses the same public synthetic inputs as the SQLite probe.
// Cross-check its legacy wire/hits against the real public producer rather
// than treating a Python rendering of current SQL as product verification.
#[test]
fn research_query_plan_legacy_matches_shared_observations() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("../../../docs/research/issue-314/cases.json")).unwrap();
    let results: serde_json::Value = serde_json::from_str(include_str!(
        "../../../docs/research/issue-314/results.json"
    ))
    .unwrap();
    for tokenizer in ["unicode61", "trigram"] {
        for case in cases.as_array().unwrap() {
            if !["phrase", "cap-ties", "percent", "underscore", "backslash"]
                .contains(&case["name"].as_str().unwrap())
            {
                continue;
            }
            let conn = Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}');
                 CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row);"
            ))
            .unwrap();
            for body in case["bodies"].as_array().unwrap() {
                conn.execute("INSERT INTO docs VALUES (?1)", [body.as_str().unwrap()])
                    .unwrap();
            }
            for mode in ["legacy", "missing-vocab"] {
                let observed = results["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|row| {
                        row["case"] == case["name"]
                            && row["tokenizer"] == tokenizer
                            && row["mode"] == mode
                    })
                    .unwrap();
                let vocab = if mode == "legacy" { "vocab" } else { "absent" };
                let query =
                    prepare_match_query(&conn, case["input"].as_str().unwrap(), vocab, &no_norm())
                        .unwrap();
                let actual = match_ids(&conn, &query);
                if case["name"] == "cap-ties" && mode == "legacy" {
                    // cnt ties have no promised order. Do not freeze this
                    // SQLite version's choice of the 25 terms into the API.
                    assert_eq!(query.as_str().split(" OR ").count(), 25);
                    assert_eq!(actual.len(), 50);
                    assert!(!actual.contains(&61), "rare term must be below the cap");
                } else {
                    assert_eq!(query.as_str(), observed["wire"].as_str().unwrap());
                    let expected: Vec<i64> = observed["hits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|id| id.as_i64().unwrap())
                        .collect();
                    assert_eq!(actual, expected, "{tokenizer}, {mode}, {}", case["name"]);
                }
            }
        }
    }
}

#[test]
fn prepare_match_query_japanese_expansion_executes_with_both_tokenizers() {
    for tokenizer in ["unicode61", "trigram"] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}');
             INSERT INTO docs VALUES ('日本語 rate-limit'), ('日本海 other'), ('日本語');
             CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row);"
        ))
        .unwrap();
        // Distinct frequencies make the existing cnt-descending order deterministic.
        for input in ["日 rate-limit", "日本 rate-limit"] {
            let query = prepare_match_query(&conn, input, "vocab", &no_norm()).unwrap();
            assert_eq!(query.as_str(), r#"("日本語" OR "日本海") AND "rate-limit""#);
            assert_eq!(match_ids(&conn, &query), [1], "{tokenizer}: {input}");
        }
        let query = prepare_match_query(&conn, "日本 日", "vocab", &no_norm()).unwrap();
        assert_eq!(
            query.as_str(),
            r#"("日本語" OR "日本海") AND ("日本語" OR "日本海")"#
        );
        assert_eq!(match_ids(&conn, &query), [1, 2, 3], "{tokenizer}");

        let query =
            prepare_match_query(&conn, "日本 rate-limit", "missing_vocab", &no_norm()).unwrap();
        assert_eq!(query.as_str(), r#""日本" AND "rate-limit""#);
        // No full unicode61 term / fewer than three trigram characters: no expansion, no hit.
        assert!(match_ids(&conn, &query).is_empty(), "{tokenizer}");
    }
}

// #224: a vocab-expanded group adjacent to another token must form a valid
// FTS5 expression. Implicit AND rejects parenthesised groups as operands
// (`(a OR b) c` is an fts5 syntax error), so the query must execute, not
// just look right as a string.
#[test]
fn prepare_match_query_expanded_group_is_executable() {
    let conn = setup_fts_db();
    let query = prepare_match_query(&conn, "au login", "fts_chunks_vocab", &no_norm()).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH ?1",
            [query.as_str()],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| panic!("MATCH rejected {:?}: {e}", query.as_str()));
    assert_eq!(
        count,
        1,
        "expected exactly 'authentication login session' to match, query: {:?}",
        query.as_str()
    );
}

#[test]
fn prepare_match_query_empty_input() {
    let conn = setup_fts_db();
    for input in ["", " \t\n　"] {
        assert_eq!(
            prepare_match_query(&conn, input, "fts_chunks_vocab", &no_norm()),
            Err(SanitizeError::EmptyInput)
        );
    }
    for input in ["NEAR(a b)", "^", "AND OR NOT"] {
        assert_eq!(
            prepare_match_query(&conn, input, "fts_chunks_vocab", &no_norm()),
            Err(SanitizeError::NoSearchableTerms)
        );
    }
}

#[test]
fn prepare_match_query_operators_are_quoted() {
    for tokenizer in ["unicode61", "trigram"] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}');
             INSERT INTO docs VALUES ('foo OR bar'), ('foo bar'), ('foo order bar'),
                                     ('foo'), ('bar'), ('foo NOT bar');
             CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row);"
        ))
        .unwrap();
        let query = prepare_match_query(&conn, "foo OR bar", "vocab", &no_norm()).unwrap();
        assert_eq!(query.as_str(), r#""foo" AND "OR" AND "bar""#);
        // OR stays literal and unexpanded (not "order"). Trigram cannot match two chars.
        let expected: &[i64] = if tokenizer == "unicode61" { &[1] } else { &[] };
        assert_eq!(match_ids(&conn, &query), expected, "{tokenizer}");

        let query = prepare_match_query(&conn, "foo NOT bar", "vocab", &no_norm()).unwrap();
        assert_eq!(query.as_str(), r#""foo" AND "NOT" AND "bar""#);
        assert_eq!(match_ids(&conn, &query), [6], "{tokenizer}");
    }
}

#[test]
fn prepare_match_query_rejects_invalid_vocab_table_name() {
    let conn = setup_fts_db();
    for vocab in ["", "1vocab"] {
        assert_eq!(
            prepare_match_query(&conn, "au", vocab, &no_norm()),
            Err(SanitizeError::InvalidVocabTable(vocab.into()))
        );
    }
}

#[test]
fn prepare_match_query_surfaces_non_missing_vocab_errors() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE bad_vocab(term TEXT)", [])
        .unwrap();

    let result = prepare_match_query(&conn, "au", "bad_vocab", &no_norm());
    assert!(
        matches!(result, Err(SanitizeError::VocabLookupFailed(_))),
        "expected vocab lookup failure, got {result:?}"
    );
}

#[test]
fn prepare_match_query_with_missing_vocab_degrades() {
    let conn = Connection::open_in_memory().unwrap();
    let result = prepare_match_query(&conn, "au login", "fts_chunks_vocab", &no_norm()).unwrap();
    assert_eq!(result.as_str(), "\"au\" AND \"login\"");
}

#[test]
fn prepare_match_query_default_normalizes_fullwidth_input() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='trigram');
         INSERT INTO docs VALUES ('rate-limit');",
    )
    .unwrap();
    // Phase 5 defaults must fold full-width letters/punctuation and lowercase
    // before sanitization, without adding literal quotes around the hyphen.
    let result = prepare_match_query(
        &conn,
        "ＲＡＴＥ－ＬＩＭＩＴ",
        "fts_chunks_vocab",
        &QueryNormalizationConfig::default(),
    )
    .unwrap();
    assert_eq!(result.as_str(), "\"rate-limit\"");
    assert_eq!(match_ids(&conn, &result), [1]);
}

fn prepare_err(conn: &Connection, sql: &str) -> rusqlite::Error {
    let Err(e) = conn.prepare_cached(sql) else {
        panic!("expected prepare_cached to fail for: {sql}");
    };
    e
}

#[test]
fn is_missing_table_error_detects_sqlite_no_such_table() {
    let conn = Connection::open_in_memory().unwrap();
    let err = prepare_err(&conn, "SELECT * FROM definitely_nonexistent");
    assert!(is_missing_table_error(&err), "got {err:?}");
}

#[test]
fn is_missing_table_error_rejects_syntax_error() {
    let conn = Connection::open_in_memory().unwrap();
    let err = prepare_err(&conn, "SELECT FROM");
    assert!(!is_missing_table_error(&err), "got {err:?}");
}

#[test]
fn is_missing_table_error_rejects_missing_column() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t(x INTEGER)", []).unwrap();
    let err = prepare_err(&conn, "SELECT y FROM t");
    assert!(!is_missing_table_error(&err), "got {err:?}");
}

fn sanitized(s: &str) -> SanitizedFtsQuery {
    SanitizedFtsQuery(s.split_whitespace().map(str::to_owned).collect())
}

fn ok(s: &str) -> Result<SanitizedFtsQuery, SanitizeError> {
    Ok(sanitized(s))
}

#[test]
fn near_removal() {
    assert_eq!(sanitize_fts_query("NEAR(a b) hello"), ok("hello"));
}

#[test]
fn near_with_distance() {
    assert_eq!(sanitize_fts_query("NEAR/3(a b c) hello"), ok("hello"));
}

#[test]
fn near_unclosed_paren() {
    assert_eq!(
        sanitize_fts_query("NEAR(a b hello"),
        Err(SanitizeError::NoSearchableTerms)
    );
}

#[test]
fn prefix_strip() {
    assert_eq!(sanitize_fts_query("^+hello"), ok("hello"));
}

#[test]
fn sandwiched_operator_preserved() {
    assert_eq!(sanitize_fts_query("foo AND bar"), ok("foo AND bar"));
    assert_eq!(sanitize_fts_query("foo OR bar"), ok("foo OR bar"));
}

#[test]
fn dangling_operator_dropped() {
    // Leading/trailing operators without both neighbours are dropped.
    assert_eq!(sanitize_fts_query("NOT secret"), ok("secret"));
    assert_eq!(sanitize_fts_query("foo OR"), ok("foo"));
}

#[test]
fn consecutive_operators_between_terms() {
    // Neither AND nor OR has a non-operator on both sides → both dropped.
    assert_eq!(sanitize_fts_query("foo AND OR bar"), ok("foo bar"));
    assert_eq!(sanitize_fts_query("NOT foo NOT"), ok("foo"));
}

#[test]
fn operator_only_returns_error() {
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
fn near_then_dangling_operator() {
    // "foo OR NEAR(bar baz)" → NEAR stripped → "foo OR" → OR dangling → "foo"
    assert_eq!(sanitize_fts_query("foo OR NEAR(bar baz)"), ok("foo"));
}

#[test]
fn case_insensitive_operators() {
    assert_eq!(sanitize_fts_query("foo or bar"), ok("foo or bar"));
    assert_eq!(sanitize_fts_query("Not secret"), ok("secret"));
}

//
// Boundary: a valid FTS5 operator at index 0 cannot have a left neighbour,
// so `has_left` is false and the operator must be dropped even when a
// non-operator follows. Pins the `i > 0` short-circuit against a future
// off-by-one rewrite that would let leading `AND` / `OR` slip through and
// form an invalid FTS5 expression.
#[test]
fn drop_dangling_operators_drops_operator_at_position_zero() {
    let tokens: Vec<String> = vec!["AND".into(), "foo".into(), "bar".into()];
    let result = drop_dangling_operators(&tokens);
    assert_eq!(
        result,
        vec!["foo", "bar"],
        "operator at index 0 has no left neighbour → must drop"
    );
}
