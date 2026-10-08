use super::*;

fn run(text: &str) -> String {
    normalize_for_fts(text, &QueryNormalizationConfig::default())
}

#[test]
fn pre_phase_5_disabled_matches_disabled() {
    assert_eq!(pre_phase_5_disabled(), QueryNormalizationConfig::disabled());
}

#[test]
fn fullwidth_latin_to_halfwidth() {
    assert_eq!(run("ＡＢＣ"), "abc");
}

#[test]
fn fullwidth_digits_to_halfwidth() {
    assert_eq!(run("１２３"), "123");
}

#[test]
fn ascii_uppercase_lowered() {
    assert_eq!(run("React Hooks"), "react hooks");
}

#[test]
fn hiragana_unchanged() {
    assert_eq!(run("ひらがな"), "ひらがな");
}

#[test]
fn katakana_unchanged() {
    // NFKC folds halfwidth katakana but does not map Hiragana ↔ Katakana.
    assert_eq!(run("カタカナ"), "カタカナ");
}

#[test]
fn halfwidth_katakana_to_fullwidth() {
    assert_eq!(run("ｱｲｳｴｵ"), "アイウエオ");
}

#[test]
fn ideographic_space_collapsed() {
    assert_eq!(run("foo　bar"), "foo bar");
}

#[test]
fn whitespace_runs_collapsed() {
    assert_eq!(run("foo   bar\t\nbaz"), "foo bar baz");
}

#[test]
fn leading_trailing_whitespace_trimmed() {
    assert_eq!(run("  hello  "), "hello");
}

#[test]
fn empty_input_stays_empty() {
    assert_eq!(run(""), "");
}

#[test]
fn whitespace_only_collapses_to_empty() {
    assert_eq!(run("   "), "");
}

#[test]
fn ascii_passthrough_idempotent() {
    let s = "react hooks";
    assert_eq!(run(s), s);
    assert_eq!(run(&run(s)), run(s));
}

#[test]
fn idempotent_on_japanese_drift() {
    let s = "ＲｅａｃｔのHooks";
    let once = run(s);
    let twice = run(&once);
    assert_eq!(once, twice, "normalize must be a fixed point");
}

#[test]
fn idempotent_on_mixed_whitespace() {
    let s = "foo　　bar  baz";
    let once = run(s);
    let twice = run(&once);
    assert_eq!(once, twice);
}

#[test]
fn disabled_returns_input_unchanged() {
    let s = "ＡＢＣ　foo";
    let result = normalize_for_fts(s, &QueryNormalizationConfig::disabled());
    assert_eq!(result, s);
}

#[test]
fn nfkc_only_skips_lowercase_and_whitespace() {
    let config = QueryNormalizationConfig {
        nfkc: true,
        ascii_lowercase: false,
        collapse_whitespace: false,
    };
    assert_eq!(normalize_for_fts("ＡＢＣ", &config), "ABC");
    assert_eq!(normalize_for_fts("foo  bar", &config), "foo  bar");
    // Maybe inputs can compose, reorder, or already be normalized. Byte
    // prefixes include multibyte characters; expectations are literal.
    for (input, expected, borrowed) in [
        ("日本語 e\u{301}", "日本語 é", false),
        ("e\u{301}x", "éx", false),
        ("x\u{301}\u{327}", "x\u{327}\u{301}", false),
        ("x\u{301}", "x\u{301}", true),
        ("é\u{301}", "é\u{301}", true),
    ] {
        let actual = normalize_for_fts_cow(input, &config);
        assert_eq!(actual, expected);
        assert_eq!(matches!(actual, Cow::Borrowed(_)), borrowed);
        assert_eq!(normalize_for_fts(input, &config), expected);
    }
}

#[test]
fn lowercase_only_skips_nfkc() {
    let config = QueryNormalizationConfig {
        nfkc: false,
        ascii_lowercase: true,
        collapse_whitespace: false,
    };
    assert_eq!(normalize_for_fts("React", &config), "react");
    assert_eq!(normalize_for_fts("ＡＢＣ", &config), "ＡＢＣ");
}

#[test]
fn whitespace_only_skips_nfkc_and_lowercase() {
    let config = QueryNormalizationConfig {
        nfkc: false,
        ascii_lowercase: false,
        collapse_whitespace: true,
    };
    // Fixed expectations also pin borrowing and equal words at different
    // positions: only the exact canonical separator may keep the input.
    for (input, expected, borrowed) in [
        ("", "", true),
        ("日本語", "日本語", true),
        ("React App", "React App", true),
        ("a a", "a a", true),
        ("a  a", "a a", false),
        ("  React   App  ", "React App", false),
        ("日本語\u{2003}日本語", "日本語 日本語", false),
        ("日本語 ", "日本語", false),
        (" \t\n　", "", false),
    ] {
        let actual = normalize_for_fts_cow(input, &config);
        assert_eq!(actual, expected);
        assert_eq!(matches!(actual, Cow::Borrowed(_)), borrowed);
        assert_eq!(normalize_for_fts(input, &config), expected);
    }
}

#[test]
fn config_round_trips_through_serde() {
    let original = QueryNormalizationConfig::default();
    let json = serde_json::to_string(&original).expect("serialise");
    let parsed: QueryNormalizationConfig = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed, original);
}

#[test]
fn fullwidth_punctuation_folds_under_nfkc() {
    assert_eq!(run("hello！"), "hello!");
}

#[test]
fn mixed_symbols_and_unicode_preserve_each_configuration() {
    // Literal expectations pin composition, ASCII-only case, Unicode spaces,
    // and punctuation independently of the implementation's transformations.
    let expected = [
        "  Ａ e\u{301}\u{2003}B%_\\  ",
        "Ａ e\u{301} B%_\\",
        "  Ａ e\u{301}\u{2003}b%_\\  ",
        "Ａ e\u{301} b%_\\",
        "  A é B%_\\  ",
        "A é B%_\\",
        "  a é b%_\\  ",
        "a é b%_\\",
    ];
    for (bits, expected) in expected.into_iter().enumerate() {
        let config = QueryNormalizationConfig {
            nfkc: bits & 4 != 0,
            ascii_lowercase: bits & 2 != 0,
            collapse_whitespace: bits & 1 != 0,
        };
        let actual = normalize_for_fts("  Ａ e\u{301}\u{2003}B%_\\  ", &config);
        assert_eq!(actual, expected, "{config:?}");
        assert_eq!(normalize_for_fts(&actual, &config), actual);
    }
}
