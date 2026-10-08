use crate::text::split_text;

// T-011: splits_large_text_into_bounded_fragments
#[test]
fn splits_large_text_into_bounded_fragments() {
    let text = "a".repeat(20_000);
    let fragments = split_text(&text, 16_000);
    assert_eq!(fragments, [&text[..16_000], &text[16_000..]]);
    assert_eq!(fragments.concat(), text);
    for (i, frag) in fragments.iter().enumerate() {
        assert!(
            frag.len() <= 16_000,
            "fragment {i} is {} bytes, exceeds 16000",
            frag.len()
        );
    }
}

// Expected fragments are literal contract examples, not computed by a splitter.
#[test]
fn split_boundary_priority_preserves_text() {
    let cases: &[(&str, &str, usize, &[&str])] = &[
        // A paragraph wins even when a later line fits in the same window.
        (
            "paragraph_before_later_line",
            "aa\n\nbbb\ncccccccc",
            8,
            &["aa\n\n", "bbb\n", "cccccccc"],
        ),
        (
            "last_paragraph_in_window",
            "a\n\nb\n\ncdefgh",
            8,
            &["a\n\nb\n\n", "cdefgh"],
        ),
        (
            "last_line_in_window",
            "a\nbb\ncccccccc",
            7,
            &["a\nbb\n", "ccccccc", "c"],
        ),
        (
            "japanese_paragraph_before_later_line",
            "あ\n\nい\nうえ",
            9,
            &["あ\n\n", "い\n", "うえ"],
        ),
        // The separator's end lies just after, at, or before the byte limit.
        (
            "paragraph_end_after_limit",
            "abc\n\n01234567",
            4,
            &["abc\n", "\n", "0123", "4567"],
        ),
        (
            "paragraph_end_at_limit",
            "abc\n\n01234567",
            5,
            &["abc\n\n", "01234", "567"],
        ),
        (
            "paragraph_end_before_limit",
            "abc\n\n01234567",
            6,
            &["abc\n\n", "012345", "67"],
        ),
        (
            "line_end_after_limit",
            "abcd\n01234567",
            4,
            &["abcd", "\n", "0123", "4567"],
        ),
        (
            "line_end_at_limit",
            "abcd\n01234567",
            5,
            &["abcd\n", "01234", "567"],
        ),
        (
            "line_end_before_limit",
            "abcd\n01234567",
            6,
            &["abcd\n", "012345", "67"],
        ),
        // CR is preserved; only LF and LF LF are the documented separators.
        (
            "crlf_end_after_limit",
            "abc\r\n01234567",
            4,
            &["abc\r", "\n", "0123", "4567"],
        ),
        (
            "crlf_end_at_limit",
            "abc\r\n01234567",
            5,
            &["abc\r\n", "01234", "567"],
        ),
        (
            "crlf_end_before_limit",
            "abc\r\n01234567",
            6,
            &["abc\r\n", "012345", "67"],
        ),
        (
            "crlf_blank_line_is_not_lf_paragraph",
            "a\r\n\r\nb\ncdefgh",
            7,
            &["a\r\n\r\nb\n", "cdefgh"],
        ),
        (
            "bare_cr_is_not_line_boundary",
            "ab\rcdefgh",
            4,
            &["ab\rc", "defg", "h"],
        ),
        (
            "character_end_after_limit",
            "あいうえおか",
            8,
            &["あい", "うえ", "おか"],
        ),
        (
            "character_end_at_limit",
            "あいうえおか",
            9,
            &["あいう", "えおか"],
        ),
        (
            "character_end_before_limit",
            "あいうえおか",
            10,
            &["あいう", "えおか"],
        ),
        (
            "four_byte_character_at_minimum_limit",
            "a😀いb",
            4,
            &["a", "😀", "いb"],
        ),
        ("two_byte_character", "ééé", 5, &["éé", "é"]),
        ("input_below_limit", "short text", 1000, &["short text"]),
        ("input_at_limit", "abcdefghij", 10, &["abcdefghij"]),
        ("input_above_limit", "abcdefghijk", 10, &["abcdefghij", "k"]),
        ("trailing_paragraph", "abcd\n\n", 4, &["abcd", "\n\n"]),
    ];

    for &(name, text, max_bytes, expected) in cases {
        let fragments = split_text(text, max_bytes);
        assert_eq!(fragments, expected, "{name}: split boundary priority");
        assert_eq!(fragments.concat(), text, "{name}: preserve original bytes");
        assert!(
            fragments.iter().all(|fragment| fragment.len() <= max_bytes),
            "{name}: fragments must fit the byte limit"
        );
    }
}

// Below four bytes, even oversized input is returned whole (including at zero).
#[test]
fn max_bytes_below_4_returns_input_as_is() {
    for max_bytes in 0..4 {
        for text in ["hello world", "あ😀\r\n\nい", ""] {
            let fragments = split_text(text, max_bytes);
            assert_eq!(fragments, [text], "max_bytes={max_bytes}");
            assert_eq!(fragments.concat(), text);
        }
    }
}

// T-105-007: split_text_returns_single_empty_fragment_for_empty_input
#[test]
fn split_text_returns_single_empty_fragment_for_empty_input() {
    let fragments = split_text("", 100);
    assert_eq!(
        fragments,
        vec![""],
        "empty input must yield a single empty fragment, not Vec::new()"
    );
}
