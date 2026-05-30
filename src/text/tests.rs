use crate::text::split_text;

// T-011: splits_large_text_into_bounded_fragments
#[test]
fn splits_large_text_into_bounded_fragments() {
    let text = "a".repeat(20_000);
    let fragments = split_text(&text, 16_000);
    assert_eq!(fragments.len(), 2);
    for (i, frag) in fragments.iter().enumerate() {
        assert!(
            frag.len() <= 16_000,
            "fragment {i} is {} bytes, exceeds 16000",
            frag.len()
        );
    }
}

// T-012: splits_at_paragraph_boundary
#[test]
fn splits_at_paragraph_boundary() {
    let text = "paragraph one\n\nparagraph two\n\nparagraph three";
    let max = "paragraph one\n\nparagraph two".len();
    let fragments = split_text(text, max);
    assert!(fragments.len() >= 2);
    assert!(!fragments[0].ends_with("\n\np"));
}

// T-013: falls_back_to_line_boundary
#[test]
fn falls_back_to_line_boundary() {
    let text = "line one\nline two\nline three\nline four";
    let max = "line one\nline two".len();
    let fragments = split_text(text, max);
    assert!(fragments.len() >= 2);
    assert!(!fragments[0].contains("line three"));
}

// T-015: no_split_when_under_limit
#[test]
fn no_split_when_under_limit() {
    let text = "short text";
    let fragments = split_text(text, 1000);
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0], text);
}

// T-016: max_bytes_below_4_returns_input_as_is
#[test]
fn max_bytes_below_4_returns_input_as_is() {
    let text = "hello world";
    let fragments = split_text(text, 3);
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0], text);
}

// T-011b: char_boundary_fallback_no_newlines
#[test]
fn char_boundary_fallback_no_newlines() {
    let text = "あいうえおか"; // 6 chars, 18 bytes
    let fragments = split_text(text, 10);
    assert!(
        fragments.len() >= 2,
        "should split, got {} fragment(s)",
        fragments.len()
    );
    for (i, frag) in fragments.iter().enumerate() {
        assert!(
            frag.len() <= 10,
            "fragment {i} is {} bytes, exceeds 10",
            frag.len()
        );
    }
    assert_eq!(
        fragments[0].chars().count(),
        3,
        "first fragment should be 3 chars (9 bytes)"
    );
}

// T-015b: exact_boundary_no_split
#[test]
fn exact_boundary_no_split() {
    let text = "abcdefghij"; // 10 bytes
    let fragments = split_text(text, 10);
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0], text);
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
