/// Split text into fragments that each fit within `max_bytes`.
///
/// Split boundary priority: paragraph (`\n\n`) > line (`\n`) > character.
/// All splits are UTF-8 safe. Returns the input as a single fragment when
/// `max_bytes < 4` (cannot guarantee a valid split below the minimum
/// UTF-8 code point width).
pub fn split_text(text: &str, max_bytes: usize) -> Vec<&str> {
    if max_bytes < 4 || text.len() <= max_bytes {
        return vec![text];
    }
    let mut fragments = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        if remaining.len() <= max_bytes {
            fragments.push(remaining);
            break;
        }
        let boundary = remaining.floor_char_boundary(max_bytes);
        let split_at = remaining[..boundary]
            .rfind("\n\n")
            .map(|p| p + 2)
            .or_else(|| remaining[..boundary].rfind('\n').map(|p| p + 1))
            .unwrap_or(boundary);
        fragments.push(&remaining[..split_at]);
        remaining = &remaining[split_at..];
    }
    fragments
}

#[cfg(test)]
mod tests {
    use crate::text::split_text;

    #[test]
    fn t011_splits_large_text_into_bounded_fragments() {
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

    #[test]
    fn t012_splits_at_paragraph_boundary() {
        let text = "paragraph one\n\nparagraph two\n\nparagraph three";
        let max = "paragraph one\n\nparagraph two".len();
        let fragments = split_text(text, max);
        assert!(fragments.len() >= 2);
        assert!(!fragments[0].ends_with("\n\np"));
    }

    #[test]
    fn t013_falls_back_to_line_boundary() {
        let text = "line one\nline two\nline three\nline four";
        let max = "line one\nline two".len();
        let fragments = split_text(text, max);
        assert!(fragments.len() >= 2);
        assert!(!fragments[0].contains("line three"));
    }

    #[test]
    fn t015_no_split_when_under_limit() {
        let text = "short text";
        let fragments = split_text(text, 1000);
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0], text);
    }

    #[test]
    fn t016_max_bytes_below_4_returns_input_as_is() {
        let text = "hello world";
        let fragments = split_text(text, 3);
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0], text);
    }

    #[test]
    fn t011b_char_boundary_fallback_no_newlines() {
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

    #[test]
    fn t015b_exact_boundary_no_split() {
        let text = "abcdefghij"; // 10 bytes
        let fragments = split_text(text, 10);
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0], text);
    }
}
