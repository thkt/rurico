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
mod tests;
