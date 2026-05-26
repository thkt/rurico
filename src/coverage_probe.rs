// Throwaway probe: verifies the diff coverage gate fails on an uncovered
// line. DELETE once verified. Not part of the public API.

/// Returns `n + 1`, saturating at [`usize::MAX`]. Intentionally untested so
/// the diff coverage gate reports its body as a missing line.
pub fn probe(n: usize) -> usize {
    n.saturating_add(1)
}
