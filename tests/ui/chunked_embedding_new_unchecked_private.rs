//! Downstream crates must not call the unchecked constructor because that
//! bypasses the non-empty chunks invariant.

fn main() {
    let _ = rurico::embed::ChunkedEmbedding::new_unchecked(Vec::new());
}
