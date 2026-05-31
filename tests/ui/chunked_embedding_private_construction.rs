//! Downstream crates must not construct `ChunkedEmbedding` with a struct
//! literal because that bypasses the non-empty chunks invariant.

fn main() {
    let _ = rurico::embed::ChunkedEmbedding {
        chunks: Vec::new(),
        chunk_ids: Vec::new(),
    };
}
