//! MLX smoke-test binary — subprocess-isolated model verification.
//!
//! Invoked by integration tests via `Command` so MLX FFI crashes
//! (SIGABRT) are contained without killing the test runner.
//!
//! Loads the default embed model from the local HF Hub cache. The model
//! must be downloaded before running smoke tests.

use rurico::embed::{self, Embed};

fn main() {
    // Also acts as a probe subprocess when probe env vars are set.
    rurico::model_probe::handle_probe_if_needed();

    let artifacts = embed::cached_artifacts(embed::ModelId::default())
        .expect("cache lookup failed")
        .expect("model not cached; run download first");

    let embedder = embed::Embedder::new(&artifacts).expect("model load");
    let dims = embedder.embedding_dims();

    // Query embedding
    let q = embedder.embed_query("authentication logic").expect("query");
    assert_eq!(q.len(), dims, "query dims");

    // Consistency
    let q2 = embedder
        .embed_query("authentication logic")
        .expect("query2");
    assert_eq!(q, q2, "deterministic");

    // Short document (single chunk)
    let d = embedder
        .embed_document("function useAuth() { return user; }")
        .expect("short doc");
    assert_eq!(d.chunks.len(), 1, "short doc: 1 chunk");
    assert_eq!(d.chunks[0].len(), dims, "short doc dims");

    // Batch
    let batch = embedder
        .embed_documents_batch(&[
            "function useAuth() { return user; }",
            "function Button() { return <div/>; }",
        ])
        .expect("batch");
    assert_eq!(batch.len(), 2, "batch count");

    // Long document (multi-chunk, prefix-merge-triggering "apple")
    let sentence = "apple pie is a traditional dessert enjoyed around the world. ";
    let long_text = sentence.repeat(800);
    let ld = embedder.embed_document(&long_text).expect("long doc");
    assert!(ld.chunks.len() >= 2, "long doc: ≥2 chunks");
    for (i, chunk) in ld.chunks.iter().enumerate() {
        assert_eq!(chunk.len(), dims, "long doc chunk {i} dims");
    }

    // Prefix-merge short texts
    for text in ["apple pie", "the cat", "Rust", "This is a test"] {
        let r = embedder.embed_document(text).expect(text);
        assert_eq!(r.chunks.len(), 1, "'{text}': 1 chunk");
        assert_eq!(r.chunks[0].len(), dims, "'{text}' dims");
    }

    eprintln!("smoke: all checks passed");
}
