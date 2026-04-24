//! Save / load / compare `Vec<ChunkedEmbedding>` in a compact self-describing
//! binary format.
//!
//! Consumers capture the output of `embed_documents_batch` on one branch and
//! replay it on another to check that a refactor preserves embeddings within a
//! tolerance (Spec NFR-001: `cosine_similarity ≥ 0.99999` AND
//! `max_abs_diff ≤ 1e-5`).
//!
//! # Format
//!
//! ```text
//! u32 LE  num_docs
//!   per doc:
//!     u32 LE  num_chunks
//!     per chunk:
//!       u32 LE  hidden_dim
//!       f32 LE × hidden_dim
//! ```
//!
//! Per-chunk `hidden_dim` makes the loader safe against model-dim drift; a
//! mismatched fixture surfaces as a load error rather than silent corruption.
//!
//! # Buffering
//!
//! [`save`] writes one 4-byte length + one `hidden_dim × 4`-byte block per
//! chunk. [`load`] mirrors it. When writing or reading from a [`std::fs::File`],
//! wrap in [`std::io::BufWriter`] / [`std::io::BufReader`] — without buffering
//! each short write/read becomes a separate syscall.

use std::io;
use std::io::{Read, Write};

use super::ChunkedEmbedding;
use crate::storage::f32_as_bytes;

/// Minimum cosine similarity for two fixtures to count as numerically
/// equivalent (Spec NFR-001).
pub const DEFAULT_COSINE_MIN: f32 = 0.99999;

/// Maximum per-element absolute difference for numerical equivalence
/// (Spec NFR-001).
pub const DEFAULT_MAX_ABS_DIFF: f32 = 1e-5;

/// Summary of the worst-case divergence between two fixtures of identical shape.
///
/// Field order mirrors the order the reader reasons about: cosine similarity
/// first (main metric), then the absolute-diff tiebreaker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureDiff {
    /// Smallest cosine similarity observed across all chunk pairs.
    pub cosine_min: f32,
    /// Largest absolute per-element difference observed.
    pub max_abs_diff: f32,
}

/// Shape mismatch between two fixtures. Carries the first offending index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMismatch {
    /// Top-level `Vec<ChunkedEmbedding>` lengths differ.
    DocCount {
        /// Document count in the expected (fixture) side.
        expected: usize,
        /// Document count in the actual (current run) side.
        actual: usize,
    },
    /// Per-document chunk counts differ.
    ChunkCount {
        /// Index of the first differing document.
        doc: usize,
        /// Chunk count in the expected side.
        expected: usize,
        /// Chunk count in the actual side.
        actual: usize,
    },
    /// Per-chunk hidden-dim lengths differ.
    Dim {
        /// Index of the differing document.
        doc: usize,
        /// Index of the differing chunk within the document.
        chunk: usize,
        /// Hidden-dim in the expected side.
        expected: usize,
        /// Hidden-dim in the actual side.
        actual: usize,
    },
}

/// Write `docs` to `w` in the format documented at the module level.
///
/// Each chunk triggers two writes: a 4-byte `hidden_dim` header and the
/// f32 payload as a single contiguous little-endian byte slice (via
/// [`f32_as_bytes`]). Callers writing to a [`std::fs::File`] must wrap it
/// in a [`std::io::BufWriter`] to avoid one syscall per chunk boundary.
pub fn save<W: Write>(w: &mut W, docs: &[ChunkedEmbedding]) -> io::Result<()> {
    let num_docs = u32::try_from(docs.len()).expect("num_docs fits in u32");
    w.write_all(&num_docs.to_le_bytes())?;
    for doc in docs {
        let num_chunks = u32::try_from(doc.chunks.len()).expect("num_chunks fits in u32");
        w.write_all(&num_chunks.to_le_bytes())?;
        for chunk in &doc.chunks {
            let dim = u32::try_from(chunk.len()).expect("hidden_dim fits in u32");
            w.write_all(&dim.to_le_bytes())?;
            w.write_all(f32_as_bytes(chunk))?;
        }
    }
    Ok(())
}

/// Read a fixture from `r` that was written by [`save`].
///
/// Uses buffered reads per chunk. Callers reading from a [`std::fs::File`]
/// should wrap it in a [`std::io::BufReader`] to avoid one syscall per chunk.
pub fn load<R: Read>(r: &mut R) -> io::Result<Vec<ChunkedEmbedding>> {
    let num_docs = read_u32(r)? as usize;
    let mut docs = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let num_chunks = read_u32(r)? as usize;
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            let dim = read_u32(r)? as usize;
            let mut bytes = vec![0u8; dim * 4];
            r.read_exact(&mut bytes)?;
            let vec: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
            chunks.push(vec);
        }
        docs.push(ChunkedEmbedding { chunks });
    }
    Ok(docs)
}

/// Compare two fixtures element-wise.
///
/// Returns `Ok(FixtureDiff)` when shapes match (even if values differ).
/// Returns `Err(ShapeMismatch)` on any structural difference.
pub fn compare(
    expected: &[ChunkedEmbedding],
    actual: &[ChunkedEmbedding],
) -> Result<FixtureDiff, ShapeMismatch> {
    if expected.len() != actual.len() {
        return Err(ShapeMismatch::DocCount {
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    let mut max_abs_diff = 0.0f32;
    let mut cosine_min = 1.0f32;
    for (d_idx, (exp_doc, act_doc)) in expected.iter().zip(actual).enumerate() {
        if exp_doc.chunks.len() != act_doc.chunks.len() {
            return Err(ShapeMismatch::ChunkCount {
                doc: d_idx,
                expected: exp_doc.chunks.len(),
                actual: act_doc.chunks.len(),
            });
        }
        for (c_idx, (exp_ch, act_ch)) in exp_doc.chunks.iter().zip(&act_doc.chunks).enumerate() {
            if exp_ch.len() != act_ch.len() {
                return Err(ShapeMismatch::Dim {
                    doc: d_idx,
                    chunk: c_idx,
                    expected: exp_ch.len(),
                    actual: act_ch.len(),
                });
            }
            for (&e, &a) in exp_ch.iter().zip(act_ch) {
                let diff = (e - a).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }
            cosine_min = cosine_min.min(cosine(exp_ch, act_ch));
        }
    }
    Ok(FixtureDiff {
        cosine_min,
        max_abs_diff,
    })
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Cosine similarity between two equal-length slices.
///
/// Returns `1.0` when both inputs are element-wise equal (including the
/// all-zero case), `0.0` when they disagree but at least one has zero norm.
/// This keeps identical zero fixtures from being reported as catastrophic
/// mismatches while still flagging zero-versus-nonzero cases.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        dot / denom
    } else if a == b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_docs() -> Vec<ChunkedEmbedding> {
        vec![
            ChunkedEmbedding {
                chunks: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            },
            ChunkedEmbedding {
                chunks: vec![vec![-0.7, 0.8, 0.9]],
            },
        ]
    }

    #[test]
    fn save_and_load_round_trips_identically() {
        let docs = sample_docs();
        let mut buf = Vec::new();
        save(&mut buf, &docs).unwrap();
        let loaded = load(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(docs.len(), loaded.len());
        for (a, b) in docs.iter().zip(&loaded) {
            assert_eq!(a.chunks, b.chunks);
        }
    }

    #[test]
    fn compare_identical_fixtures_reports_zero_diff() {
        let docs = sample_docs();
        let diff = compare(&docs, &docs).unwrap();
        assert_eq!(diff.max_abs_diff, 0.0);
        assert!(
            (diff.cosine_min - 1.0).abs() < 1e-6,
            "expected cosine=1.0, got {}",
            diff.cosine_min
        );
    }

    // Regression: identical all-zero fixtures must report cosine=1.0 (not 0.0)
    // so the NFR-001 threshold stays passable on zero-norm inputs.
    #[test]
    fn compare_identical_zero_fixtures_reports_cosine_one() {
        let zeros = vec![ChunkedEmbedding {
            chunks: vec![vec![0.0f32; 8]],
        }];
        let diff = compare(&zeros, &zeros).unwrap();
        assert_eq!(diff.cosine_min, 1.0);
        assert_eq!(diff.max_abs_diff, 0.0);
    }

    // Regression: zero vs nonzero must report cosine=0.0 (clear mismatch).
    #[test]
    fn compare_zero_versus_nonzero_reports_cosine_zero() {
        let a = vec![ChunkedEmbedding {
            chunks: vec![vec![0.0f32; 3]],
        }];
        let b = vec![ChunkedEmbedding {
            chunks: vec![vec![1.0, 2.0, 3.0]],
        }];
        let diff = compare(&a, &b).unwrap();
        assert_eq!(diff.cosine_min, 0.0);
    }

    #[test]
    fn compare_divergent_fixtures_reports_max_abs_diff() {
        let a = sample_docs();
        let mut b = sample_docs();
        b[0].chunks[0][0] += 0.01;
        let diff = compare(&a, &b).unwrap();
        assert!((diff.max_abs_diff - 0.01).abs() < 1e-6);
        assert!(diff.cosine_min < 1.0);
    }

    #[test]
    fn compare_doc_count_mismatch_returns_err() {
        let a = sample_docs();
        let b: Vec<_> = a.iter().skip(1).cloned().collect();
        assert_eq!(
            compare(&a, &b),
            Err(ShapeMismatch::DocCount {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn compare_chunk_count_mismatch_returns_err_with_doc_index() {
        let a = sample_docs();
        let mut b = sample_docs();
        b[0].chunks.pop();
        match compare(&a, &b) {
            Err(ShapeMismatch::ChunkCount {
                doc,
                expected,
                actual,
            }) => {
                assert_eq!(doc, 0);
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected ChunkCount mismatch, got {other:?}"),
        }
    }

    #[test]
    fn compare_dim_mismatch_returns_err_with_chunk_index() {
        let a = sample_docs();
        let mut b = sample_docs();
        b[1].chunks[0].push(0.0);
        match compare(&a, &b) {
            Err(ShapeMismatch::Dim {
                doc,
                chunk,
                expected,
                actual,
            }) => {
                assert_eq!(doc, 1);
                assert_eq!(chunk, 0);
                assert_eq!(expected, 3);
                assert_eq!(actual, 4);
            }
            other => panic!("expected Dim mismatch, got {other:?}"),
        }
    }

    #[test]
    fn default_tolerances_match_spec_nfr_001() {
        assert!((DEFAULT_COSINE_MIN - 0.99999).abs() < f32::EPSILON);
        assert!((DEFAULT_MAX_ABS_DIFF - 1e-5).abs() < f32::EPSILON);
    }
}
