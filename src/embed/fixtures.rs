//! Fixture serialisation for Phase 2 numerical-equivalence checks.
//!
//! Saves and loads `Vec<ChunkedEmbedding>` in a compact little-endian binary
//! format so the main-branch output can be captured as a fixture and compared
//! bit-exactly (or within tolerance) against bucket-batched output in later PRs.
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
//! Self-describing (per-chunk dim), so fixtures survive model-dim changes at read
//! time and the loader can validate against the current model.

use std::io;
use std::io::{Read, Write};

use super::ChunkedEmbedding;

/// Minimum cosine similarity between two fixtures to count as numerically equivalent.
/// Phase 2 Spec NFR-001.
pub const DEFAULT_COSINE_MIN: f32 = 0.99999;

/// Maximum per-element absolute difference for numerical equivalence.
/// Phase 2 Spec NFR-001.
pub const DEFAULT_MAX_ABS_DIFF: f32 = 1e-5;

/// Summary of the worst-case divergence between two fixtures of identical shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureDiff {
    /// Largest absolute per-element difference observed.
    pub max_abs_diff: f32,
    /// Smallest cosine similarity observed across all chunk pairs.
    pub cosine_min: f32,
}

/// Shape mismatch between two fixtures. Contains the first offending index.
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
pub fn save<W: Write>(w: &mut W, docs: &[ChunkedEmbedding]) -> io::Result<()> {
    let num_docs = u32::try_from(docs.len()).expect("num_docs fits in u32");
    w.write_all(&num_docs.to_le_bytes())?;
    for doc in docs {
        let num_chunks = u32::try_from(doc.chunks.len()).expect("num_chunks fits in u32");
        w.write_all(&num_chunks.to_le_bytes())?;
        for chunk in &doc.chunks {
            let dim = u32::try_from(chunk.len()).expect("hidden_dim fits in u32");
            w.write_all(&dim.to_le_bytes())?;
            for &v in chunk {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Read a fixture from `r` that was written by [`save`].
pub fn load<R: Read>(r: &mut R) -> io::Result<Vec<ChunkedEmbedding>> {
    let num_docs = read_u32(r)? as usize;
    let mut docs = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let num_chunks = read_u32(r)? as usize;
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            let dim = read_u32(r)? as usize;
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(read_f32(r)?);
            }
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
        max_abs_diff,
        cosine_min,
    })
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

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
    if denom > 0.0 { dot / denom } else { 0.0 }
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

    // save + load round-trip preserves values
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

    // compare: identical fixtures → diff=0, cosine≈1 (f32 rounding makes an exact
    // equality unattainable even when the inputs are bitwise identical)
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

    // compare: element difference surfaces in max_abs_diff
    #[test]
    fn compare_divergent_fixtures_reports_max_abs_diff() {
        let a = sample_docs();
        let mut b = sample_docs();
        b[0].chunks[0][0] += 0.01; // perturb one element
        let diff = compare(&a, &b).unwrap();
        assert!((diff.max_abs_diff - 0.01).abs() < 1e-6);
        assert!(diff.cosine_min < 1.0);
    }

    // compare: shape mismatch at doc-count returns Err
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

    // compare: shape mismatch at chunk-count returns Err with doc index
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

    // compare: shape mismatch at dim returns Err with (doc, chunk) index
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

    // Defaults match Spec NFR-001
    #[test]
    fn default_tolerances_match_spec_nfr_001() {
        assert!((DEFAULT_COSINE_MIN - 0.99999).abs() < f32::EPSILON);
        assert!((DEFAULT_MAX_ABS_DIFF - 1e-5).abs() < f32::EPSILON);
    }
}
