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
/// `bytemuck::cast_slice`). Callers writing to a [`std::fs::File`] must wrap
/// it in a [`std::io::BufWriter`] to avoid one syscall per chunk boundary.
pub fn save<W: Write>(w: &mut W, docs: &[ChunkedEmbedding]) -> io::Result<()> {
    let num_docs = u32::try_from(docs.len()).expect("num_docs fits in u32");
    w.write_all(&num_docs.to_le_bytes())?;
    for doc in docs {
        let num_chunks = u32::try_from(doc.chunks.len()).expect("num_chunks fits in u32");
        w.write_all(&num_chunks.to_le_bytes())?;
        for chunk in &doc.chunks {
            let dim = u32::try_from(chunk.len()).expect("hidden_dim fits in u32");
            w.write_all(&dim.to_le_bytes())?;
            w.write_all(bytemuck::cast_slice::<f32, u8>(chunk))?;
        }
    }
    Ok(())
}

/// Read a fixture from `r` that was written by [`save`].
///
/// Uses buffered reads per chunk. Callers reading from a [`std::fs::File`]
/// should wrap it in a [`std::io::BufReader`] to avoid one syscall per chunk.
pub fn load<R: Read>(r: &mut R) -> io::Result<Vec<ChunkedEmbedding>> {
    let num_docs = u32_to_usize(read_u32(r)?, "num_docs")?;
    let mut docs = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let num_chunks = u32_to_usize(read_u32(r)?, "num_chunks")?;
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            let dim = u32_to_usize(read_u32(r)?, "hidden_dim")?;
            // `dim * size_of::<f32>()` (bytes per chunk) overflows `usize` on
            // 32-bit targets when `dim > usize::MAX / 4`; `checked_mul` surfaces a
            // clear error instead of panicking. Unreachable on 64-bit.
            const BYTES_PER_F32: usize = size_of::<f32>();
            let bytes_len = dim.checked_mul(BYTES_PER_F32).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("hidden_dim ({dim}) * {BYTES_PER_F32} overflows usize"),
                )
            })?;
            let mut bytes = vec![0u8; bytes_len];
            r.read_exact(&mut bytes)?;
            let vec: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
            chunks.push(vec);
        }
        docs.push(ChunkedEmbedding::new(chunks));
    }
    Ok(docs)
}

/// Convert a `u32` header field to `usize`, returning a structured `InvalidData`
/// error if the value cannot fit. Only fails on 16-bit `usize` targets; on
/// 32/64-bit, `From<u32> for usize` makes the conversion infallible — the
/// `Err` arm exists for type-system completeness, not as a runtime guard.
fn u32_to_usize(value: u32, field: &'static str) -> io::Result<usize> {
    usize::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} ({value}) exceeds usize"),
        )
    })
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
mod tests;
