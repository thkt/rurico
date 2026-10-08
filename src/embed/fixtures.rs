//! Save / load / compare `Vec<ChunkedEmbedding>` in a compact self-describing
//! binary format.
//!
//! Consumers capture the output of `embed_documents_batch` on one branch and
//! replay it on another to check that a refactor preserves embeddings within a
//! tolerance (Spec NFR-001: `cosine_similarity ≥ 0.99999` AND
//! `max_abs_diff ≤ 1e-5`).
//!
//! Legacy files contain u32 LE document/chunk/dimension counts and f32 LE values.
//! Version 1 adds magic, version and JSON generation conditions before that payload.
//! See `tests/fixtures/phase2_baseline/README.md` for bounds and migration.
//! [`load_fixture`] preserves provenance; [`load`] returns vectors only.
//! Legacy provenance is unknown, never inferred from the current model.

use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::io;
use std::io::{Cursor, Read, Write};

use super::ChunkedEmbedding;

/// Minimum cosine similarity for two fixtures to count as numerically
/// equivalent (Spec NFR-001).
pub const DEFAULT_COSINE_MIN: f32 = 0.99999;

/// Maximum per-element absolute difference for numerical equivalence
/// (Spec NFR-001).
pub const DEFAULT_MAX_ABS_DIFF: f32 = 1e-5;

/// Summary of the worst-case divergence between two fixtures of identical shape.
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

/// Default serialized fixture limit: 64 MiB (not an inference memory limit).
pub const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const MAGIC: u32 = u32::MAX;

/// Implementation that generated the vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Producer {
    /// rurico inference output.
    Rurico,
    /// Output from an official reference implementation.
    OfficialReference,
    /// Public synthetic vectors, generated without model inference.
    Synthetic,
}

/// Caller-recorded generation conditions, not an authenticity guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConditions {
    /// Vector producer.
    pub producer: Producer,
    /// Model repository identifier.
    pub model: String,
    /// Model revision used for generation.
    pub model_revision: String,
    /// Tokenizer identifier and revision (or content hash).
    pub tokenizer: String,
    /// Exact public inputs in document order. Do not persist private text.
    pub inputs: Vec<String>,
    /// Generation code revision and, if dirty, diff/content identifiers.
    pub generation_code: String,
    /// Prefix, chunking, pooling and other generation settings.
    pub settings: String,
}

impl GenerationConditions {
    fn validate(&self, docs: usize) -> io::Result<()> {
        if [
            &self.model,
            &self.model_revision,
            &self.tokenizer,
            &self.generation_code,
            &self.settings,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
            || self.inputs.len() != docs
        {
            return Err(invalid(
                "missing generation conditions or input count mismatch",
            ));
        }
        Ok(())
    }
}

/// Loaded vectors and optional generation conditions.
#[derive(Debug)]
pub struct Fixture {
    /// Validated embedding documents.
    pub docs: Vec<ChunkedEmbedding>,
    /// `None` means legacy provenance is unknown.
    pub generation: Option<GenerationConditions>,
}

/// Write version 1 with explicit generation conditions.
///
/// # Errors
/// Returns [`io::Error`] on invalid content/conditions, excessive size or write failure.
/// No bytes are written until validation and size checks succeed.
pub fn save_versioned<W: Write>(
    w: &mut W,
    docs: &[ChunkedEmbedding],
    generation: &GenerationConditions,
) -> io::Result<()> {
    save_versioned_with_limit(w, docs, generation, DEFAULT_MAX_BYTES)
}

fn save_versioned_with_limit<W: Write>(
    w: &mut W,
    docs: &[ChunkedEmbedding],
    generation: &GenerationConditions,
    max_bytes: usize,
) -> io::Result<()> {
    generation.validate(docs.len())?;
    validate_docs(docs).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let metadata = serde_json::to_vec(generation).map_err(invalid)?;
    // Count the full fixture before building the binary output buffer.
    let mut counter = ByteCounter {
        bytes: 12usize
            .checked_add(metadata.len())
            .ok_or_else(|| invalid("size overflow"))?,
        max_bytes,
    };
    save(&mut counter, docs)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(metadata.len())
            .map_err(invalid)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&metadata);
    save(&mut bytes, docs)?;
    w.write_all(&bytes)
}

struct ByteCounter {
    bytes: usize,
    max_bytes: usize,
}
impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("size overflow"))?;
        if self.bytes > self.max_bytes {
            return Err(invalid("fixture exceeds byte limit"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn invalid(message: impl Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// Write legacy vectors without provenance or content validation.
/// Prefer [`save_versioned`] for newly generated fixtures.
///
/// # Errors
/// Returns [`io::Error`] on lengths exceeding u32 or write failure.
///
/// Each chunk triggers two writes: a 4-byte `hidden_dim` header and the
/// f32 payload as a single contiguous little-endian byte slice (via
/// `bytemuck::cast_slice`). Callers writing to a [`std::fs::File`] must wrap
/// it in a [`std::io::BufWriter`] to avoid one syscall per chunk boundary.
pub fn save<W: Write>(w: &mut W, docs: &[ChunkedEmbedding]) -> io::Result<()> {
    let num_docs = u32::try_from(docs.len()).map_err(invalid)?;
    w.write_all(&num_docs.to_le_bytes())?;
    for doc in docs {
        let num_chunks = u32::try_from(doc.chunks().len()).map_err(invalid)?;
        w.write_all(&num_chunks.to_le_bytes())?;
        for chunk in doc.chunks() {
            let dim = u32::try_from(chunk.len()).map_err(invalid)?;
            w.write_all(&dim.to_le_bytes())?;
            w.write_all(bytemuck::cast_slice::<f32, u8>(chunk))?;
        }
    }
    Ok(())
}

/// Read validated vectors, discarding any recorded provenance.
///
/// # Errors
/// Same failures as [`load_fixture`] with [`DEFAULT_MAX_BYTES`].
pub fn load<R: Read>(r: &mut R) -> io::Result<Vec<ChunkedEmbedding>> {
    Ok(load_fixture(r, DEFAULT_MAX_BYTES)?.docs)
}

/// Read one complete legacy or version 1 fixture with a caller-selected byte limit.
/// Header-derived capacities are never allocated. Payload bounds are checked
/// before vector allocation; count headers must fit the remaining serialized data.
/// Dimensions are validated per document, not across documents or against a model.
///
/// # Errors
/// Returns [`io::ErrorKind::InvalidData`] for size, count, overflow, version,
/// metadata, content or trailing-data violations; truncation is `UnexpectedEof`.
/// Content errors retain [`super::EmbeddingValidationError`] as their cause.
/// Reader I/O failures are propagated. Zero documents are valid.
pub fn load_fixture<R: Read>(r: &mut R, max_bytes: usize) -> io::Result<Fixture> {
    let read_limit = u64::try_from(max_bytes)
        .map_err(invalid)?
        .checked_add(1)
        .ok_or_else(|| invalid("byte limit overflow"))?;
    let mut bytes = Vec::new();
    r.take(read_limit).read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(invalid("fixture exceeds byte limit"));
    }
    let mut cursor = Cursor::new(bytes.as_slice());
    let first = read_u32(&mut cursor)?;
    let generation = if first == MAGIC {
        if read_u32(&mut cursor)? != 1 {
            return Err(invalid("unsupported fixture version"));
        }
        let len = u32_to_usize(read_u32(&mut cursor)?, "metadata length")?;
        let metadata = take_bytes(&mut cursor, len)?;
        Some(serde_json::from_slice::<GenerationConditions>(metadata).map_err(invalid)?)
    } else {
        None
    };
    let num_docs = if generation.is_some() {
        read_u32(&mut cursor)?
    } else {
        first
    };
    let num_docs = u32_to_usize(num_docs, "num_docs")?;
    check_count(&cursor, num_docs, 4)?;
    let mut docs = Vec::new();
    for _ in 0..num_docs {
        let num_chunks = u32_to_usize(read_u32(&mut cursor)?, "num_chunks")?;
        check_count(&cursor, num_chunks, 4)?;
        let mut chunks = Vec::new();
        for _ in 0..num_chunks {
            let dim = u32_to_usize(read_u32(&mut cursor)?, "hidden_dim")?;
            let len = dim
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| invalid("hidden_dim byte size overflow"))?;
            let payload = take_bytes(&mut cursor, len)?;
            let vector: Vec<f32> = payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let expected = chunks.first().map_or(dim, |v: &Vec<f32>| v.len());
            super::validate_vector(&vector, chunks.len(), expected)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            chunks.push(vector);
        }
        docs.push(
            // Each vector was validated above and remains owned by this reader.
            ChunkedEmbedding::try_new(chunks)
                .map_err(super::EmbeddingValidationError::from)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        );
    }
    if cursor.position() != bytes.len() as u64 {
        return Err(invalid("trailing fixture data"));
    }
    if let Some(generation) = &generation {
        generation.validate(docs.len())?;
    }
    Ok(Fixture { docs, generation })
}

fn check_count(cursor: &Cursor<&[u8]>, count: usize, minimum: usize) -> io::Result<()> {
    let len = count
        .checked_mul(minimum)
        .ok_or_else(|| invalid("count byte size overflow"))?;
    if len > cursor.get_ref().len() - usize::try_from(cursor.position()).map_err(invalid)? {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "count exceeds remaining fixture bytes",
        ));
    }
    Ok(())
}

fn take_bytes<'a>(cursor: &mut Cursor<&'a [u8]>, len: usize) -> io::Result<&'a [u8]> {
    let start = usize::try_from(cursor.position()).map_err(invalid)?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid("payload size overflow"))?;
    let bytes = cursor
        .get_ref()
        .get(start..end)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated fixture payload"))?;
    cursor.set_position(end as u64);
    Ok(bytes)
}

fn u32_to_usize(value: u32, field: &'static str) -> io::Result<usize> {
    usize::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} ({value}) exceeds usize"),
        )
    })
}

/// Which input contains invalid content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureSide {
    /// Expected fixture.
    Expected,
    /// Actual fixture.
    Actual,
}

/// Comparison failure, distinct from a finite numerical divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CompareError {
    /// Fixture shapes differ.
    #[error("shape mismatch: {0:?}")]
    Shape(ShapeMismatch),
    /// Invalid legacy content; document and nested vector position are retained.
    #[error("{side:?} document {doc}: {source}")]
    InvalidContent {
        /// Input side.
        side: FixtureSide,
        /// Document index.
        doc: usize,
        /// Content validation failure.
        source: super::EmbeddingValidationError,
    },
    /// A metric cannot be represented as finite f32.
    #[error("non-finite comparison metric at document {doc}, chunk {chunk}")]
    NonFiniteMetric {
        /// Document index.
        doc: usize,
        /// Chunk index.
        chunk: usize,
    },
}

fn validate_docs(docs: &[ChunkedEmbedding]) -> Result<(), super::EmbeddingValidationError> {
    for doc in docs {
        validate_doc(doc)?;
    }
    Ok(())
}
fn validate_doc(doc: &ChunkedEmbedding) -> Result<(), super::EmbeddingValidationError> {
    let first = doc
        .chunks()
        .first()
        .ok_or(super::EmbeddingValidationError::EmptyChunks(
            super::EmptyChunksError,
        ))?;
    for (chunk, vector) in doc.chunks().iter().enumerate() {
        super::validate_vector(vector, chunk, first.len())?;
    }
    Ok(())
}

/// Compare two fixtures element-wise.
///
/// Returns `Ok(FixtureDiff)` when shapes match (even if values differ).
///
/// # Errors
/// Returns [`CompareError`] for invalid content on either side, shape mismatch,
/// or non-finite metrics. Content is checked before shape comparison.
pub fn compare(
    expected: &[ChunkedEmbedding],
    actual: &[ChunkedEmbedding],
) -> Result<FixtureDiff, CompareError> {
    for (side, docs) in [
        (FixtureSide::Expected, expected),
        (FixtureSide::Actual, actual),
    ] {
        for (doc, value) in docs.iter().enumerate() {
            validate_doc(value).map_err(|source| CompareError::InvalidContent {
                side,
                doc,
                source,
            })?;
        }
    }
    if expected.len() != actual.len() {
        return Err(CompareError::Shape(ShapeMismatch::DocCount {
            expected: expected.len(),
            actual: actual.len(),
        }));
    }
    let mut max_abs_diff = 0.0f32;
    let mut cosine_min = 1.0f32;
    for (d_idx, (exp_doc, act_doc)) in expected.iter().zip(actual).enumerate() {
        if exp_doc.chunks().len() != act_doc.chunks().len() {
            return Err(CompareError::Shape(ShapeMismatch::ChunkCount {
                doc: d_idx,
                expected: exp_doc.chunks().len(),
                actual: act_doc.chunks().len(),
            }));
        }
        for (c_idx, (exp_ch, act_ch)) in exp_doc.chunks().iter().zip(act_doc.chunks()).enumerate() {
            if exp_ch.len() != act_ch.len() {
                return Err(CompareError::Shape(ShapeMismatch::Dim {
                    doc: d_idx,
                    chunk: c_idx,
                    expected: exp_ch.len(),
                    actual: act_ch.len(),
                }));
            }
            for (&e, &a) in exp_ch.iter().zip(act_ch) {
                let diff = (e - a).abs();
                if !diff.is_finite() {
                    return Err(CompareError::NonFiniteMetric {
                        doc: d_idx,
                        chunk: c_idx,
                    });
                }
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }
            let similarity = cosine(exp_ch, act_ch);
            if !similarity.is_finite() {
                return Err(CompareError::NonFiniteMetric {
                    doc: d_idx,
                    chunk: c_idx,
                });
            }
            cosine_min = cosine_min.min(similarity);
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
// A bounded cosine is rounded back to the existing f32 metric API.
#[allow(clippy::cast_possible_truncation)]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        (dot / denom) as f32
    } else if a == b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests;
