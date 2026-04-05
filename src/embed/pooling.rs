use super::EmbedError;

/// Compute mean pooling over hidden states, weighted by attention mask.
///
/// `data` is a flat row-major buffer of shape `[seq_len × hidden_size]`.
/// Mask values act as weights: 0 excludes the token, 1 includes it equally,
/// and values > 1 increase the token's contribution proportionally.
/// Only the first `seq_len` mask entries are read (mask may be longer).
///
/// # Preconditions
///
/// - `attention_mask` must contain at least `seq_len` entries (only the first
///   `seq_len` entries are read; the caller is responsible for this invariant).
/// - At least one mask entry in `[0..seq_len]` must be non-zero. If all
///   entries are zero, pooling returns a zero vector (division is skipped),
///   which after L2 normalisation remains zero. Callers must ensure the
///   tokenizer always emits at least one real token (BOS/EOS count).
pub(crate) fn mean_pooling(
    data: &[f32],
    seq_len: usize,
    hidden_size: usize,
    attention_mask: &[u32],
) -> Vec<f32> {
    let mut result = vec![0.0f32; hidden_size];
    let mut mask_sum = 0.0f32;

    for (t, &m) in attention_mask.iter().enumerate().take(seq_len) {
        if m > 0 {
            let mf = m as f32;
            let offset = t * hidden_size;
            for d in 0..hidden_size {
                result[d] += data[offset + d] * mf;
            }
            mask_sum += mf;
        }
    }

    if mask_sum > 0.0 {
        for v in &mut result {
            *v /= mask_sum;
        }
    }

    result
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub(crate) fn postprocess_embedding(
    flat: &[f32],
    seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<f32>, EmbedError> {
    if seq_len == 0 {
        return Err(EmbedError::EmptySequence);
    }
    if !flat.len().is_multiple_of(seq_len) {
        return Err(EmbedError::inference(format!(
            "flat buffer length {} not divisible by seq_len {}",
            flat.len(),
            seq_len
        )));
    }
    let hidden_size = flat.len() / seq_len;
    if attention_mask.len() < seq_len {
        return Err(EmbedError::inference(format!(
            "attention_mask length {} < seq_len {}",
            attention_mask.len(),
            seq_len
        )));
    }
    let mut pooled = mean_pooling(flat, seq_len, hidden_size, attention_mask);
    l2_normalize(&mut pooled);
    if pooled.iter().any(|v| !v.is_finite()) {
        return Err(EmbedError::NonFiniteOutput);
    }
    Ok(pooled)
}
