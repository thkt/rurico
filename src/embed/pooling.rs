use super::{EMBEDDING_DIMS, EmbedError};

pub fn mean_pooling(
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

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn postprocess_embedding(
    flat: &[f32],
    seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<f32>, EmbedError> {
    if seq_len == 0 {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: 0,
        });
    }
    if flat.len() % seq_len != 0 {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: flat.len(),
        });
    }
    let hidden_size = flat.len() / seq_len;
    if hidden_size != EMBEDDING_DIMS as usize {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: hidden_size,
        });
    }
    let mut pooled = mean_pooling(flat, seq_len, hidden_size, attention_mask);
    l2_normalize(&mut pooled);
    Ok(pooled)
}
