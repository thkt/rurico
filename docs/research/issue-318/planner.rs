//! Research-only prepared batches; not a production runtime or public API.
use std::mem::size_of;
use std::time::Instant;

#[derive(Debug)]
pub struct Row {
    pub doc: usize,
    pub chunk: usize,
    pub ids: Box<[u32]>,
    pub mask: Box<[u32]>,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub rows: usize,
    pub padded_tokens: usize,
    /// Prepared payload only, including reserved Row slots. Not RSS/GPU memory.
    pub bytes: usize,
}

#[derive(Debug)]
pub struct PreparedBatch {
    pub rows: Vec<Row>,
    pub bucket: usize,
    limits: Limits,
}

#[derive(Debug)]
pub enum Rejected {
    Invalid,
    Oversize,
    Full(Row),
}

impl PreparedBatch {
    pub fn new(limits: Limits) -> Self {
        assert!(limits.rows > 0 && limits.rows <= limits.bytes / size_of::<Row>());
        Self {
            rows: Vec::with_capacity(limits.rows),
            bucket: 128,
            limits,
        }
    }

    pub fn bytes(&self) -> usize {
        self.rows.capacity() * size_of::<Row>()
            + self
                .rows
                .iter()
                .map(|r| (r.ids.len() + r.mask.len()) * 4)
                .sum::<usize>()
    }

    pub fn push(&mut self, row: Row) -> Result<(), Rejected> {
        let n = row.ids.len();
        if n == 0 || n > 8192 || row.mask.len() != n {
            return Err(Rejected::Invalid);
        }
        let bucket = [128, 512, 2048, 8192]
            .into_iter()
            .find(|&b| b >= n)
            .unwrap();
        // Reject even when the batch is nonempty; never silently forward one
        // oversize record as the existing best-effort hint would.
        let extra = n * 8;
        let slots = self.rows.capacity() * size_of::<Row>();
        if bucket > self.limits.padded_tokens || extra > self.limits.bytes - slots {
            return Err(Rejected::Oversize);
        }
        let next_bucket = self.bucket.max(bucket);
        if self.rows.len() == self.limits.rows
            || (self.rows.len() + 1) * next_bucket > self.limits.padded_tokens
            || extra > self.limits.bytes - self.bytes()
        {
            return Err(Rejected::Full(row));
        }
        self.bucket = next_bucket;
        self.rows.push(row);
        Ok(())
    }
}

/// Stop at a forward boundary. In-flight GPU work is not interruptible here.
pub fn stopped(cancel: bool, deadline: Option<Instant>) -> bool {
    cancel || deadline.is_some_and(|d| Instant::now() >= d)
}

/// Experimental fairness: at most two query turns while bulk is ready.
pub fn query_turn(query_ready: bool, bulk_ready: bool, query_burst: usize) -> bool {
    query_ready && (!bulk_ready || query_burst < 2)
}

/// Vec adapter accepts only a complete ordered stream. Caller owns O(N) output.
pub fn collect_complete(
    batches: impl IntoIterator<Item = Result<Vec<(usize, usize, f32)>, &'static str>>,
    expected: usize,
) -> Result<Vec<f32>, &'static str> {
    let mut values = Vec::new();
    for batch in batches {
        for (doc, chunk, value) in batch? {
            if doc != values.len() || chunk != 0 || !value.is_finite() {
                return Err("invalid stream order/content");
            }
            values.push(value);
        }
    }
    if values.len() != expected {
        return Err("incomplete stream");
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(doc: usize, n: usize) -> Row {
        Row {
            doc,
            chunk: 0,
            ids: vec![u32::try_from(doc).unwrap(); n].into_boxed_slice(),
            mask: vec![1; n].into_boxed_slice(),
        }
    }

    #[test]
    fn refuses_oversize_and_flushes_before_bucket_growth_or_byte_overflow() {
        let limits = Limits {
            rows: 2,
            padded_tokens: 256,
            bytes: 2 * size_of::<Row>() + 32,
        };
        let mut batch = PreparedBatch::new(limits);
        batch.push(row(0, 2)).unwrap();
        batch.push(row(1, 2)).unwrap();
        assert_eq!(batch.bytes(), limits.bytes);
        assert!(matches!(batch.push(row(2, 2)), Err(Rejected::Full(_))));
        assert!(matches!(batch.push(row(2, 129)), Err(Rejected::Oversize)));
        let mut byte_oversize = PreparedBatch::new(limits);
        assert!(matches!(
            byte_oversize.push(row(2, 5)),
            Err(Rejected::Oversize)
        ));
        assert!(byte_oversize.rows.is_empty());
        byte_oversize.push(row(0, 2)).unwrap();
        let retained_bytes = byte_oversize.bytes();
        assert!(matches!(
            byte_oversize.push(row(2, 5)),
            Err(Rejected::Oversize)
        ));
        assert_eq!(byte_oversize.bytes(), retained_bytes);
        assert_eq!(byte_oversize.bucket, 128);
        assert_eq!(byte_oversize.rows.len(), 1);
        assert_eq!(byte_oversize.rows[0].doc, 0);
        assert_eq!(&*byte_oversize.rows[0].ids, &[0, 0]);
        assert_eq!(&*byte_oversize.rows[0].mask, &[1, 1]);
        let mut batch = PreparedBatch::new(Limits {
            rows: 4,
            padded_tokens: 512,
            bytes: 8192,
        });
        batch.push(row(0, 2)).unwrap();
        assert!(matches!(batch.push(row(1, 129)), Err(Rejected::Full(_))));
        assert_eq!(batch.rows.len(), 1);
        assert_eq!(batch.bucket, 128);
        let mut byte_limited = PreparedBatch::new(Limits {
            rows: 4,
            padded_tokens: 512,
            bytes: 4 * size_of::<Row>() + 32,
        });
        byte_limited.push(row(0, 3)).unwrap();
        assert!(matches!(
            byte_limited.push(row(1, 2)),
            Err(Rejected::Full(_))
        ));
        assert_eq!(byte_limited.rows.len(), 1);
        let mut malformed = row(1, 2);
        malformed.mask = vec![1].into_boxed_slice();
        assert!(matches!(batch.push(malformed), Err(Rejected::Invalid)));
    }

    #[test]
    fn increasing_input_keeps_prepared_payload_bounded_and_ordered() {
        let limits = Limits {
            rows: 10,
            padded_tokens: 1280,
            bytes: 4096,
        };
        for count in [50, 500, 5000] {
            let mut batch = PreparedBatch::new(limits);
            let mut output = Vec::new();
            let mut peak = 0;
            for doc in 0..count {
                if let Err(Rejected::Full(row)) = batch.push(row(doc, 5)) {
                    output.extend(batch.rows.iter().map(|r| (r.doc, r.chunk, r.ids[0] as f32)));
                    batch = PreparedBatch::new(limits);
                    batch.push(row).unwrap();
                }
                peak = peak.max(batch.bytes());
            }
            output.extend(batch.rows.iter().map(|r| (r.doc, r.chunk, r.ids[0] as f32)));
            assert_eq!(peak, 10 * size_of::<Row>() + 400);
            assert_eq!(
                collect_complete([Ok(output)], count).unwrap(),
                (0..count).map(|n| n as f32).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn partial_failure_and_expired_deadline_never_become_complete_vec() {
        assert_eq!(
            collect_complete([Ok(vec![(0, 0, 0.5)]), Err("forward failed")], 2),
            Err("forward failed")
        );
        assert!(collect_complete([Ok(vec![(1, 0, 0.5)])], 1).is_err());
        assert!(collect_complete([Ok(vec![(0, 0, f32::NAN)])], 1).is_err());
        assert!(collect_complete([Ok(vec![(0, 0, 0.5)])], 2).is_err());
        assert!(stopped(true, None));
        assert!(stopped(false, Some(Instant::now())));
        assert!(!stopped(false, None));
        let mut burst = 0;
        let turns: Vec<_> = (0..9)
            .map(|_| {
                let query = query_turn(true, true, burst);
                burst = if query { burst + 1 } else { 0 };
                query
            })
            .collect();
        assert_eq!(
            turns,
            [true, true, false, true, true, false, true, true, false]
        );
        assert!(query_turn(true, false, 20));
        assert!(!query_turn(false, true, 0));
    }
}
