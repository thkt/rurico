mod query_normalize;
mod search;

use std::sync::OnceLock;

pub use query_normalize::{QueryNormalizationConfig, normalize_for_fts, pre_phase_5_disabled};
pub use search::{
    MatchFtsQuery, SanitizeError, fts_quote, prepare_match_query, recency_decay, rrf_merge,
};

#[cfg(not(target_endian = "little"))]
compile_error!("rurico requires a little-endian target for f32↔u8 embedding storage");

/// Reinterpret `&[f32]` as `&[u8]` (zero-copy, little-endian).
pub fn f32_as_bytes(slice: &[f32]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

/// Register sqlite-vec as a process-global auto-extension.
///
/// Idempotent — subsequent calls are no-ops. All SQLite connections opened
/// after this call will have the vec extension available.
///
/// # Errors
///
/// Returns an opaque error string if sqlite-vec registration fails. The string
/// includes the SQLite return code, but its exact text is not part of the
/// stable API contract.
pub fn ensure_sqlite_vec() -> Result<(), String> {
    static INIT: OnceLock<Result<(), i32>> = OnceLock::new();
    let init_result = INIT.get_or_init(|| {
        let rc = rurico_ffi::sqlite_vec_register();
        if rc == 0 { Ok(()) } else { Err(rc) }
    });
    if let Err(rc) = init_result {
        return Err(format!(
            "sqlite-vec extension failed to register (sqlite3 rc={rc}). \
             This is a process-level initialization error that cannot be retried."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_sqlite_vec_idempotent() {
        ensure_sqlite_vec().unwrap();
        ensure_sqlite_vec().unwrap();
    }
}
