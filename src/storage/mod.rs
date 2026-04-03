mod search;

pub use search::{
    MatchFtsQuery, SanitizeError, fts_quote, prepare_match_query, recency_decay, rrf_merge,
};

use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

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
    static INIT: std::sync::OnceLock<Result<(), i32>> = std::sync::OnceLock::new();
    let init_result = INIT.get_or_init(|| {
        // SAFETY: sqlite3_vec_init is the auto-extension entry point exported by sqlite-vec.
        // sqlite-vec exports it as `unsafe extern "C" fn()`, while rusqlite's
        // sqlite3_auto_extension expects the full init signature. Both are C fn pointers
        // with compatible calling conventions; SQLite calls it with the correct arguments.
        let rc = unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<
                unsafe extern "C" fn(),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(sqlite3_vec_init)))
        };
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
