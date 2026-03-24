mod search;

pub use search::{fts_expand_short_terms, fts_quote, rrf_merge};

use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

#[cfg(not(target_endian = "little"))]
compile_error!("rurico requires a little-endian target for f32↔u8 embedding storage");

pub fn f32_as_bytes(slice: &[f32]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

/// Register sqlite-vec as an auto-extension. Safe to call multiple times (no-op after first).
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

    #[test]
    fn f32_as_bytes_correct_length() {
        let data = [1.0f32, 2.0, 3.0];
        let bytes = f32_as_bytes(&data);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn f32_as_bytes_roundtrip() {
        let original = [1.0f32, 2.0, 3.0];
        let bytes = f32_as_bytes(&original);
        let restored: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(restored, &original);
    }
}
