//! Safe wrapper for sqlite-vec FFI registration.

use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

/// Register sqlite-vec as a process-global SQLite auto-extension.
///
/// Returns the SQLite return code (0 = success).
///
/// Idempotent — `sqlite3_auto_extension` deduplicates registrations by
/// function pointer, so repeated calls are safe.
#[allow(unsafe_code)]
pub fn sqlite_vec_register() -> i32 {
    // SAFETY: sqlite3_vec_init is the auto-extension entry point exported by
    // sqlite-vec. sqlite-vec exports it as `unsafe extern "C" fn()`, while
    // rusqlite's sqlite3_auto_extension expects the full init signature. Both
    // are C fn pointers with compatible calling conventions; SQLite calls the
    // function with the correct arguments at connection open time.
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::os::raw::c_int,
        >(sqlite3_vec_init)))
    }
}
