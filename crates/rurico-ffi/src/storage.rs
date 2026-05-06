//! Safe wrapper for sqlite-vec FFI registration.

use std::mem::transmute;
use std::os::raw::{c_char, c_int};

use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
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
        sqlite3_auto_extension(Some(transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(
                *mut sqlite3,
                *mut *mut c_char,
                *const sqlite3_api_routines,
            ) -> c_int,
        >(sqlite3_vec_init)))
    }
}
