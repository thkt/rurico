//! Safe wrapper for sqlite-vec FFI registration.

use std::os::raw::{c_char, c_int};

use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};

// Pull in the upstream `sqlite-vec` crate solely so its `build.rs` produces
// `libsqlite_vec0.a` and that static library reaches the link step. We do
// not import its `sqlite3_vec_init` symbol because the upstream binding is
// declared as zero-arg (`extern "C" { fn sqlite3_vec_init(); }`), which
// hides the real C ABI and would force a `transmute` at the
// `sqlite3_auto_extension` call site.
use sqlite_vec as _;

// Re-declare `sqlite3_vec_init` with the real signature from `sqlite-vec.h`,
// matching `int sqlite3_vec_init(sqlite3*, char**, const sqlite3_api_routines*)`.
// Declaring it here lets the compiler verify ABI compatibility at the
// `Some(sqlite3_vec_init)` coercion site instead of relying on a `transmute`.
#[allow(unsafe_code)]
unsafe extern "C" {
    fn sqlite3_vec_init(
        db: *mut sqlite3,
        pz_err_msg: *mut *mut c_char,
        p_api: *const sqlite3_api_routines,
    ) -> c_int;
}

/// Register sqlite-vec as a process-global SQLite auto-extension.
///
/// Returns the SQLite return code (0 = success).
///
/// Idempotent — `sqlite3_auto_extension` deduplicates registrations by
/// function pointer, so repeated calls are safe.
#[allow(unsafe_code)]
pub fn sqlite_vec_register() -> i32 {
    // SAFETY: the locally re-declared `sqlite3_vec_init` matches the C
    // signature in `sqlite-vec.h`, so the function-pointer coercion into
    // `sqlite3_auto_extension`'s expected type is ABI-correct without a
    // `transmute`. SQLite invokes the function with the declared argument
    // types at connection-open time.
    unsafe { sqlite3_auto_extension(Some(sqlite3_vec_init)) }
}

#[cfg(test)]
mod tests {
    use super::sqlite_vec_register;
    use rusqlite::Connection;

    #[test]
    fn sqlite_vec_register_enables_vec_version_function() {
        let rc = sqlite_vec_register();
        assert_eq!(rc, 0, "sqlite3_auto_extension returned non-zero rc {rc}");

        let conn = Connection::open_in_memory().unwrap();
        let version: String = conn
            .query_row("select vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(
            version.starts_with('v'),
            "expected sqlite-vec version to start with 'v', got {version:?}"
        );
    }
}
