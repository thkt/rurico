//! Safe wrapper for sqlite-vec FFI registration.

use std::os::raw::{c_char, c_int};

use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};

// Link `libsqlite_vec0.a` produced by the upstream `sqlite-vec` build script;
// the symbol itself is re-declared below with the real C signature.
use sqlite_vec as _;

// `sqlite3_vec_init` from `sqlite-vec.h`:
//   int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
//                        const sqlite3_api_routines *pApi);
#[allow(unsafe_code)]
unsafe extern "C" {
    fn sqlite3_vec_init(
        db: *mut sqlite3,
        err_msg: *mut *mut c_char,
        api: *const sqlite3_api_routines,
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
    // SAFETY: the local re-declaration of `sqlite3_vec_init` matches the
    // exact symbol signature exported by `libsqlite_vec0.a`. `extern "C" fn`
    // items have `'static` lifetime, which is what `sqlite3_auto_extension`
    // requires for an auto-extension entry.
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
