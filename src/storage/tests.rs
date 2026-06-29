use super::*;

use rusqlite::Connection;

#[test]
fn ensure_sqlite_vec_idempotent() {
    // First call registers the sqlite-vec auto-extension; the second must be a
    // no-op that still returns Ok — the idempotency contract of the OnceLock.
    ensure_sqlite_vec().unwrap();
    ensure_sqlite_vec().unwrap();

    // Outcome: a connection opened after registration actually has the vec
    // extension loaded — `vec0` resolves as a virtual-table module and a real
    // vector table can be created. Without registration this CREATE errors with
    // "no such module: vec0".
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE VIRTUAL TABLE vec_probe USING vec0(embedding float[4])")
        .expect("vec0 module must be available after ensure_sqlite_vec registers the extension");
}
