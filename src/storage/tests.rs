use super::*;

#[test]
fn ensure_sqlite_vec_idempotent() {
    ensure_sqlite_vec().unwrap();
    ensure_sqlite_vec().unwrap();
}
