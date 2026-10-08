use super::*;

use rusqlite::{Connection, Error, ErrorCode, params};

const KNN_SQL: &str = "SELECT rowid, distance FROM vec_probe
                      WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance";

fn vector_connection() -> Connection {
    ensure_sqlite_vec().unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE vec_probe USING vec0(embedding float[3] distance_metric=l2)",
    )
    .unwrap();
    conn
}

fn neighbors(conn: &Connection, bytes: &[u8], k: i64) -> rusqlite::Result<Vec<(i64, f64)>> {
    conn.prepare(KNN_SQL)?
        .query_map(params![bytes, k], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect()
}

#[test]
fn ensure_sqlite_vec_idempotent() {
    // Each helper call checks registration before opening its connection;
    // repeated registration must succeed and both connections remain usable together.
    let connections = [vector_connection(), vector_connection()];
    for conn in &connections {
        let query = [1.0_f32, -2.0, 0.5];
        // Deliberately different insertion/rowid/distance order. Relative to
        // query, the offsets have L2 lengths 5, 10, 3, 2 (no ties).
        for (id, vector) in [
            (10, [4.0_f32, 2.0, 0.5]),
            (40, [11.0, -2.0, 0.5]),
            (90, [-1.0, 0.0, -0.5]),
            (70, [1.0, -2.0, 2.5]),
        ] {
            let bytes: &[u8] = bytemuck::cast_slice(&vector);
            conn.execute(
                "INSERT INTO vec_probe(rowid, embedding) VALUES (?1, ?2)",
                params![id, bytes],
            )
            .unwrap();
        }

        // Pin little-endian f32 storage independently of cast_slice: -1, 0, -0.5.
        let stored: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM vec_probe WHERE rowid = 90",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored, [0, 0, 128, 191, 0, 0, 0, 0, 0, 0, 0, 191]);

        let bytes: &[u8] = bytemuck::cast_slice(&query);
        assert_eq!(
            neighbors(conn, bytes, 3).unwrap(),
            [(70, 2.0), (90, 3.0), (10, 5.0)]
        );
    }
}

#[test]
fn vector_bind_rejects_invalid_bytes_and_dimensions() {
    let conn = vector_connection();
    let wrong_dimensions = [1.0_f32, -2.0];
    let cases: &[(&[u8], &str)] = &[
        (&[0, 0, 0], "invalid float32 vector BLOB length"),
        (&[], "zero-length vectors are not supported"),
        (
            bytemuck::cast_slice(&wrong_dimensions),
            "Dimension mismatch",
        ),
    ];
    for &(bytes, diagnostic) in cases {
        let insert_error = conn
            .execute(
                "INSERT INTO vec_probe(rowid, embedding) VALUES (1, ?1)",
                [bytes],
            )
            .unwrap_err();
        let query_error = neighbors(&conn, bytes, 3).unwrap_err();
        for error in [insert_error, query_error] {
            let Error::SqliteFailure(code, Some(message)) = error else {
                panic!("expected sqlite-vec error, got {error:?}");
            };
            assert_eq!(code.code, ErrorCode::Unknown); // SQLITE_ERROR
            assert!(message.contains(diagnostic), "{message}");
        }
    }
    // Invalid writes leave no candidates; a valid write/query still works.
    let count: i64 = conn
        .query_row("SELECT count(*) FROM vec_probe", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
    let vector = [1.0_f32, -2.0, 0.5];
    let bytes: &[u8] = bytemuck::cast_slice(&vector);
    conn.execute(
        "INSERT INTO vec_probe(rowid, embedding) VALUES (1, ?1)",
        [bytes],
    )
    .unwrap();
    assert_eq!(neighbors(&conn, bytes, 3).unwrap(), [(1, 0.0)]);
}

#[test]
fn vector_search_empty_table_returns_no_neighbors() {
    let conn = vector_connection();
    let query = [1.0_f32, -2.0, 0.5];
    let bytes: &[u8] = bytemuck::cast_slice(&query);
    assert_eq!(neighbors(&conn, bytes, 3).unwrap(), []);
}
