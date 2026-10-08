//! CPU-only source probe; generated module copies are byte-for-byte source.
//! This standalone harness never loads rurico's MLX modules.
mod baseline;
mod current;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::{CStr, c_void};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use rusqlite::{Connection, ffi};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
static VALIDATIONS: AtomicUsize = AtomicUsize::new(0);
// Only the standalone probe uses unsafe allocator/SQLite observation APIs.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) { ALLOCS.fetch_add(1, Ordering::Relaxed); }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { unsafe { System.dealloc(ptr, layout) } }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) { ALLOCS.fetch_add(1, Ordering::Relaxed); }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
unsafe extern "C" fn trace(_: u32, _: *mut c_void, stmt: *mut c_void, _: *mut c_void) -> i32 {
    let sql = unsafe { CStr::from_ptr(ffi::sqlite3_sql(stmt.cast())) };
    if sql.to_bytes().starts_with(b"SELECT term FROM vocab ") {
        LOOKUPS.fetch_add(1, Ordering::Relaxed);
    }
    if sql.to_bytes().starts_with(b"SELECT term, cnt FROM vocab WHERE 0") {
        VALIDATIONS.fetch_add(1, Ordering::Relaxed);
    }
    0
}
fn norm(version: usize, input: &str) -> String {
    if version == 0 {
        baseline::query_normalize::normalize_for_fts(input, &Default::default())
    } else {
        current::query_normalize::normalize_for_fts(input, &Default::default())
    }
}
fn query(version: usize, conn: &Connection, input: &str) -> String {
    if version == 0 {
        baseline::search::prepare_match_query(conn, input, "vocab", &Default::default()).unwrap().into_string()
    } else {
        current::search::prepare_match_query(conn, input, "vocab", &Default::default()).unwrap().into_string()
    }
}
fn measure(mut action: impl FnMut() -> String, iterations: usize) -> (f64, usize) {
    for _ in 0..20 { black_box(action()); }
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    black_box(action());
    COUNTING.store(false, Ordering::Relaxed);
    let allocations = ALLOCS.load(Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..iterations { black_box(action()); }
    (start.elapsed().as_nanos() as f64 / iterations as f64, allocations)
}
fn main() {
    // Maybe plus a changed suffix exposes repeated normalization of the prefix.
    let maybe_long = format!("{}e\u{301}", "日本語 prefix ".repeat(128));
    let maybe_expected = format!("{}é", "日本語 prefix ".repeat(128));
    for version in 0..2 {
        assert_eq!(norm(version, "e\u{301}"), "é");
        assert_eq!(norm(version, &maybe_long), maybe_expected);
        assert_eq!(norm(version, "x\u{301}"), "x\u{301}");
    }
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE VIRTUAL TABLE docs USING fts5(body); CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row);").unwrap();
    for n in 0..1000 {
        conn.execute("INSERT INTO docs VALUES (?1)", [format!("audit{n:04} authentication{n:04} login")]).unwrap();
    }
    // Different literal oracle and all eight configs, including disabled.
    for input in ["", " \t\n　", "react hooks", "日本語 カタカナ", "ＡＢＣ　Foo", "e\u{301} %_\\", "foo\u{2003}bar  baz", "Ａ\"B-C"] {
        for bits in 0..8 {
            let old = baseline::query_normalize::QueryNormalizationConfig { nfkc: bits & 4 != 0, ascii_lowercase: bits & 2 != 0, collapse_whitespace: bits & 1 != 0 };
            let new = current::query_normalize::QueryNormalizationConfig { nfkc: old.nfkc, ascii_lowercase: old.ascii_lowercase, collapse_whitespace: old.collapse_whitespace };
            assert_eq!(baseline::query_normalize::normalize_for_fts(input, &old), current::query_normalize::normalize_for_fts(input, &new));
        }
    }
    let long_unicode = "日本語".repeat(1024);
    let queries = [("long", "authentication login"), ("long_unicode", long_unicode.as_str()), ("unique", "au zz"), ("repeated", "au au au au"), ("repeated_miss", "zz zz zz zz")];
    for (_, input) in queries { assert_eq!(query(0, &conn, input), query(1, &conn, input)); }
    // Observe this database again after update, never reusing across calls.
    conn.execute("INSERT INTO docs VALUES ('autumn')", []).unwrap();
    for (_, input) in queries { assert_eq!(query(0, &conn, input), query(1, &conn, input)); }
    // Trace only in the query-count experiment, outside timed intervals.
    unsafe { assert_eq!(ffi::sqlite3_trace_v2(conn.handle(), ffi::SQLITE_TRACE_STMT, Some(trace), std::ptr::null_mut()), ffi::SQLITE_OK); }
    for (name, input) in queries {
        for version in 0..2 {
            LOOKUPS.store(0, Ordering::Relaxed);
            VALIDATIONS.store(0, Ordering::Relaxed);
            black_box(query(version, &conn, input));
            println!("{}", serde_json::json!({"kind":"lookup", "case":name, "version":version, "count":LOOKUPS.load(Ordering::Relaxed), "schema_validations":VALIDATIONS.load(Ordering::Relaxed)}));
        }
    }
    unsafe { ffi::sqlite3_trace_v2(conn.handle(), 0, None, std::ptr::null_mut()); }
    for repeat in 0..7 {
        for version in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
            for (name, input) in queries {
                let (ns, allocations) = measure(|| query(version, &conn, input), 1000);
                println!("{}", serde_json::json!({"kind":"query", "case":name, "version":version, "repeat":repeat, "ns_per_call":ns, "rust_allocations":allocations}));
            }
        }
    }
}
