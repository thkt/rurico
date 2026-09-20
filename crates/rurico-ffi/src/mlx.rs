//! Safe wrappers for `mlx_sys` FFI functions.

/// Clear the MLX buffer pool.
///
/// Returns the MLX return code (0 = success).
///
/// # Caller contract
/// - No MLX arrays are borrowed by the caller after this returns.
/// - Model weights remain live on the caller side.
/// - Serialization across threads is the caller's responsibility
///   (see `rurico::mlx_cache::MLX_CACHE_LOCK`).
#[allow(unsafe_code)]
pub fn mlx_clear_cache() -> i32 {
    // SAFETY: the caller must ensure no MLX arrays are borrowed at this point
    // and that concurrent calls are serialized via MLX_CACHE_LOCK. MLX mutates
    // process-global cache state internally; this wrapper makes no post-panic
    // consistency guarantee beyond forwarding the MLX return code.
    unsafe { mlx_sys::mlx_clear_cache() }
}

/// Return codes from a failed compile-cache cleanup (0 = success).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileCacheError {
    pub acquire_code: i32,
    /// `None` when acquisition failed and clearing was not attempted.
    pub clear_code: Option<i32>,
    pub free_code: i32,
}

/// Acquire and clear the current MLX compile cache, then release its handle.
///
/// Always attempts handle release, including after acquisition or clear failure.
/// Reports both the operation and release codes so cleanup cannot hide an error.
///
/// # Caller contract
/// Same as [`mlx_clear_cache`].
#[allow(unsafe_code)]
pub fn mlx_compile_clear_cache() -> Result<(), CompileCacheError> {
    clear_compile_cache_with(
        // SAFETY: new creates an empty handle. Acquire initializes it from the
        // current cache before clear; free accepts empty and initialized handles.
        // The helper frees exactly once and never uses the handle afterward.
        // Cache operations retain mlx_clear_cache's caller invariants.
        || unsafe { mlx_sys::mlx_compile_cache_new() },
        |cache| unsafe { mlx_sys::mlx_detail_compile_cache(cache) },
        |cache| unsafe { mlx_sys::mlx_detail_compile_clear_cache(cache) },
        |cache| unsafe { mlx_sys::mlx_compile_cache_free(cache) },
    )
}

fn clear_compile_cache_with<H: Copy>(
    new: impl FnOnce() -> H,
    acquire: impl FnOnce(&mut H) -> i32,
    clear: impl FnOnce(H) -> i32,
    free: impl FnOnce(H) -> i32,
) -> Result<(), CompileCacheError> {
    let mut cache = new();
    let acquire_code = acquire(&mut cache);
    let clear_code = (acquire_code == 0).then(|| clear(cache));
    let free_code = free(cache);
    if acquire_code == 0 && clear_code == Some(0) && free_code == 0 {
        Ok(())
    } else {
        Err(CompileCacheError {
            acquire_code,
            clear_code,
            free_code,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{CompileCacheError, clear_compile_cache_with};

    #[test]
    fn cleanup_preserves_failures_and_releases_the_acquired_handle() {
        // Distinct codes expose masking of an operation error by free's result.
        for (acquire_code, clear_code, free_code) in [
            (0, 0, 0),
            (11, 0, 0),
            (0, 22, 0),
            (0, 0, 33),
            (11, 0, 33),
            (0, 22, 33),
        ] {
            let calls = RefCell::new(Vec::new());
            let result = clear_compile_cache_with(
                || {
                    calls.borrow_mut().push("new");
                    0
                },
                |handle| {
                    calls.borrow_mut().push("acquire");
                    assert_eq!(*handle, 0);
                    if acquire_code == 0 {
                        *handle = 42;
                    }
                    acquire_code
                },
                |handle| {
                    calls.borrow_mut().push("clear");
                    assert_eq!(handle, 42, "clear must use the acquired cache");
                    clear_code
                },
                |handle| {
                    calls.borrow_mut().push("free");
                    assert_eq!(handle, if acquire_code == 0 { 42 } else { 0 });
                    free_code
                },
            );
            let expected = if acquire_code == 0 && clear_code == 0 && free_code == 0 {
                Ok(())
            } else {
                Err(CompileCacheError {
                    acquire_code,
                    clear_code: if acquire_code == 0 {
                        Some(clear_code)
                    } else {
                        None
                    },
                    free_code,
                })
            };
            assert_eq!(result, expected);
            let expected_calls = if acquire_code == 0 {
                vec!["new", "acquire", "clear", "free"]
            } else {
                vec!["new", "acquire", "free"]
            };
            assert_eq!(*calls.borrow(), expected_calls);
        }
    }
}
