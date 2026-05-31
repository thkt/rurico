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

/// Clear the MLX compiled Metal kernel cache.
///
/// Returns the MLX return code (0 = success).
///
/// # Caller contract
/// Same as [`mlx_clear_cache`].
#[allow(unsafe_code)]
pub fn mlx_compile_clear_cache() -> i32 {
    // SAFETY: same invariants as mlx_clear_cache.
    unsafe { mlx_sys::mlx_detail_compile_clear_cache() }
}
