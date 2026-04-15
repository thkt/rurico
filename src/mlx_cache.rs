//! Process-global MLX cache lock shared by embed and reranker modules.
//!
//! `mlx_clear_cache()` and `mlx_detail_compile_clear_cache()` are global FFI
//! calls that are not thread-safe. Both embed and reranker call them after
//! inference to free GPU memory. A single `Mutex` ensures these calls are
//! serialized.

use std::sync::Mutex;

/// Process-global lock for [`mlx_sys::mlx_clear_cache`] and [`mlx_sys::mlx_detail_compile_clear_cache`] calls.
///
/// Recover from poison: cache-clear is stateless (guards `()`), safe to
/// proceed after a panic in another thread.
pub(crate) static MLX_CACHE_LOCK: Mutex<()> = Mutex::new(());

/// Consume an MLX output array and attempt to clear the GPU cache.
///
/// Takes `output` by value to enforce drop-before-clear ordering at compile
/// time. Cache-clear failures are non-fatal: logged as warnings.
///
/// Clears both the buffer pool (`mlx_clear_cache`) and the compiled Metal
/// kernel cache (`mlx_detail_compile_clear_cache`). The compile cache grows
/// with each unique `(batch_size, seq_len)` pair; clearing it after every
/// batch prevents Metal OOM across long embedding runs.
///
/// # Safety (caller invariants)
/// 1. `output` is taken by value — no borrows remain after this call.
/// 2. Model weights must remain live on the caller — only unused cache buffers are freed.
/// 3. `MLX_CACHE_LOCK` serializes concurrent calls across all modules.
pub(crate) fn release_inference_output(output: mlx_rs::Array) {
    drop(output);
    let _guard = MLX_CACHE_LOCK.lock().unwrap_or_else(|e| {
        log::warn!("MLX cache lock was poisoned; recovering");
        e.into_inner()
    });
    let code = rurico_ffi::mlx_clear_cache();
    if code != 0 {
        log::warn!("mlx_clear_cache failed (code: {code})");
    }
    let code = rurico_ffi::mlx_compile_clear_cache();
    if code != 0 {
        log::warn!("mlx_detail_compile_clear_cache failed (code: {code})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_cache_lock_is_acquirable() {
        let guard = MLX_CACHE_LOCK.lock().unwrap();
        drop(guard);
    }
}
