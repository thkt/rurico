//! Process-global MLX cache lock shared by embed and reranker modules.
//!
//! `mlx_clear_cache()` is a global FFI call that is not thread-safe.
//! Both embed and reranker call it after inference to free GPU memory.
//! A single `Mutex` ensures these calls are serialized.

/// Process-global lock for [`mlx_sys::mlx_clear_cache`] calls.
///
/// Recover from poison: cache-clear is stateless (guards `()`), safe to
/// proceed after a panic in another thread.
pub(crate) static MLX_CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Consume an MLX output array and attempt to clear the GPU cache.
///
/// Takes `output` by value to enforce drop-before-clear ordering at compile
/// time. Cache-clear failure is non-fatal: logged as a warning.
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
    // SAFETY: enforced by the caller invariants documented above.
    let code = unsafe { mlx_sys::mlx_clear_cache() };
    if code != 0 {
        log::warn!("mlx_clear_cache failed (code: {code})");
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
