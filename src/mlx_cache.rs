//! Process-global MLX cache lock shared by embed and reranker modules.
//!
//! Both embed and reranker clear the global buffer pool and the current
//! thread's compile cache after inference. A single `Mutex` serializes the
//! entire cleanup, including compile-cache handle acquisition and release.

use std::sync::Mutex;

/// Caller identifier for cache-clear telemetry.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Component {
    Embed,
    Reranker,
}

impl Component {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Reranker => "reranker",
        }
    }
}

/// Process-global lock for [`mlx_sys::mlx_clear_cache`] and [`mlx_sys::mlx_detail_compile_clear_cache`] calls.
///
/// Poison recovery is best-effort. The Rust-side guard is stateless (`()`), but
/// the protected MLX cache state is process-global FFI state and may be
/// internally inconsistent after a panic during a prior cache-clear call.
/// Continuing is acceptable because the worst expected failure mode is a
/// leaked compile-cache entry, not a Rust memory-safety violation.
pub(crate) static MLX_CACHE_LOCK: Mutex<()> = Mutex::new(());

/// Consume an MLX output array and attempt to clear the GPU cache.
///
/// Takes `output` by value to enforce drop-before-clear ordering at compile
/// time. Cache-clear failures are non-fatal: logged as warnings.
///
/// Clears both the buffer pool (`mlx_clear_cache`) and the current thread's
/// compile cache (`mlx_detail_compile_clear_cache`). The compile cache grows
/// with each unique `(batch_size, seq_len)` pair; clearing it after every
/// batch prevents Metal OOM across long embedding runs.
///
/// # Safety (caller invariants)
/// 1. `output` is taken by value — no borrows remain after this call.
/// 2. Model weights must remain live on the caller — only unused cache buffers are freed.
/// 3. `MLX_CACHE_LOCK` serializes concurrent calls across all modules.
pub(crate) fn release_inference_output(output: mlx_rs::Array, component: Component) {
    drop(output);
    clear_inference_cache(component);
}

/// Clear the MLX GPU caches without consuming an Array.
///
/// Sibling of [`release_inference_output`] for the case where a forward
/// pass succeeded (model weights uploaded, kernels compiled) but a
/// downstream MLX op (`gpu_pool_and_normalize`, `eval`) errored, leaving
/// no Array to drop. Skips the drop step but keeps the same cache-clear
/// contract so error paths do not leak compile-cache entries across long
/// embedding runs (Codex CX-001 regression guard for Phase 3b).
///
/// # Safety
///
/// Only call this when there is no live `Array` from the just-failed
/// forward pass. If an Array exists, prefer [`release_inference_output`]
/// to preserve the drop-before-clear ordering.
pub(crate) fn clear_inference_cache(component: Component) {
    clear_inference_cache_with(
        component,
        rurico_ffi::mlx_clear_cache,
        rurico_ffi::mlx_compile_clear_cache,
    );
}

fn clear_inference_cache_with(
    component: Component,
    clear_buffers: impl FnOnce() -> i32,
    clear_compile: impl FnOnce() -> Result<(), rurico_ffi::CompileCacheError>,
) {
    let component = component.as_str();
    let _guard = MLX_CACHE_LOCK.lock().unwrap_or_else(|e| {
        tracing::warn!(component, "MLX cache lock was poisoned; recovering");
        e.into_inner()
    });
    let code = clear_buffers();
    if code != 0 {
        tracing::warn!(component, code, "mlx_clear_cache failed");
    }
    if let Err(error) = clear_compile() {
        tracing::warn!(component, ?error, "MLX compile cache cleanup failed");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::TryLockError;

    use rurico_ffi::CompileCacheError;
    use tracing_test::traced_test;

    use super::{Component, MLX_CACHE_LOCK, clear_inference_cache, clear_inference_cache_with};

    #[test]
    #[traced_test]
    fn current_cache_cleanup_succeeds_without_warnings() {
        // Exercises the production caller and both real FFI wrappers. No model
        // is needed, but MLX may initialize Metal: use an unsandboxed GPU host.
        clear_inference_cache(Component::Embed);
        logs_assert(|logs| {
            if logs.is_empty() {
                Ok(())
            } else {
                Err(format!("cache cleanup emitted warnings: {logs:?}"))
            }
        });
    }

    #[test]
    #[traced_test]
    fn cleanup_failures_are_logged_under_the_shared_lock() {
        clear_inference_cache_with(
            Component::Embed,
            || {
                assert!(matches!(
                    MLX_CACHE_LOCK.try_lock(),
                    Err(TryLockError::WouldBlock)
                ));
                7
            },
            || {
                // A buffer failure must not skip compile-cache cleanup.
                assert!(matches!(
                    MLX_CACHE_LOCK.try_lock(),
                    Err(TryLockError::WouldBlock)
                ));
                Err(CompileCacheError {
                    acquire_code: 11,
                    clear_code: None,
                    free_code: 33,
                })
            },
        );
        clear_inference_cache_with(
            Component::Reranker,
            || 0,
            || {
                Err(CompileCacheError {
                    acquire_code: 0,
                    clear_code: Some(22),
                    free_code: 0,
                })
            },
        );
        assert!(logs_contain("mlx_clear_cache failed"));
        assert!(logs_contain("code=7"));
        assert!(logs_contain("MLX compile cache cleanup failed"));
        assert!(logs_contain(
            "acquire_code: 11, clear_code: None, free_code: 33"
        ));
        assert!(logs_contain("clear_code: Some(22)"));
        assert!(logs_contain("component=\"embed\""));
        assert!(logs_contain("component=\"reranker\""));
    }
}
