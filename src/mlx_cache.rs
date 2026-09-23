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

/// Run forward and CPU readback in one resource scope, then clean up once.
///
/// `forward` must own any partial inference Arrays; `readback` consumes its
/// output and must return only CPU data/errors, retaining no inference Arrays.
/// Both closures (including unused captures on forward failure) are dropped
/// before cleanup. Model weights may remain live in the caller.
///
/// Returned errors are preserved. This covers `Result` failures, not panics or
/// backend aborts. Cleanup remains best-effort and must not replace the result.
pub(crate) fn run_inference<A, T, E>(
    forward: impl FnOnce() -> Result<A, E>,
    readback: impl FnOnce(A) -> Result<T, E>,
    cleanup: impl FnOnce(),
) -> Result<T, E> {
    let result = forward().and_then(readback);
    cleanup();
    result
}

/// Clear the buffer pool and current thread's compile cache after all temporary
/// inference Arrays have been dropped. Keep the conservative per-forward policy;
/// cache-policy optimization and GPU memory bounds require separate evidence.
pub(crate) fn clear_inference_cache(component: Component) {
    #[cfg(test)]
    testing::record_cleanup();
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

#[cfg(test)]
pub(crate) mod testing;
