//! Safe FFI wrappers for rurico.
//!
//! This crate isolates all `unsafe` code behind safe public functions.
//! Workspace lints deny `unsafe`; the narrow FFI exemptions live here.

mod mlx;
mod process;
mod storage;

pub use mlx::{CompileCacheError, mlx_clear_cache, mlx_compile_clear_cache};
pub use process::{kill_process_group, process_group_exists, set_nonblocking};
pub use storage::sqlite_vec_register;
