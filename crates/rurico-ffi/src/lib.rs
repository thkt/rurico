//! Safe FFI wrappers for rurico.
//!
//! This crate isolates all `unsafe` code behind safe public functions.
//! `rurico` itself is `#![forbid(unsafe_code)]`; all FFI boundaries live here.

mod mlx;
mod storage;

pub use mlx::{mlx_clear_cache, mlx_compile_clear_cache};
pub use storage::sqlite_vec_register;
