"""Exact, fail-closed patches applied only to #310's isolated source copies."""
VARIANTS = ("current", "shared", "gpu", "buffer", "compile")


def replace(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"source changed: {path.name}: {old[:80]}")
    path.write_text(source.replace(old, new))


def patch(root, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    model = root / "src/modernbert/model.rs"
    # Identical lazy-stage instrumentation in every comparison copy; no eval.
    replace(model, "        let local_mask = if let Some(cached)",
            "        let mask_started = std::time::Instant::now();\n        let local_mask = if let Some(cached)")
    replace(model, "        for layer in &mut self.layers {\n            xs = layer.forward",
            '        eprintln!("mask310 {}", serde_json::json!({"event":"mask_graph", "seq_len":seq_len, "batch_size":batch_size, "cache_shapes":self.local_mask_cache.len(), "local_graph_ns":mask_started.elapsed().as_nanos()}));\n        for layer in &mut self.layers {\n            xs = layer.forward')
    replace(model, "        let mask = if self.uses_local_attention {",
            "        let compose_started = std::time::Instant::now();\n        let mask = if self.uses_local_attention {")
    replace(model, "        let output = scaled_dot_product_attention(q,",
            '        eprintln!("mask310 {}", serde_json::json!({"event":"compose_graph", "local":self.uses_local_attention, "compose_graph_ns":compose_started.elapsed().as_nanos()}));\n        let output = scaled_dot_product_attention(q,')
    if variant == "shared":
        old = "                global_mask.add(local)?"
        replace(model, old, "                local.clone()")
        replace(model, "        for layer in &mut self.layers {\n            xs = layer.forward(&xs, &global_mask, Some(&local_mask))?;",
                "        let local_mask = global_mask.add(&local_mask)?;\n        for layer in &mut self.layers {\n            xs = layer.forward(&xs, &global_mask, Some(&local_mask))?;")
    elif variant == "gpu":
        replace(model, "            let m = get_local_attention_mask(seq_len, half)?;",
                "            let m = spike310::gpu_mask(seq_len, half)?;")
    elif variant in ("buffer", "compile"):
        path = root / "src/mlx_cache.rs"
        replace(path, "    let result = forward().and_then(readback);\n    cleanup();",
                ('    mlx_rs::memory::set_cache_limit(64 * 1024 * 1024).expect("experimental cache limit");\n'
                 if variant == "buffer" else "") +
                "    let result = forward().and_then(readback);\n    SPIKE_FAILED.set(result.is_err());\n    cleanup();")
        replace(path, "use std::sync::Mutex;", """use std::sync::Mutex;
thread_local! {
    static SPIKE_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
""" + ('    static SPIKE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };\n'
       if variant == "compile" else "") + """}
""")
        if variant == "buffer":
            replace(path, "        rurico_ffi::mlx_clear_cache,", """        || {
            if SPIKE_FAILED.get() || mlx_rs::memory::cache_memory().expect("cache observation") > 64 * 1024 * 1024 {
                rurico_ffi::mlx_clear_cache()
            } else { 0 }
        },""")
        else:
            replace(path, "        rurico_ffi::mlx_compile_clear_cache,", """        || {
            let count = SPIKE_CALLS.get() + 1;
            if SPIKE_FAILED.get() || count >= 8 {
                SPIKE_CALLS.set(0);
                rurico_ffi::mlx_compile_clear_cache()
            } else {
                SPIKE_CALLS.set(count);
                Ok(())
            }
        },""")
    # Observe actual clear calls without changing the existing FFI error contracts.
    path = root / "src/mlx_cache.rs"
    replace(path, "use std::sync::Mutex;", """use std::sync::Mutex;
fn spike_buffer_clear() -> i32 {
    eprintln!("mask310 {}", serde_json::json!({"event":"buffer_clear"}));
    rurico_ffi::mlx_clear_cache()
}
fn spike_compile_clear() -> Result<(), rurico_ffi::CompileCacheError> {
    eprintln!("mask310 {}", serde_json::json!({"event":"compile_clear"}));
    rurico_ffi::mlx_compile_clear_cache()
}
""")
    source = path.read_text()
    # Replace uses in the caller only; wrappers continue to invoke real FFI.
    start = source.index("pub(crate) fn clear_inference_cache(component:")
    end = source.index("\nfn clear_inference_cache_with", start)
    caller = source[start:end].replace("rurico_ffi::mlx_clear_cache", "spike_buffer_clear").replace("rurico_ffi::mlx_compile_clear_cache", "spike_compile_clear")
    path.write_text(source[:start] + caller + source[end:])
    # Record memory only after the existing resource-scope cleanup.
    path = root / "src/mlx_cache.rs"
    replace(path, "    cleanup();\n    result", """    cleanup();
    eprintln!("mask310 {}", serde_json::json!({"event":"cleanup", "failed":result.is_err(),
        "active_bytes":mlx_rs::memory::active_memory().expect("active memory"),
        "cache_bytes":mlx_rs::memory::cache_memory().expect("cache memory")}));
    result""")
    runtime = root / "src/mlx_cache/testing/runtime.rs"
    for kind, label in (("embedding", "embed"), ("reranker", "reranker")):
        old = f'        .expect("cache {label} model first");'
        replace(runtime, old, old + f'\n    eprintln!("mask310 {{}}", serde_json::json!({{"event":"model_files", "kind":"{kind}", "paths":[artifacts.paths.model, artifacts.paths.config, artifacts.paths.tokenizer]}}));')
