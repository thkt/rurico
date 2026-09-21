use super::{EmbedError, pool_output};
use mlx_rs::Array;

// T-014 / FR-002b / AC-1
//
// [T-014] Compile-time signature lock for `pool_output`. Mirrors T-004
// in pooling.rs which guards `gpu_pool_and_normalize` against a future
// refactor that relaxes `output: Array` to `&Array` (which would
// defeat the drop-before-clear contract carried by
// `release_inference_output`). `pool_output` is the layer above and
// must not relax the same contract — relaxing it here would re-expose
// the same regression vector at the higher abstraction.
//
// The coercion also pins the **return** type as owned `Array` (not
// `&Array`); a refactor returning `Result<&Array, _>` would similarly
// defeat `release_inference_output(pooled)` and is caught by the same
// line below.
#[test]
fn pool_output_signature_consumes_output_by_value() {
    let _coerce: fn(Array, &[u32], i32, i32) -> Result<Array, EmbedError> = pool_output;
}
