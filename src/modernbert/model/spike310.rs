//! #310 isolated-copy candidate and full-element oracle; never a product option.
use mlx_rs::{Array, error::Exception, memory, ops::select};
use std::time::Instant;

use crate::{
    mlx_cache::{Component, clear_inference_cache},
    sandbox::require_unsandboxed_mlx_runtime,
};

use super::{MASK_FILL, get_local_attention_mask, prepare_4d_attention_mask};

pub(super) fn gpu_mask(seq_len: i32, half: i32) -> Result<Array, Exception> {
    if seq_len < 0 || half < 0 {
        return Err(Exception::custom("negative mask dimension/window"));
    }
    let positions = Array::arange::<_, i32>(0, seq_len, None)?;
    let distance = positions
        .reshape(&[seq_len, 1])?
        .subtract(positions.reshape(&[1, seq_len])?)?
        .abs()?;
    // Selection, not multiplication by -inf: preserves the local fill semantics.
    select(
        distance.le(Array::from_int(half))?,
        Array::from_f32(0.0),
        Array::from_f32(f32::NEG_INFINITY),
    )
}

#[test]
#[ignore = "Issue #310: full mask readback on unsandboxed Metal"]
fn all_mask_elements_and_costs() {
    require_unsandboxed_mlx_runtime();
    for n in [1_i32, 8, 9, 17, 128, 512, 2048, 8192] {
        for half in [0_i32, 3, 64] {
            for gpu in [false, true] {
                memory::reset_peak_memory().unwrap();
                let started = Instant::now();
                let local = if gpu {
                    gpu_mask(n, half)
                } else {
                    get_local_attention_mask(n, half)
                }
                .unwrap();
                let graph_ns = started.elapsed().as_nanos();
                local.eval().unwrap();
                let eval_ns = started.elapsed().as_nanos() - graph_ns;
                let readback_started = Instant::now();
                let values = local.as_slice::<f32>();
                let local_readback_ns = readback_started.elapsed().as_nanos();
                for (index, &value) in values.iter().enumerate() {
                    let row = i32::try_from(index).unwrap() / n;
                    let col = i32::try_from(index).unwrap() % n;
                    assert_eq!(
                        value,
                        if (row - col).abs() <= half {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        },
                        "n={n} half={half} row={row} col={col}"
                    );
                }
                // Two rows, right padding and interior masked keys. Every batch/
                // query/key element, including both bidirectional endpoints.
                let padding: Vec<u32> = (0..2 * n)
                    .map(|i| u32::from(i % n == 0 || i % n < n / 2 && i % 3 != 1))
                    .collect();
                let global =
                    prepare_4d_attention_mask(&Array::from_slice(&padding, &[2, n]), 2, n).unwrap();
                let combined = global.add(&local).unwrap();
                combined.eval().unwrap();
                for (index, &value) in combined.as_slice::<f32>().iter().enumerate() {
                    let index = i32::try_from(index).unwrap();
                    let batch = index / (n * n);
                    let row = index / n % n;
                    let col = index % n;
                    let expected = if (row - col).abs() > half {
                        f32::NEG_INFINITY
                    } else if padding[usize::try_from(batch * n + col).unwrap()] == 0 {
                        MASK_FILL
                    } else {
                        0.0
                    };
                    assert_eq!(value, expected);
                }
                eprintln!(
                    "mask310 {}",
                    serde_json::json!({"event":"mask_elements", "seq_len":n, "half_window":half, "gpu":gpu, "elements":2*n*n, "graph_ns":graph_ns, "eval_ns":eval_ns, "local_readback_ns":local_readback_ns, "metal_peak_bytes":memory::peak_memory().unwrap()})
                );
                drop(combined);
                drop(global);
                drop(local);
                clear_inference_cache(Component::Embed);
            }
        }
    }
}
