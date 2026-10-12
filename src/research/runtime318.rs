//! Opt-in metadata observations. No extra eval, readback, or Array retention.
use mlx_rs::Array;
use std::cell::Cell;
#[cfg(test)]
use std::thread;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct Observation;
impl Observation {
    pub(crate) fn begin() -> Self {
        ENABLED.set(true);
        Self
    }
}
impl Drop for Observation {
    fn drop(&mut self) {
        ENABLED.set(false);
    }
}

pub(crate) fn tensor(stage: &str, array: &Array) {
    if ENABLED.get() {
        eprintln!(
            "runtime318 {}",
            serde_json::json!({"event":"tensor", "stage":stage,
            "shape":array.shape(), "dtype":format!("{:?}", array.dtype()),
            "observation":"lazy Array metadata; not allocated or live GPU bytes"})
        );
    }
}

#[test]
fn observations_are_scoped_and_do_not_enable_other_threads() {
    assert!(!ENABLED.get());
    {
        let _observation = Observation::begin();
        assert!(ENABLED.get());
        thread::spawn(|| assert!(!ENABLED.get())).join().unwrap();
        // A small model-free Array verifies metadata access without eval.
        let values = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        tensor("test", &values);
        assert_eq!(values.shape(), &[1, 2]);
        assert_eq!(values.dtype(), mlx_rs::Dtype::Float32);
    }
    assert!(!ENABLED.get());
}
