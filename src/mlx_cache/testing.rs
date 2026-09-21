//! Failure injection is test-only and thread-local; production has no fault mode.
use std::cell::Cell;

use mlx_rs::error::Exception;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    Forward,
    Pool,
    Eval,
    Readback,
}

thread_local! {
    static FAILURE: Cell<Option<Stage>> = const { Cell::new(None) };
    static CLEANUPS: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn checkpoint(stage: Stage) -> Result<(), Exception> {
    if FAILURE.get() == Some(stage) {
        Err(Exception::custom(format!("injected {stage:?} failure")))
    } else {
        Ok(())
    }
}

pub(crate) fn record_cleanup() {
    CLEANUPS.set(CLEANUPS.get() + 1);
}

#[test]
fn inference_drops_resources_before_one_cleanup_and_preserves_result() {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::run_inference;

    struct Resource<'a>(&'static str, &'a RefCell<Vec<&'static str>>);
    impl Drop for Resource<'_> {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    // No GPU: deterministic lifetime/error regression, not memory measurement.
    for failure in [
        None,
        Some(Stage::Forward),
        Some(Stage::Pool),
        Some(Stage::Eval),
        Some(Stage::Readback),
    ] {
        let events = RefCell::new(Vec::new());
        let original = Rc::new(failure);
        let fail = |stage| {
            if failure == Some(stage) {
                Err(Rc::clone(&original))
            } else {
                Ok(())
            }
        };
        let readback_capture = Resource("readback capture", &events);
        let result = run_inference(
            || {
                let partial = Resource("forward", &events);
                fail(Stage::Forward)?;
                Ok(partial)
            },
            |output| {
                let _capture = readback_capture;
                fail(Stage::Pool)?;
                let _pooled = Resource("pooled", &events);
                drop(output);
                fail(Stage::Eval)?;
                fail(Stage::Readback)?;
                Ok(vec![0.25_f32, -0.5])
            },
            || events.borrow_mut().push("cleanup"),
        );
        if failure.is_some() {
            assert!(Rc::ptr_eq(&result.unwrap_err(), &original));
        } else {
            assert_eq!(result.unwrap(), vec![0.25, -0.5]);
        }
        let mut events = events.into_inner();
        assert_eq!(events.pop(), Some("cleanup"), "{failure:?}");
        // Relative order among Arrays is immaterial; every resource must be
        // released exactly once, and all releases must precede cleanup.
        events.sort_unstable();
        let expected = if matches!(failure, Some(Stage::Forward | Stage::Pool)) {
            vec!["forward", "readback capture"]
        } else {
            vec!["forward", "pooled", "readback capture"]
        };
        assert_eq!(events, expected, "{failure:?}");
    }
}

#[test]
fn injection_controls_select_only_the_requested_stage_and_thread() {
    use std::thread;

    let stages = [
        (Stage::Forward, "injected Forward failure"),
        (Stage::Pool, "injected Pool failure"),
        (Stage::Eval, "injected Eval failure"),
        (Stage::Readback, "injected Readback failure"),
    ];
    for (stage, _) in stages {
        assert!(checkpoint(stage).is_ok(), "injection must default to off");
    }
    for (selected, message) in stages {
        FAILURE.set(Some(selected));
        for (stage, _) in stages {
            let result = checkpoint(stage);
            if stage == selected {
                assert_eq!(result.unwrap_err().what(), message);
            } else {
                result.unwrap();
            }
        }
        // Injection must persist until reset, even if a checkpoint is repeated.
        assert_eq!(checkpoint(selected).unwrap_err().what(), message);
        FAILURE.set(None);
        for (stage, _) in stages {
            checkpoint(stage).unwrap();
        }
    }

    FAILURE.set(Some(Stage::Forward));
    CLEANUPS.set(0);
    record_cleanup();
    thread::spawn(|| {
        checkpoint(Stage::Forward).unwrap();
        assert_eq!(CLEANUPS.get(), 0);
        FAILURE.set(Some(Stage::Readback));
        record_cleanup();
        record_cleanup();
        assert_eq!(CLEANUPS.get(), 2);
    })
    .join()
    .unwrap();
    assert_eq!(
        checkpoint(Stage::Forward).unwrap_err().what(),
        "injected Forward failure"
    );
    checkpoint(Stage::Readback).unwrap();
    assert_eq!(CLEANUPS.get(), 1);
    FAILURE.set(None);
    CLEANUPS.set(0);
}

#[cfg(feature = "test-mlx")]
mod runtime;
