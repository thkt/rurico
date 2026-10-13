//! Lock lifetime and wall/lock metrics for the experimental entry.
use super::*;
use crate::embed::query317::Queries;
pub(super) fn measured_entry<T>(
    lock: impl FnOnce() -> Result<T, EmbedError>,
    run: impl FnOnce(&mut T) -> Result<Queries, EmbedError>,
) -> Result<Queries, EmbedError> {
    let start = Instant::now();
    let mut inner = lock()?;
    let lock_wait = start.elapsed();
    let mut measured = run(&mut inner)?;
    drop(inner);
    let metrics = measured.metrics.as_mut().unwrap();
    metrics.lock_wait = lock_wait;
    metrics.wall = start.elapsed();
    Ok(measured)
}

#[test]
fn measured_entry_releases_lock_and_preserves_failures() {
    use std::sync::Mutex;
    let lock = Mutex::new(0);
    let result = measured_entry(
        || Ok(lock.lock().unwrap()),
        |guard| {
            **guard += 1;
            assert!(lock.try_lock().is_err());
            Ok(Queries {
                values: vec![vec![3., -2.]],
                metrics: Some(InferenceMetrics::default()),
            })
        },
    )
    .unwrap();
    assert_eq!(*lock.try_lock().unwrap(), 1);
    assert_eq!(result.values, [vec![3., -2.]]);
    let metrics = result.metrics.unwrap();
    assert!(metrics.wall >= metrics.lock_wait);
    assert!(matches!(
        measured_entry(
            || Err::<(), _>(EmbedError::NonFiniteOutput),
            |_| panic!("run after lock failure")
        ),
        Err(EmbedError::NonFiniteOutput)
    ));
    assert!(matches!(
        measured_entry(
            || Ok(lock.lock().unwrap()),
            |_| Err(EmbedError::NonFiniteOutput)
        ),
        Err(EmbedError::NonFiniteOutput)
    ));
    assert!(lock.try_lock().is_ok());
}
