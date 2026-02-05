use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use rustorch::governance::DeterministicScheduler;

fn sample_run_dirs(count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|idx| PathBuf::from(format!("/tmp/schedule-run-{idx}")))
        .collect()
}

#[test]
fn scheduler_is_deterministic_under_concurrency() {
    let run_dirs = Arc::new(sample_run_dirs(64));
    let seed = 1337_u64;
    let mut handles = Vec::new();

    for _ in 0..4 {
        let run_dirs = Arc::clone(&run_dirs);
        handles.push(thread::spawn(move || {
            let scheduler =
                DeterministicScheduler::new(seed, run_dirs.as_ref().clone()).expect("scheduler");
            scheduler
                .drain()
                .into_iter()
                .map(|entry| entry.run_id)
                .collect::<Vec<String>>()
        }));
    }

    let first = handles
        .pop()
        .expect("handle")
        .join()
        .expect("join");
    for handle in handles {
        let next = handle.join().expect("join");
        assert_eq!(first, next);
    }
}

#[test]
fn scheduler_handles_large_inputs() {
    let run_dirs = sample_run_dirs(2048);
    let scheduler = DeterministicScheduler::new(9001, run_dirs).expect("scheduler");
    let entries = scheduler.drain();
    assert_eq!(entries.len(), 2048);
    let mut run_ids = entries
        .into_iter()
        .map(|entry| entry.run_id)
        .collect::<Vec<String>>();
    run_ids.sort();
    run_ids.dedup();
    assert_eq!(run_ids.len(), 2048);
}
