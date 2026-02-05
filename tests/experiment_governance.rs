use std::sync::Arc;
use std::thread;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, RunGovernanceConfig};

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_governance_test_{suffix}"))
}

fn create_run(store: &ExperimentStore, name: &str) {
    let mut run = store
        .create_run(name, serde_json::json!({}), Vec::new())
        .expect("run");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
}

#[test]
fn governance_quarantines_invalid_summary() {
    let root = temp_root();
    let store = ExperimentStore::new(&root).expect("store");
    let mut run = store
        .create_run("quarantine", serde_json::json!({}), Vec::new())
        .expect("run");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
    let run_id = run.metadata().id.clone();
    let summary_path = store.root().join(&run_id).join("run_summary.json");
    let mut summary_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).expect("read summary"))
            .expect("parse summary");
    summary_value["duration_secs"] = serde_json::json!(-1.0);
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary_value).expect("serialize"),
    )
    .expect("write summary");

    let mut config = RunGovernanceConfig::default();
    config.quarantine = true;
    let report = store.validate_runs(&config).expect("report");

    assert_eq!(report.summary.total_runs, 1);
    assert_eq!(report.summary.invalid_runs, 1);
    assert_eq!(report.summary.quarantined_runs, 1);
    let result = report.results.first().expect("result");
    assert!(result.quarantined);
    let quarantine_path = result.quarantine_path.as_ref().expect("path");
    assert!(quarantine_path.exists());
    assert!(!store.root().join(&run_id).exists());
}

#[test]
fn governance_parallel_validation_stress() {
    let root = temp_root();
    let store = Arc::new(ExperimentStore::new(&root).expect("store"));
    let mut handles = Vec::new();
    let runs_per_thread = 6;
    for idx in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for run_idx in 0..runs_per_thread {
                create_run(&store, &format!("stress-{idx}-{run_idx}"));
            }
        }));
    }
    for handle in handles {
        handle.join().expect("join");
    }

    let mut config = RunGovernanceConfig::default();
    config.max_workers = 4;
    let report = store.validate_runs(&config).expect("report");
    let expected = runs_per_thread * 4;

    assert_eq!(report.summary.total_runs, expected);
    assert_eq!(report.summary.valid_runs, expected);
    assert_eq!(report.summary.invalid_runs, 0);
}
