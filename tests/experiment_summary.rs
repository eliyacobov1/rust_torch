use std::collections::BTreeMap;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, RunStatus};

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_summary_test_{suffix}"))
}

#[test]
fn write_summary_rolls_up_metrics() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run = store
        .create_run("summary", serde_json::json!({}), Vec::new())
        .expect("run");
    let mut steps = Vec::new();
    for (idx, loss) in [4.0_f32, 3.0, 2.0, 1.0].iter().enumerate() {
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), *loss);
        metrics.insert("accuracy".to_string(), 0.1 + idx as f32 * 0.1);
        run.log_metrics(idx + 1, metrics).expect("log metrics");
        steps.push(idx + 1);
    }

    run.mark_completed().expect("mark completed");
    let summary = run.write_summary(None).expect("summary");
    assert_eq!(summary.status, RunStatus::Completed);
    assert_eq!(summary.metrics.total_records, 4);
    assert_eq!(summary.metrics.first_step, Some(*steps.first().unwrap()));
    assert_eq!(summary.metrics.last_step, Some(*steps.last().unwrap()));

    let loss_stats = summary.metrics.metrics.get("loss").expect("loss stats");
    assert_eq!(loss_stats.count, 4);
    assert_eq!(loss_stats.min, 1.0);
    assert_eq!(loss_stats.max, 4.0);
    assert_eq!(loss_stats.mean, 2.5);
    assert_eq!(loss_stats.p50, 3.0);
    assert_eq!(loss_stats.p95, 4.0);
    assert_eq!(loss_stats.last, 1.0);

    let accuracy_stats = summary
        .metrics
        .metrics
        .get("accuracy")
        .expect("accuracy stats");
    assert_eq!(accuracy_stats.count, 4);
    assert!((accuracy_stats.mean - 0.25).abs() < f32::EPSILON);
    assert_eq!(accuracy_stats.p50, 0.3);
    assert_eq!(accuracy_stats.p95, 0.4);
    assert!(summary.telemetry.is_none());
}
