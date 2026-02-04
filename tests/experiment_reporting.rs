use std::collections::BTreeMap;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, RunFilter, RunStatus};

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_reporting_test_{suffix}"))
}

#[test]
fn list_overviews_filters_and_exports_csv() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run = store
        .create_run(
            "alpha_run",
            serde_json::json!({}),
            vec!["alpha".to_string(), "baseline".to_string()],
        )
        .expect("run");
    let mut metrics = BTreeMap::new();
    metrics.insert("loss".to_string(), 1.25);
    metrics.insert("accuracy".to_string(), 0.75);
    run.log_metrics(1, metrics).expect("log metrics");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");

    let mut run_beta = store
        .create_run("beta_run", serde_json::json!({}), vec!["beta".to_string()])
        .expect("run");
    run_beta.mark_completed().expect("mark completed");
    run_beta.write_summary(None).expect("summary");

    let filter = RunFilter {
        tags: vec!["alpha".to_string()],
        statuses: Some(vec![RunStatus::Completed]),
    };
    let overviews = store.list_overviews(&filter).expect("overviews");
    assert_eq!(overviews.len(), 1);
    assert_eq!(overviews[0].metadata.name, "alpha_run");

    let output = store.root().join("runs_summary.csv");
    let report = store
        .export_run_summaries_csv(&output, &filter)
        .expect("export");
    assert_eq!(report.rows, 1);
    assert!(report.bytes_written > 0);
    let csv = std::fs::read_to_string(output).expect("read csv");
    assert!(csv.contains("metric.loss.mean"));
    assert!(csv.contains(&run.metadata().id));
}

#[test]
fn metrics_summary_uses_streaming_quantiles() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run = store
        .create_run("quantile_run", serde_json::json!({}), Vec::new())
        .expect("run");
    for step in 1..=100 {
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), step as f32);
        run.log_metrics(step, metrics).expect("log metrics");
    }
    run.mark_completed().expect("mark completed");
    let summary = run.write_summary(None).expect("summary");
    let stats = summary.metrics.metrics.get("loss").expect("loss stats");
    assert!((stats.p50 - 50.0).abs() <= 2.0);
    assert!((stats.p95 - 95.0).abs() <= 5.0);
    assert_eq!(stats.count, 100);
}
