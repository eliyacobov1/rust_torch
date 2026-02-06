use std::collections::BTreeMap;
use std::sync::Arc;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{
    ExperimentStore, MetricAggregation, RunComparisonConfig, RunFilter, RunStatus,
};

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_compare_test_{suffix}"))
}

fn create_run(
    store: &ExperimentStore,
    name: &str,
    metric_rows: Vec<BTreeMap<String, f32>>,
    tags: Vec<String>,
) -> String {
    let mut run = store
        .create_run(name, serde_json::json!({}), tags)
        .expect("run");
    for (step, row) in metric_rows.into_iter().enumerate() {
        run.log_metrics(step + 1, row).expect("log metrics");
    }
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
    run.metadata().id.clone()
}

fn metric_row(loss: f32, accuracy: f32) -> BTreeMap<String, f32> {
    let mut row = BTreeMap::new();
    row.insert("loss".to_string(), loss);
    row.insert("accuracy".to_string(), accuracy);
    row
}

#[test]
fn compare_runs_reports_deltas() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let baseline_id = create_run(
        &store,
        "baseline",
        vec![metric_row(1.0, 0.8), metric_row(0.9, 0.82)],
        vec!["baseline".to_string()],
    );
    let candidate_id = create_run(
        &store,
        "candidate",
        vec![metric_row(0.8, 0.75), metric_row(0.7, 0.78)],
        vec!["candidate".to_string()],
    );

    let config = RunComparisonConfig {
        run_ids: vec![baseline_id.clone(), candidate_id.clone()],
        filter: RunFilter::default(),
        baseline_id: Some(baseline_id.clone()),
        metric_aggregation: MetricAggregation::Last,
        top_k: 2,
        build_graph: true,
        deterministic_seed: None,
        regression_gate: None,
    };
    let report = store.compare_runs(&config).expect("compare");
    assert_eq!(report.baseline_id, baseline_id);
    assert_eq!(report.comparisons.len(), 1);
    let comparison = &report.comparisons[0];
    assert_eq!(comparison.run_id, candidate_id);
    assert_eq!(comparison.status, RunStatus::Completed);
    assert_eq!(comparison.deltas.len(), 2);
    let loss_delta = comparison
        .deltas
        .iter()
        .find(|delta| delta.metric == "loss")
        .expect("loss delta");
    assert!(loss_delta.delta < 0.0);
    let accuracy_delta = comparison
        .deltas
        .iter()
        .find(|delta| delta.metric == "accuracy")
        .expect("accuracy delta");
    assert!(accuracy_delta.delta < 0.0);
    let graph = report.graph.expect("graph");
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 2);
}

#[test]
fn compare_runs_is_thread_safe_under_stress() {
    let store = Arc::new(ExperimentStore::new(temp_root()).expect("store"));
    let mut run_ids = Vec::new();
    for idx in 0..12 {
        let loss = 1.0 - idx as f32 * 0.02;
        let acc = 0.7 + idx as f32 * 0.01;
        let run_id = create_run(
            &store,
            &format!("run_{idx}"),
            vec![metric_row(loss, acc), metric_row(loss * 0.9, acc + 0.01)],
            vec!["stress".to_string()],
        );
        run_ids.push(run_id);
    }

    let config = RunComparisonConfig {
        run_ids: run_ids.clone(),
        filter: RunFilter::default(),
        baseline_id: Some(run_ids[0].clone()),
        metric_aggregation: MetricAggregation::Mean,
        top_k: 3,
        build_graph: false,
        deterministic_seed: Some(99),
        regression_gate: None,
    };

    let mut handles = Vec::new();
    for _ in 0..8 {
        let store = Arc::clone(&store);
        let config = config.clone();
        handles.push(std::thread::spawn(move || {
            let report = store.compare_runs(&config).expect("compare");
            assert_eq!(report.comparisons.len(), config.run_ids.len() - 1);
        }));
    }

    for handle in handles {
        handle.join().expect("thread join");
    }
}
