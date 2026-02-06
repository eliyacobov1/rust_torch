use std::collections::BTreeMap;
use std::sync::{Arc, Barrier};

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{
    ExperimentStore, MetricAggregation, RegressionGateConfig, RegressionPolicy,
    RegressionSeverity, RunComparisonConfig, RunFilter,
};
use rustorch::deterministic_run_order;

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_graph_test_{suffix}"))
}

fn create_run(
    store: &ExperimentStore,
    name: &str,
    metric_rows: Vec<BTreeMap<String, f32>>,
) -> String {
    let mut run = store
        .create_run(name, serde_json::json!({}), Vec::new())
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
fn regression_gate_flags_blocker_findings() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let baseline_id = create_run(
        &store,
        "baseline",
        vec![metric_row(0.5, 0.9), metric_row(0.45, 0.92)],
    );
    let candidate_id = create_run(
        &store,
        "candidate",
        vec![metric_row(0.6, 0.85), metric_row(0.62, 0.83)],
    );

    let gate = RegressionGateConfig {
        policies: vec![RegressionPolicy {
            metric: "accuracy".to_string(),
            max_regression_pct: Some(1.0),
            max_regression_abs: None,
            severity: RegressionSeverity::Blocker,
        }],
        allow_missing_metrics: false,
        warn_only: false,
    };

    let config = RunComparisonConfig {
        run_ids: vec![baseline_id.clone(), candidate_id.clone()],
        filter: RunFilter::default(),
        baseline_id: Some(baseline_id.clone()),
        metric_aggregation: MetricAggregation::Last,
        top_k: 2,
        build_graph: false,
        deterministic_seed: Some(11),
        regression_gate: Some(gate),
    };

    let report = store.compare_runs(&config).expect("compare");
    let comparison = &report.comparisons[0];
    let gate_report = comparison.regression_gate.as_ref().expect("gate report");
    assert_eq!(gate_report.baseline_id, baseline_id);
    assert_eq!(gate_report.candidate_id, candidate_id);
    assert_eq!(gate_report.status, rustorch::experiment::RegressionGateStatus::Fail);
    assert_eq!(gate_report.findings.len(), 1);
}

#[test]
fn deterministic_seed_orders_graph_nodes() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run_ids = Vec::new();
    for idx in 0..4 {
        let run_id = create_run(
            &store,
            &format!("run_{idx}"),
            vec![metric_row(1.0 - idx as f32 * 0.1, 0.8 + idx as f32 * 0.01)],
        );
        run_ids.push(run_id);
    }
    let seed = 42;
    let config = RunComparisonConfig {
        run_ids: run_ids.clone(),
        filter: RunFilter::default(),
        baseline_id: None,
        metric_aggregation: MetricAggregation::Last,
        top_k: 1,
        build_graph: true,
        deterministic_seed: Some(seed),
        regression_gate: None,
    };
    let report = store.compare_runs(&config).expect("compare");
    let graph = report.graph.expect("graph");
    let expected = deterministic_run_order(seed, &run_ids);
    assert_eq!(graph.nodes, expected);
}

#[test]
fn compare_runs_is_concurrent_and_stable() {
    let store = Arc::new(ExperimentStore::new(temp_root()).expect("store"));
    let mut run_ids = Vec::new();
    for idx in 0..8 {
        let run_id = create_run(
            &store,
            &format!("run_{idx}"),
            vec![metric_row(1.0 - idx as f32 * 0.02, 0.7 + idx as f32 * 0.01)],
        );
        run_ids.push(run_id);
    }

    let gate = RegressionGateConfig {
        policies: vec![RegressionPolicy {
            metric: "loss".to_string(),
            max_regression_pct: Some(5.0),
            max_regression_abs: Some(0.1),
            severity: RegressionSeverity::Major,
        }],
        allow_missing_metrics: false,
        warn_only: true,
    };

    let config = RunComparisonConfig {
        run_ids: run_ids.clone(),
        filter: RunFilter::default(),
        baseline_id: Some(run_ids[0].clone()),
        metric_aggregation: MetricAggregation::Mean,
        top_k: 2,
        build_graph: true,
        deterministic_seed: Some(7),
        regression_gate: Some(gate),
    };

    let barrier = Arc::new(Barrier::new(5));
    let mut handles = Vec::new();
    for _ in 0..5 {
        let store = Arc::clone(&store);
        let config = config.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let report = store.compare_runs(&config).expect("compare");
            assert_eq!(report.comparisons.len(), config.run_ids.len() - 1);
            report
        }));
    }
    let mut reports = Vec::new();
    for handle in handles {
        reports.push(handle.join().expect("thread join"));
    }

    let first = &reports[0].comparisons;
    for report in &reports[1..] {
        assert_eq!(report.comparisons.len(), first.len());
        assert_eq!(report.graph.as_ref().unwrap().nodes, reports[0].graph.as_ref().unwrap().nodes);
    }
}
