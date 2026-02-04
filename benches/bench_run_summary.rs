use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, RunFilter};

fn bench_run_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_summary");
    group.bench_function("rollup_1000", |b| {
        b.iter_batched(
            || create_run_with_metrics(1000),
            |run| {
                run.write_summary(Some(Duration::from_millis(250)))
                    .expect("summary");
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("rollup_100000", |b| {
        b.iter_batched(
            || create_run_with_metrics(100000),
            |run| {
                run.write_summary(Some(Duration::from_millis(250)))
                    .expect("summary");
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("export_csv_100_runs", |b| {
        b.iter_batched(
            || create_store_with_runs(100),
            |store| {
                let output = store.root().join("runs_summary.csv");
                let filter = RunFilter::default();
                store
                    .export_run_summaries_csv(&output, &filter)
                    .expect("export");
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("read_summary_validation", |b| {
        b.iter_batched(
            || create_store_with_summary(),
            |(store, run_id)| {
                store.read_summary(&run_id).expect("read summary");
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn create_run_with_metrics(steps: usize) -> rustorch::experiment::RunHandle {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let run = store
        .create_run("summary_bench", serde_json::json!({}), Vec::new())
        .expect("run");
    for step in 0..steps {
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), (steps - step) as f32 * 0.001);
        metrics.insert("throughput".to_string(), 512.0);
        run.log_metrics(step, metrics).expect("log metrics");
    }
    run
}

fn create_store_with_runs(count: usize) -> ExperimentStore {
    let store = ExperimentStore::new(temp_root()).expect("store");
    for idx in 0..count {
        let mut run = store
            .create_run(
                &format!("run_{idx}"),
                serde_json::json!({ "idx": idx }),
                vec!["bench".to_string()],
            )
            .expect("run");
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), 1.0 / (idx as f32 + 1.0));
        run.log_metrics(idx, metrics).expect("log metrics");
        run.mark_completed().expect("mark completed");
        run.write_summary(Some(Duration::from_millis(10)))
            .expect("summary");
    }
    store
}

fn create_store_with_summary() -> (ExperimentStore, String) {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run = store
        .create_run("summary_read", serde_json::json!({}), Vec::new())
        .expect("run");
    let mut metrics = BTreeMap::new();
    metrics.insert("loss".to_string(), 0.125);
    run.log_metrics(1, metrics).expect("log metrics");
    run.mark_completed().expect("mark completed");
    run.write_summary(Some(Duration::from_millis(5)))
        .expect("summary");
    let run_id = run.metadata().id.clone();
    (store, run_id)
}

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_bench_summary_{suffix}"))
}

criterion_group!(benches, bench_run_summary);
criterion_main!(benches);
