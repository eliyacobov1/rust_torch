use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::ExperimentStore;

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
