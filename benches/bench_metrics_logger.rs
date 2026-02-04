use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, MetricsLoggerConfig};
use rustorch::telemetry::jsonl_recorder_from_env;

fn bench_metrics_logger(c: &mut Criterion) {
    let telemetry = init_telemetry();
    let mut group = c.benchmark_group("metrics_logger");
    for &batch_size in &[1usize, 8, 32, 128] {
        let config = MetricsLoggerConfig {
            batch_size,
            flush_interval_ms: 50,
            max_queue: 2048,
        };
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &config,
            |b, config| {
                b.iter_batched(
                    || create_logger(config.clone(), telemetry.clone()),
                    |logger| {
                        for step in 0..500usize {
                            let mut metrics = BTreeMap::new();
                            metrics.insert("loss".to_string(), step as f32 * 0.001);
                            metrics.insert("throughput".to_string(), 256.0);
                            logger.log_metrics(step, metrics).expect("log metrics");
                        }
                        logger.flush().expect("flush");
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn create_logger(
    config: MetricsLoggerConfig,
    telemetry: Option<
        std::sync::Arc<rustorch::telemetry::TelemetryRecorder<rustorch::telemetry::JsonlSink>>,
    >,
) -> rustorch::experiment::MetricsLogger {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let run = store
        .create_run("bench", serde_json::json!({}), Vec::new())
        .expect("run");
    run.start_metrics_logger(config, telemetry).expect("logger")
}

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_bench_{suffix}"))
}

criterion_group!(benches, bench_metrics_logger);
criterion_main!(benches);

fn init_telemetry(
) -> Option<std::sync::Arc<rustorch::telemetry::TelemetryRecorder<rustorch::telemetry::JsonlSink>>>
{
    match jsonl_recorder_from_env("RUSTORCH_BENCH_TELEMETRY") {
        Ok(recorder) => recorder.map(std::sync::Arc::new),
        Err(err) => {
            eprintln!("telemetry disabled: {err:?}");
            None
        }
    }
}
