use criterion::{criterion_group, criterion_main, Criterion};
use rustorch::telemetry::{jsonl_recorder_from_env, TelemetryRecorder};

fn bench_add(c: &mut Criterion) {
    let telemetry = init_telemetry();
    // Placeholder: you can wire a Rust-only bench here.
    c.bench_function("add_1e6", |b| {
        b.iter(|| {
            let _timer = telemetry
                .as_ref()
                .map(|recorder| recorder.timer("bench_add").with_tag("size", "1e6"));
            let n = 1_000_000;
            let a: Vec<f32> = vec![1.0; n];
            let b_: Vec<f32> = vec![2.0; n];
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = a[i] + b_[i];
            }
        });
    });
}
criterion_group!(benches, bench_add);
criterion_main!(benches);

fn init_telemetry() -> Option<TelemetryRecorder<rustorch::telemetry::JsonlSink>> {
    match jsonl_recorder_from_env("RUSTORCH_BENCH_TELEMETRY") {
        Ok(recorder) => recorder,
        Err(err) => {
            eprintln!("telemetry disabled: {err:?}");
            None
        }
    }
}
