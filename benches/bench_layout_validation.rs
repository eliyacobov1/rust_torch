use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rustorch::telemetry::{jsonl_recorder_from_env, TelemetryRecorder};
use rustorch::tensor::Tensor;

fn required_len(shape: &[usize], strides: &[usize]) -> usize {
    if shape.is_empty() {
        return 0;
    }
    let mut total = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        if dim == 0 {
            return 0;
        }
        total = total.saturating_add((dim.saturating_sub(1)).saturating_mul(stride));
    }
    total
}

fn padded_strides(shape: &[usize], padding: usize) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (idx, dim) in shape.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(*dim);
    }
    for stride in strides.iter_mut() {
        *stride = stride.saturating_add(padding);
    }
    strides
}

fn bench_layout_validation(c: &mut Criterion) {
    let telemetry = init_telemetry();
    let mut group = c.benchmark_group("layout_validation");
    for &rank in &[2usize, 4, 8, 12] {
        let shape = vec![2usize; rank];
        let strides = padded_strides(&shape, 1);
        let len = required_len(&shape, &strides);
        group.bench_with_input(BenchmarkId::from_parameter(rank), &len, |b, &len| {
            b.iter_batched(
                || vec![0.0f32; len],
                |data| {
                    let _timer = telemetry.as_ref().map(|recorder| {
                        recorder
                            .timer("bench_layout_validation")
                            .with_tag("rank", format!("{rank}"))
                    });
                    let _ =
                        Tensor::try_from_vec_f32_with_strides(data, &shape, &strides, None, false)
                            .expect("layout validation should succeed");
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_layout_validation);
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
