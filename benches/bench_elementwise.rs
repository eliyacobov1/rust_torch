use criterion::{criterion_group, criterion_main, Criterion};

fn bench_add(c: &mut Criterion) {
    // Placeholder: you can wire a Rust-only bench here.
    c.bench_function("add_1e6", |b| {
        b.iter(|| {
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
