use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rustorch::{autograd, ops, tensor::Tensor};

fn build_graph(batch: usize) -> Tensor {
    let len = batch * batch;
    let a = Tensor::from_vec_f32(vec![1.0; len], &[batch, batch], None, true);
    let b = Tensor::from_vec_f32(vec![2.0; len], &[batch, batch], None, true);
    let c = Tensor::from_vec_f32(vec![3.0; len], &[batch, batch], None, true);
    let d = Tensor::from_vec_f32(vec![4.0; len], &[batch, batch], None, true);

    let left = ops::add(&a, &b);
    let right = ops::add(&c, &d);
    let out = ops::mul(&left, &right);
    let targets = Tensor::from_vec_f32(vec![1.0; len], &[batch, batch], None, false);
    ops::mse_loss(&out, &targets)
}

fn bench_autograd_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_backward");
    for batch in [4usize, 8, 16] {
        group.bench_with_input(BenchmarkId::new("serial", batch), &batch, |b, &batch| {
            b.iter_batched(
                || build_graph(batch),
                |loss| {
                    let _ = autograd::backward(&loss).expect("backward");
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("parallel_2", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    || build_graph(batch),
                    |loss| {
                        let config = autograd::BackwardConfig::new(2);
                        let _ = autograd::backward_with_config(&loss, &config).expect("backward");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_autograd_backward);
criterion_main!(benches);
