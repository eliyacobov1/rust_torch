use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rustorch::autograd;
use rustorch::data::{make_synthetic_classification, SyntheticClassificationConfig};
use rustorch::models::MlpClassifier;
use rustorch::ops;
use rustorch::optim::{Optimizer, Sgd};

fn bench_mlp_classifier_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_classifier_train_step");
    let features = 8usize;
    let classes = 4usize;
    let hidden = 16usize;
    for batch in [16usize, 32, 64] {
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, &batch| {
            b.iter_batched(
                || {
                    let config = SyntheticClassificationConfig {
                        samples: batch,
                        features,
                        classes,
                        cluster_std: 0.3,
                        seed: 13,
                    };
                    let data = make_synthetic_classification(&config).expect("classification data");
                    let model =
                        MlpClassifier::new(features, hidden, classes, 17).expect("classifier");
                    (data.dataset, model)
                },
                |(dataset, mut model)| {
                    let batch = dataset
                        .batch_iter(batch)
                        .expect("batch iter")
                        .next()
                        .expect("batch")
                        .expect("batch ok");
                    let log_probs = model.forward(&batch.features).expect("forward");
                    let loss = ops::nll_loss(&log_probs, &batch.targets);
                    let _stats = autograd::backward(&loss).expect("backward");
                    let mut optimizer = Sgd::new(0.05, 0.0).expect("optimizer");
                    let mut params = model.parameters_mut();
                    optimizer.step(&mut params).expect("step");
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mlp_classifier_train_step);
criterion_main!(benches);
