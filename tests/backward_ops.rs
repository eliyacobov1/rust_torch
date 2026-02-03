mod common;

use rustorch::{autograd, ops, tensor::Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::{assert_approx_eq, matmul, transpose};

#[test]
fn tensor_add_backward_propagates_upstream_gradient() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let b = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], None, true);

    let out = ops::add(&a, &b);

    let upstream = vec![0.1, 0.2, 0.3, 0.4];
    out.grad_fn().expect("add grad fn").backward(&upstream);

    let grad_a = a.grad().expect("gradient for a");
    let grad_b = b.grad().expect("gradient for b");
    assert_approx_eq(&grad_a.data, upstream.as_slice(), 1e-6);
    assert_approx_eq(&grad_b.data, upstream.as_slice(), 1e-6);
}

#[test]
fn tensor_mul_backward_matches_expected_gradients() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let b = Tensor::from_vec_f32(vec![2.0, 3.0, 4.0, 5.0], &[2, 2], None, true);

    let out = ops::mul(&a, &b);

    let upstream = vec![0.1, 0.2, 0.3, 0.4];
    out.grad_fn().expect("mul grad fn").backward(&upstream);

    let grad_a = a.grad().expect("gradient for a");
    let grad_b = b.grad().expect("gradient for b");

    let expected_grad_a: Vec<f32> = upstream
        .iter()
        .zip(b.storage().data.iter())
        .map(|(&u, &b_val)| u * b_val)
        .collect();
    let expected_grad_b: Vec<f32> = upstream
        .iter()
        .zip(a.storage().data.iter())
        .map(|(&u, &a_val)| u * a_val)
        .collect();

    assert_approx_eq(&grad_a.data, expected_grad_a.as_slice(), 1e-6);
    assert_approx_eq(&grad_b.data, expected_grad_b.as_slice(), 1e-6);
}

#[test]
fn tensor_mul_backward_reduces_broadcasted_grad() {
    let a = Tensor::from_vec_f32(vec![2.0, 3.0], &[2, 1], None, true);
    let b = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], None, true);

    let out = ops::mul(&a, &b);

    let upstream = vec![1.0; 6];
    out.grad_fn().expect("mul grad fn").backward(&upstream);

    let grad_a = a.grad().expect("gradient for a");
    let grad_b = b.grad().expect("gradient for b");

    // grad_a sums over the broadcasted axis: sum(b) per row
    let expected_grad_a = vec![1.0 + 2.0 + 3.0, 4.0 + 5.0 + 6.0];
    // grad_b equals upstream * a broadcasted
    let expected_grad_b = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    assert_approx_eq(&grad_a.data, expected_grad_a.as_slice(), 1e-6);
    assert_approx_eq(&grad_b.data, expected_grad_b.as_slice(), 1e-6);
}

#[test]
fn tensor_matmul_backward_matches_manual_gradient() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], None, true);
    let b = Tensor::from_vec_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], None, true);

    let out = ops::matmul(&a, &b);

    let upstream = vec![0.5, -1.0, 1.5, 2.0];
    out.grad_fn().expect("matmul grad fn").backward(&upstream);

    let grad_a_expected = matmul(
        &upstream,
        2,
        2,
        &transpose(b.storage().data.as_slice(), 3, 2),
        2,
        3,
    );
    let grad_b_expected = matmul(
        &transpose(a.storage().data.as_slice(), 2, 3),
        3,
        2,
        &upstream,
        2,
        2,
    );

    let grad_a = a.grad().expect("gradient for a");
    let grad_b = b.grad().expect("gradient for b");
    assert_approx_eq(&grad_a.data, grad_a_expected.as_slice(), 1e-5);
    assert_approx_eq(&grad_b.data, grad_b_expected.as_slice(), 1e-5);
}

#[test]
fn tensor_reshape_backward_preserves_gradient_layout() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let reshaped = a.reshape(&[4, 1]);

    let upstream = vec![0.4, 0.3, 0.2, 0.1];
    reshaped
        .grad_fn()
        .expect("reshape grad fn")
        .backward(&upstream);

    let grad_a = a.grad().expect("gradient for a");
    assert_approx_eq(&grad_a.data, &[0.4, 0.3, 0.2, 0.1], 1e-6);
}

#[test]
fn tensor_broadcast_backward_sums_along_broadcast_axes() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], &[2, 1], None, true);
    let broadcasted = a.broadcast_to(&[2, 3]);

    let upstream = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    broadcasted
        .grad_fn()
        .expect("broadcast grad fn")
        .backward(&upstream);

    let grad_a = a.grad().expect("gradient for a");
    assert_approx_eq(&grad_a.data, &[6.0, 15.0], 1e-5);
}

#[test]
fn mse_loss_backward_matches_expected_derivative() {
    let predictions = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let targets = Tensor::from_vec_f32(vec![0.5, 1.5, 2.5, 3.5], &[2, 2], None, true);

    let loss = ops::mse_loss(&predictions, &targets);
    loss.grad_fn().expect("mse loss grad fn").backward(&[1.0]);

    let grad_predictions = predictions.grad().expect("grad for predictions");
    let grad_targets = targets.grad().expect("grad for targets");

    let expected_grad: Vec<f32> = predictions
        .storage()
        .data
        .iter()
        .zip(targets.storage().data.iter())
        .map(|(p, t)| 0.5 * (p - t))
        .collect();
    let expected_target_grad: Vec<f32> = expected_grad.iter().map(|g| -g).collect();

    assert_approx_eq(&grad_predictions.data, expected_grad.as_slice(), 1e-6);
    assert_approx_eq(&grad_targets.data, expected_target_grad.as_slice(), 1e-6);
}

#[test]
fn relu_backward_masks_negative_inputs() {
    let x = Tensor::from_vec_f32(vec![-1.0, 0.0, 2.0, -3.0], &[2, 2], None, true);
    let out = ops::relu(&x);

    let upstream = vec![1.0, 2.0, 3.0, 4.0];
    out.grad_fn().expect("relu grad fn").backward(&upstream);

    let grad_x = x.grad().expect("gradient for x");
    assert_approx_eq(&grad_x.data, &[0.0, 0.0, 3.0, 0.0], 1e-6);
}

#[test]
fn linear_backward_propagates_to_weights_and_bias() {
    let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], None, true);
    let w = Tensor::from_vec_f32(vec![0.5, -1.0, 1.5, 2.0, -0.5, 0.25], &[3, 2], None, true);
    let b = Tensor::from_vec_f32(vec![0.1, -0.2], &[2], None, true);

    let out = ops::linear(&x, &w, &b);
    let targets = Tensor::from_vec_f32(vec![0.0, 1.0, -1.0, 2.0], &[2, 2], None, false);
    let loss = ops::mse_loss(&out, &targets);
    autograd::backward(&loss).expect("backward failed");

    let grad_out: Vec<f32> = out
        .storage()
        .data
        .iter()
        .zip(targets.storage().data.iter())
        .map(|(pred, target)| {
            let diff = pred - target;
            (2.0 / 4.0) * diff
        })
        .collect();

    let grad_x_expected = matmul(
        &grad_out,
        2,
        2,
        &transpose(w.storage().data.as_slice(), 3, 2),
        2,
        3,
    );
    let grad_w_expected = matmul(
        &transpose(x.storage().data.as_slice(), 2, 3),
        3,
        2,
        &grad_out,
        2,
        2,
    );
    let grad_b_expected = vec![grad_out[0] + grad_out[2], grad_out[1] + grad_out[3]];

    let grad_x = x.grad().expect("gradient for x");
    let grad_w = w.grad().expect("gradient for w");
    let grad_b = b.grad().expect("gradient for b");

    assert_approx_eq(&grad_x.data, grad_x_expected.as_slice(), 1e-6);
    assert_approx_eq(&grad_w.data, grad_w_expected.as_slice(), 1e-6);
    assert_approx_eq(&grad_b.data, grad_b_expected.as_slice(), 1e-6);
}

#[derive(Default)]
struct CountingObserver {
    batches: AtomicUsize,
    starts: AtomicUsize,
    ends: AtomicUsize,
}

impl autograd::BackwardObserver for CountingObserver {
    fn on_batch_start(&self, _batch_size: usize) {
        self.batches.fetch_add(1, Ordering::Relaxed);
    }

    fn on_node_start(&self, _node_id: usize) {
        self.starts.fetch_add(1, Ordering::Relaxed);
    }

    fn on_node_end(&self, _node_id: usize, _duration: Duration) {
        self.ends.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn backward_with_config_tracks_parallel_batches() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let b = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], None, true);
    let c = Tensor::from_vec_f32(vec![1.5, 2.5, 3.5, 4.5], &[2, 2], None, true);
    let d = Tensor::from_vec_f32(vec![0.5, 1.5, 2.5, 3.5], &[2, 2], None, true);

    let left = ops::add(&a, &b);
    let right = ops::add(&c, &d);
    let out = ops::mul(&left, &right);
    let targets = Tensor::from_vec_f32(vec![1.0, 1.0, 1.0, 1.0], &[2, 2], None, false);
    let loss = ops::mse_loss(&out, &targets);

    let observer = Arc::new(CountingObserver::default());
    let config = autograd::BackwardConfig::new(2).with_observer(observer.clone());
    let stats = autograd::backward_with_config(&loss, &config).expect("backward with config");

    assert!(stats.max_batch_size >= 2);
    assert!(stats.max_parallelism >= 2);
    assert_eq!(observer.starts.load(Ordering::Relaxed), stats.nodes);
    assert_eq!(observer.ends.load(Ordering::Relaxed), stats.nodes);
    assert!(observer.batches.load(Ordering::Relaxed) >= 1);
}

#[test]
fn backward_accumulates_gradients_for_shared_nodes() {
    let x = Tensor::from_vec_f32(vec![1.0, 2.0], &[2], None, true);
    let y = &x + &x;
    let loss = ops::sum(&y);

    autograd::backward(&loss).expect("backward failed");

    let grad_x = x.grad().expect("gradient for x");
    assert_approx_eq(&grad_x.data, &[2.0, 2.0], 1e-6);
}

#[test]
fn dropout_backward_propagates_masked_gradient() {
    let input = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let out = ops::dropout(&input, 0.5, true, Some(123));
    let upstream = vec![1.0; 4];
    out.grad_fn().expect("dropout grad fn").backward(&upstream);

    let grad_input = input.grad().expect("gradient for input");
    let expected: Vec<f32> = out
        .storage()
        .data
        .iter()
        .zip(input.storage().data.iter())
        .map(|(&out_val, &inp)| if out_val == 0.0 { 0.0 } else { out_val / inp })
        .collect();
    assert_approx_eq(&grad_input.data, expected.as_slice(), 1e-6);
}

#[test]
fn max_pool2d_backward_routes_gradient_to_max() {
    let input = Tensor::from_vec_f32(vec![1.0, 3.0, 2.0, 4.0], &[1, 1, 2, 2], None, true);
    let out = ops::max_pool2d(&input, 2, None, 0, 1, false);
    out.grad_fn().expect("max_pool2d grad fn").backward(&[1.0]);
    let grad_input = input.grad().expect("gradient for input");
    assert_approx_eq(&grad_input.data, &[0.0, 0.0, 0.0, 1.0], 1e-6);
}

#[test]
fn batch_norm_backward_matches_expected_gradients() {
    let input = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], None, true);
    let weight = Tensor::from_vec_f32(vec![1.0], &[1], None, true);
    let bias = Tensor::from_vec_f32(vec![0.0], &[1], None, true);

    let out = ops::batch_norm(
        &input,
        None,
        None,
        Some(&weight),
        Some(&bias),
        true,
        0.1,
        1e-5,
    );
    let upstream = vec![1.0, 2.0, 3.0, 4.0];
    out.grad_fn()
        .expect("batch_norm grad fn")
        .backward(&upstream);

    let grad_input = input.grad().expect("gradient for input");
    let grad_weight = weight.grad().expect("gradient for weight");
    let grad_bias = bias.grad().expect("gradient for bias");

    let expected_input = vec![-1.0732997e-05, -3.5776658e-06, 3.5776658e-06, 1.0732997e-05];
    assert_approx_eq(&grad_input.data, expected_input.as_slice(), 1e-6);
    assert_approx_eq(&grad_weight.data, &[4.472118], 1e-5);
    assert_approx_eq(&grad_bias.data, &[10.0], 1e-6);
}

#[test]
fn log_softmax_backward_matches_expected() {
    let values = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let out = ops::log_softmax(&values, 1);

    let upstream = vec![0.1, -0.2, 0.3, -0.4];
    out.grad_fn()
        .expect("log_softmax grad fn")
        .backward(&upstream);

    let mut expected = Vec::new();
    for (row_idx, row) in [[1.0f32, 2.0f32], [3.0f32, 4.0f32]].iter().enumerate() {
        let max = row[0].max(row[1]);
        let sum = (row[0] - max).exp() + (row[1] - max).exp();
        let logsum = max + sum.ln();
        let log_softmax_row = [row[0] - logsum, row[1] - logsum];
        let softmax_row = [log_softmax_row[0].exp(), log_softmax_row[1].exp()];
        let grad_row = [upstream[row_idx * 2], upstream[row_idx * 2 + 1]];
        let sum_grad = grad_row[0] + grad_row[1];
        expected.push(grad_row[0] - softmax_row[0] * sum_grad);
        expected.push(grad_row[1] - softmax_row[1] * sum_grad);
    }

    let grad_values = values.grad().expect("grad for values");
    assert_approx_eq(&grad_values.data, expected.as_slice(), 1e-6);
}

#[test]
fn nll_loss_backward_matches_expected() {
    let log_probs = Tensor::from_vec_f32(
        vec![-0.2, -1.4, -2.1, -1.9, -0.4, -0.7],
        &[2, 3],
        None,
        true,
    );
    let targets = Tensor::from_vec_f32(vec![1.0, 2.0], &[2], None, false);

    let loss = ops::nll_loss(&log_probs, &targets);
    loss.grad_fn().expect("nll_loss grad fn").backward(&[1.0]);

    let grad_log_probs = log_probs.grad().expect("grad for log_probs");
    let expected = vec![0.0, -0.5, 0.0, 0.0, 0.0, -0.5];
    assert_approx_eq(&grad_log_probs.data, expected.as_slice(), 1e-6);
}

#[test]
fn sum_backward_propagates_scalar_gradient() {
    let x = Tensor::from_vec_f32(vec![1.0, -2.0, 3.0, 4.0], &[2, 2], None, true);
    let out = ops::sum(&x);

    out.grad_fn().expect("sum grad fn").backward(&[0.5]);

    let grad_x = x.grad().expect("gradient for x");
    assert_approx_eq(&grad_x.data, &[0.5, 0.5, 0.5, 0.5], 1e-6);
}
