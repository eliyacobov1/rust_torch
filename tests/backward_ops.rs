mod common;

use rustorch::{ops, tensor::Tensor};

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
