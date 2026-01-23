mod common;

use rustorch::{ops, tensor::Tensor};

use common::assert_approx_eq;

#[test]
fn tensor_add_forward_matches_expected() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, false);
    let b = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], None, false);

    let out = ops::add(&a, &b);
    assert_eq!(out.shape(), &[2, 2]);
    assert_approx_eq(out.storage().data.as_slice(), &[6.0, 8.0, 10.0, 12.0], 1e-6);
}

#[test]
fn tensor_mul_forward_matches_expected() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, false);
    let b = Tensor::from_vec_f32(vec![2.0, 3.0, 4.0, 5.0], &[2, 2], None, false);

    let out = ops::mul(&a, &b);
    assert_eq!(out.shape(), &[2, 2]);
    assert_approx_eq(out.storage().data.as_slice(), &[2.0, 6.0, 12.0, 20.0], 1e-6);
}

#[test]
fn tensor_matmul_forward_matches_manual_result() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], None, false);
    let b = Tensor::from_vec_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], None, false);

    let out = ops::matmul(&a, &b);
    assert_eq!(out.shape(), &[2, 2]);
    assert_approx_eq(
        out.storage().data.as_slice(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-6,
    );
}

#[test]
fn tensor_reshape_forward_keeps_values() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, false);

    let reshaped = a.reshape(&[4, 1]);
    assert_eq!(reshaped.shape(), &[4, 1]);
    assert_approx_eq(reshaped.storage().data.as_slice(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

#[test]
fn tensor_broadcast_forward_repeats_elements() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], &[2, 1], None, false);

    let broadcasted = a.broadcast_to(&[2, 3]);
    assert_eq!(broadcasted.shape(), &[2, 3]);
    assert_approx_eq(
        broadcasted.storage().data.as_slice(),
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        1e-6,
    );
}

#[test]
fn mse_loss_forward_matches_expected_value() {
    let predictions = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, false);
    let targets = Tensor::from_vec_f32(vec![0.5, 1.5, 2.5, 3.5], &[2, 2], None, false);

    let loss = ops::mse_loss(&predictions, &targets);
    assert_eq!(loss.shape(), &[1]);
    assert_approx_eq(loss.storage().data.as_slice(), &[0.25], 1e-6);
}
