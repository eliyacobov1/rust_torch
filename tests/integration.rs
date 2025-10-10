mod common;

use rustorch::{ops, tensor::Tensor};

use common::{assert_approx_eq, clone_data, matmul, transpose};

#[test]
fn manual_backprop_pipeline_matches_manual_gradients() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    let b = Tensor::from_vec_f32(vec![0.5, -1.0, 1.5, 0.0], &[2, 2], None, true);
    let weight = Tensor::from_vec_f32(vec![2.0, -1.0, 0.0, 3.0], &[2, 2], None, true);

    let add_out = ops::add(&a, &b);
    let predictions = ops::matmul(&add_out, &weight);
    let targets = Tensor::from_vec_f32(vec![1.0, 0.5, -0.5, 2.0], &[2, 2], None, false);
    let loss = ops::mse_loss(&predictions, &targets);

    let preds_data = clone_data(&predictions);
    let targets_data = clone_data(&targets);
    let add_data = clone_data(&add_out);
    let weight_data = clone_data(&weight);

    let n = preds_data.len() as f32;
    let grad_pred: Vec<f32> = preds_data
        .iter()
        .zip(targets_data.iter())
        .map(|(p, t)| (2.0 / n) * (p - t))
        .collect();
    let weight_t = transpose(weight_data.as_slice(), 2, 2);
    let grad_add_expected = matmul(grad_pred.as_slice(), 2, 2, weight_t.as_slice(), 2, 2);
    let add_t = transpose(add_data.as_slice(), 2, 2);
    let grad_weight_expected = matmul(add_t.as_slice(), 2, 2, grad_pred.as_slice(), 2, 2);

    let grad_a_expected = grad_add_expected.clone();
    let grad_b_expected = grad_add_expected.clone();

    loss.grad_fn().expect("loss grad fn").backward(&[1.0]);
    let grad_pred_clone = predictions.grad().expect("pred grad");
    predictions
        .grad_fn()
        .expect("matmul grad fn")
        .backward(&grad_pred_clone.data);

    let grad_add_clone = add_out.grad().expect("add grad data");
    add_out
        .grad_fn()
        .expect("add grad fn")
        .backward(&grad_add_clone.data);

    let grad_weight = weight.grad().expect("weight grad");
    let grad_a = a.grad().expect("a grad");
    let grad_b = b.grad().expect("b grad");

    assert_approx_eq(&grad_pred_clone.data, grad_pred.as_slice(), 1e-5);
    assert_approx_eq(&grad_add_clone.data, grad_add_expected.as_slice(), 1e-5);
    assert_approx_eq(&grad_weight.data, grad_weight_expected.as_slice(), 1e-5);
    assert_approx_eq(&grad_a.data, grad_a_expected.as_slice(), 1e-5);
    assert_approx_eq(&grad_b.data, grad_b_expected.as_slice(), 1e-5);
}
