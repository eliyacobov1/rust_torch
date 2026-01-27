use rustorch::ops;
use rustorch::tensor::Tensor;
use rustorch::TorchError;

#[test]
fn add_returns_broadcast_error() {
    let a = Tensor::ones(&[2, 3], false);
    let b = Tensor::ones(&[4, 5], false);
    let err = ops::try_add(&a, &b).unwrap_err();
    assert!(matches!(err, TorchError::BroadcastMismatch { .. }));
}

#[test]
fn log_softmax_invalid_dim() {
    let x = Tensor::ones(&[2, 3], false);
    let err = ops::try_log_softmax(&x, 5).unwrap_err();
    assert!(matches!(err, TorchError::InvalidDim { .. }));
}

#[test]
fn dropout_invalid_probability() {
    let x = Tensor::ones(&[2, 2], false);
    let err = ops::try_dropout(&x, 1.0, true, None).unwrap_err();
    assert!(matches!(err, TorchError::InvalidArgument { .. }));
}
