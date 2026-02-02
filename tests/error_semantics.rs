use rustorch::ops;
use rustorch::tensor::Tensor;
use rustorch::TorchError;

#[test]
fn add_returns_broadcast_error() {
    let a = Tensor::ones(&[2, 3], false);
    let b = Tensor::ones(&[4, 5], false);
    let err = ops::try_add(&a, &b).err().expect("expected error");
    assert!(matches!(err, TorchError::BroadcastMismatch { .. }));
}

#[test]
fn log_softmax_invalid_dim() {
    let x = Tensor::ones(&[2, 3], false);
    let err = ops::try_log_softmax(&x, 5).err().expect("expected error");
    assert!(matches!(err, TorchError::InvalidDim { .. }));
}

#[test]
fn dropout_invalid_probability() {
    let x = Tensor::ones(&[2, 2], false);
    let err = ops::try_dropout(&x, 1.0, true, None)
        .err()
        .expect("expected error");
    assert!(matches!(err, TorchError::InvalidArgument { .. }));
}

#[test]
fn non_contiguous_layout_errors() {
    let data = vec![1.0; 6];
    let tensor = Tensor::try_from_vec_f32_with_strides(data, &[2, 3], &[1, 2], None, false)
        .expect("tensor creation should succeed");
    let err = ops::try_relu(&tensor).err().expect("expected error");
    assert!(matches!(err, TorchError::NonContiguous { .. }));
}

#[test]
fn invalid_layout_errors() {
    let data = vec![1.0; 4];
    let err = Tensor::try_from_vec_f32_with_strides(data, &[2, 3], &[3, 1], None, false)
        .err()
        .expect("expected error");
    assert!(matches!(err, TorchError::InvalidLayout { .. }));
}

#[test]
fn overlapping_layout_errors() {
    let data = vec![1.0; 3];
    let err = Tensor::try_from_vec_f32_with_strides(data, &[2, 2], &[1, 1], None, false)
        .err()
        .expect("expected error");
    assert!(matches!(err, TorchError::OverlappingLayout { .. }));
}

#[test]
fn contiguous_layout_validates_storage_len() {
    let data = vec![1.0; 5];
    let err = Tensor::try_from_vec_f32(data, &[2, 3], None, false)
        .err()
        .expect("expected error");
    assert!(matches!(err, TorchError::InvalidLayout { .. }));
}

#[test]
fn non_contiguous_layout_requires_correct_storage_len() {
    let data = vec![0.0; 5];
    let tensor = Tensor::try_from_vec_f32_with_strides(data, &[2, 2], &[3, 1], None, false)
        .expect("tensor creation should succeed");
    let err = ops::try_relu(&tensor).err().expect("expected error");
    assert!(matches!(err, TorchError::NonContiguous { .. }));
}

#[test]
fn zero_sized_layout_requires_empty_storage() {
    let data = vec![1.0];
    let err = Tensor::try_from_vec_f32_with_strides(data, &[0, 2], &[2, 1], None, false)
        .err()
        .expect("expected error");
    assert!(matches!(err, TorchError::InvalidLayout { .. }));
}
