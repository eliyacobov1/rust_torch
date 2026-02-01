use rustorch::tensor::{Tensor, TensorLayout};
use rustorch::TorchError;

#[test]
fn layout_reports_contiguous_and_strided() {
    let contiguous = Tensor::ones(&[2, 2], false);
    assert_eq!(contiguous.layout(), TensorLayout::Contiguous);
    assert!(contiguous.is_contiguous());

    let data = vec![0.0; 5];
    let strided = Tensor::try_from_vec_f32_with_strides(data, &[2, 2], &[3, 1], None, false)
        .expect("strided tensor should be valid");
    assert_eq!(strided.layout(), TensorLayout::Strided);
    assert!(!strided.is_contiguous());
}

#[test]
fn validate_strided_layout_allows_non_contiguous() {
    let data = vec![0.0; 5];
    let strided = Tensor::try_from_vec_f32_with_strides(data, &[2, 2], &[3, 1], None, false)
        .expect("strided tensor should be valid");
    strided
        .validate_strided_layout("layout_semantics")
        .expect("strided layout should validate");

    let err = strided
        .validate_layout("layout_semantics")
        .err()
        .expect("contiguous validation should fail");
    assert!(matches!(err, TorchError::NonContiguous { .. }));
}
