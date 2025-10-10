#![allow(dead_code)]

use rustorch::tensor::Tensor;

pub fn assert_approx_eq(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() <= tol,
            "Mismatch at index {}: actual={} expected={} with tol={}",
            i,
            a,
            e,
            tol
        );
    }
}

pub fn matmul(
    left: &[f32],
    left_rows: usize,
    left_cols: usize,
    right: &[f32],
    right_rows: usize,
    right_cols: usize,
) -> Vec<f32> {
    assert_eq!(left_cols, right_rows);
    let mut result = vec![0.0; left_rows * right_cols];
    for i in 0..left_rows {
        for j in 0..right_cols {
            let mut acc = 0.0;
            for k in 0..left_cols {
                acc += left[i * left_cols + k] * right[k * right_cols + j];
            }
            result[i * right_cols + j] = acc;
        }
    }
    result
}

pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

pub fn clone_data(tensor: &Tensor) -> Vec<f32> {
    tensor.storage().data.clone()
}
