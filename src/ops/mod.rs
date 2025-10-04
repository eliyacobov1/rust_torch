
pub mod kernels;

use std::sync::{Arc};
use kernels::matmul_kernel;
use crate::tensor::Tensor;
use crate::tensor::TensorInner;
use crate::autograd::{make_add_grad, make_matmul_grad};

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape());
    let n = a.numel();
    let mut out_data = vec![0.0f32; n];
    for i in 0..n 
    {
        out_data[i] = a.storage().data[i] + b.storage().data[i];
    }
    let requires_grad = a.requires_grad() || b.requires_grad();
    let grad_fn = if requires_grad { Some(make_add_grad(a, b)) } else { None };
    let out = Tensor::new(out_data, &a.shape(), grad_fn, requires_grad);
    out
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let requires_grad = a.requires_grad() || b.requires_grad();

    // Compute broadcasted batch shape
    let a_batch = a.batch_dims();
    let b_batch = b.batch_dims();
    let batch_shape = Tensor::broadcast_shape(a_batch, b_batch);
    let [m, k1] = a.matrix_dims();
    let [k2, n] = b.matrix_dims();
    assert_eq!(k1, k2);

    let grad_fn = if requires_grad { Some(make_matmul_grad(a, b)) } else { None };
    let mut out = Tensor::zeros(&[&batch_shape[..], &[m, n]].concat(), requires_grad);
    if requires_grad {
        out.set_grad_fn(grad_fn);
    }

    // Iterate over batches
    for ([a_mat, b_mat], [out_mat_buf]) in TensorInner::iter_batches(
        &[a.inner.as_ref(), b.inner.as_ref()],
        &mut [Arc::get_mut(&mut out.inner).unwrap()],
    ) {
        let result_data = matmul_kernel(a_mat.storage().data.as_slice(), b_mat.storage().data.as_slice(), m, k1, n);
        out_mat_buf.copy_from_slice(&result_data);
    }

    out
}
