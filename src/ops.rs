
use crate::tensor::Tensor;
use crate::autograd::{make_add_grad, make_matmul_grad};
use std::sync::Arc;

pub fn add(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let n = a.numel();
    let mut out_data = vec![0.0f32; n];
    for i in 0..n 
    {
        out_data[i] = a.storage.data[i] + b.storage.data[i];
    }
    let requires_grad = a.requires_grad || b.requires_grad;
    let grad_fn = if requires_grad { Some(make_add_grad(a.clone(), b.clone())) } else { None };
    let out = Tensor::new(out_data, &a.shape, grad_fn, requires_grad);
    out
}

pub fn matmul(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2);
    let mut out_data = vec![0.0f32; [m,n].iter().product()];
    for i in 0..m {
        for j in 0..n {
            let mut acc=0.0;
            for p in 0..k { acc += a.storage.data[i*k+p]*b.storage.data[p*n+j]; }
            out_data[i*n+j]=acc;
        }
    }
    let requires_grad = a.requires_grad || b.requires_grad;
    let grad_fn = if requires_grad { Some(make_matmul_grad(a.clone(), b.clone())) } else { None };
    let out = Tensor::new(out_data, &[m,n], grad_fn, requires_grad);
    out
}
