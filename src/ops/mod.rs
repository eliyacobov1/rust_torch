pub mod kernels;

use crate::autograd::{
    make_add_grad, make_log_softmax_grad, make_matmul_grad, make_mse_loss_grad, make_mul_grad,
    make_nll_loss_grad, make_relu_grad, make_sum_grad,
};
use crate::tensor::Tensor;
use crate::tensor::TensorInner;
use kernels::{add_kernel, matmul_kernel, mul_kernel};
use std::sync::Arc;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = Tensor::broadcast_shape(a.shape(), b.shape());

    // Materialize broadcasted views (copy version you already wrote)
    let a_b = Tensor::broadcast_to(a, &shape);
    let b_b = Tensor::broadcast_to(b, &shape);
    let result_data = add_kernel(a_b.storage().data.as_slice(), b_b.storage().data.as_slice());

    let requires_grad = a.requires_grad() || b.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_add_grad(a, b, &shape))
    } else {
        None
    };
    let out = Tensor::new(result_data, &shape, grad_fn, requires_grad);

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

    let grad_fn = if requires_grad {
        Some(make_matmul_grad(a, b))
    } else {
        None
    };
    let mut out = Tensor::zeros(&[&batch_shape[..], &[m, n]].concat(), requires_grad);
    if requires_grad {
        out.set_grad_fn(grad_fn);
    }

    // Iterate over batches
    for ([a_mat, b_mat], [out_mat_buf]) in TensorInner::iter_batches(
        &[a.inner.as_ref(), b.inner.as_ref()],
        &mut [Arc::get_mut(&mut out.inner).unwrap()],
    ) {
        let result_data = matmul_kernel(
            a_mat.storage().data.as_slice(),
            b_mat.storage().data.as_slice(),
            m,
            k1,
            n,
        );
        out_mat_buf.copy_from_slice(&result_data);
    }

    out
}

pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    // Mean Squared Error: (1/N) * sum((predictions - targets)^2)
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Predictions and targets must have the same shape"
    );

    let requires_grad = predictions.requires_grad() || targets.requires_grad();
    let n = predictions.numel() as f32;

    // Compute loss directly for efficiency
    let mut loss = 0.0;
    for (pred, target) in predictions
        .storage()
        .data
        .iter()
        .zip(targets.storage().data.iter())
    {
        let diff = pred - target;
        loss += diff * diff;
    }
    loss /= n;

    let grad_fn = if requires_grad {
        Some(make_mse_loss_grad(predictions, targets, n))
    } else {
        None
    };

    Tensor::new(vec![loss], &[1], grad_fn, requires_grad)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = Tensor::broadcast_shape(a.shape(), b.shape());

    let a_b = Tensor::broadcast_to(a, &shape);
    let b_b = Tensor::broadcast_to(b, &shape);
    let result_data = mul_kernel(a_b.storage().data.as_slice(), b_b.storage().data.as_slice());

    let requires_grad = a.requires_grad() || b.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_mul_grad(a, b, &shape))
    } else {
        None
    };
    Tensor::new(result_data, &shape, grad_fn, requires_grad)
}

pub fn relu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.storage().data.iter().map(|&v| v.max(0.0)).collect();
    let requires_grad = x.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_relu_grad(x))
    } else {
        None
    };
    Tensor::new(data, x.shape(), grad_fn, requires_grad)
}

fn normalize_dim(dim: isize, rank: usize) -> usize {
    let mut dim = dim;
    if dim < 0 {
        dim += rank as isize;
    }
    if dim < 0 || dim as usize >= rank {
        panic!("log_softmax: dim {} out of range for rank {}", dim, rank);
    }
    dim as usize
}

pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    let shape = x.shape();
    let rank = shape.len();
    assert!(rank > 0, "log_softmax requires a non-empty shape");
    let dim = normalize_dim(dim, rank);
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let outer_size: usize = shape[..dim].iter().product();
    let mut out = vec![0.0f32; x.numel()];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                max_val = max_val.max(x.storage().data[idx]);
            }
            let mut sum_exp = 0.0f32;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                sum_exp += (x.storage().data[idx] - max_val).exp();
            }
            let logsumexp = max_val + sum_exp.ln();
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                out[idx] = x.storage().data[idx] - logsumexp;
            }
        }
    }

    let requires_grad = x.requires_grad();
    let mut output = Tensor::new(out.clone(), shape, None, requires_grad);
    if requires_grad {
        let grad_fn = Some(make_log_softmax_grad(x, out, dim, shape));
        output.set_grad_fn(grad_fn);
    }
    output
}

pub fn sum(x: &Tensor) -> Tensor {
    let total: f32 = x.storage().data.iter().sum();
    let requires_grad = x.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_sum_grad(x))
    } else {
        None
    };
    Tensor::new(vec![total], &[1], grad_fn, requires_grad)
}

pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    // x: [batch, in_features], w: [in_features, out_features], b: [out_features]
    let out = matmul(x, w);
    add(&out, b)
}

pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    let shape = log_probs.shape();
    assert_eq!(shape.len(), 2, "nll_loss expects [batch, classes] input");
    let batch = shape[0];
    let classes = shape[1];
    assert_eq!(targets.shape(), &[batch], "targets must match batch size");

    let mut loss = 0.0f32;
    for i in 0..batch {
        let target = targets.storage().data[i] as isize;
        assert!(
            target >= 0 && (target as usize) < classes,
            "target index out of range: {}",
            target
        );
        let idx = i * classes + target as usize;
        loss -= log_probs.storage().data[idx];
    }
    loss /= batch as f32;

    let requires_grad = log_probs.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_nll_loss_grad(log_probs, targets, batch, classes))
    } else {
        None
    };
    Tensor::new(vec![loss], &[1], grad_fn, requires_grad)
}
