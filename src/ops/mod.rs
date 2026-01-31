pub mod kernels;

use crate::autograd::{
    make_add_grad, make_batch_norm_grad, make_dropout_grad, make_log_softmax_grad,
    make_matmul_grad, make_max_pool2d_grad, make_mse_loss_grad, make_mul_grad, make_nll_loss_grad,
    make_relu_grad, make_sum_grad,
};
use crate::error::{Result, TorchError};
use crate::tensor::Tensor;
use crate::tensor::TensorInner;
use kernels::{add_kernel, matmul_kernel, mul_kernel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

fn validate_input_layouts(op: &'static str, tensors: &[&Tensor]) -> Result<()> {
    for tensor in tensors {
        tensor.validate_layout(op)?;
    }
    Ok(())
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    try_add(a, b).expect("add: invalid inputs")
}

pub fn try_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    validate_input_layouts("add", &[a, b])?;
    let shape = Tensor::try_broadcast_shape(a.shape(), b.shape(), "add")?;

    // Materialize broadcasted views (copy version you already wrote)
    let a_b = a.try_broadcast_to(&shape, "add")?;
    let b_b = b.try_broadcast_to(&shape, "add")?;
    let result_data = add_kernel(a_b.storage().data.as_slice(), b_b.storage().data.as_slice());

    let requires_grad = a.requires_grad() || b.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_add_grad(a, b, &shape))
    } else {
        None
    };
    let out = Tensor::new(result_data, &shape, grad_fn, requires_grad);

    Ok(out)
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    try_matmul(a, b).expect("matmul: invalid inputs")
}

pub fn try_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    validate_input_layouts("matmul", &[a, b])?;
    let requires_grad = a.requires_grad() || b.requires_grad();

    // Compute broadcasted batch shape
    if a.shape().len() < 2 || b.shape().len() < 2 {
        return Err(TorchError::InvalidArgument {
            op: "matmul",
            msg: "matmul expects tensors with at least 2 dimensions".to_string(),
        });
    }
    let a_batch = a.batch_dims();
    let b_batch = b.batch_dims();
    let batch_shape = Tensor::try_broadcast_shape(a_batch, b_batch, "matmul")?;
    let [m, k1] = a.matrix_dims();
    let [k2, n] = b.matrix_dims();
    if k1 != k2 {
        return Err(TorchError::InvalidArgument {
            op: "matmul",
            msg: format!(
                "inner dimensions must match (lhs={:?}, rhs={:?})",
                [m, k1],
                [k2, n]
            ),
        });
    }

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

    Ok(out)
}

pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    try_mse_loss(predictions, targets).expect("mse_loss: invalid inputs")
}

pub fn try_mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    validate_input_layouts("mse_loss", &[predictions, targets])?;
    // Mean Squared Error: (1/N) * sum((predictions - targets)^2)
    if predictions.shape() != targets.shape() {
        return Err(TorchError::InvalidArgument {
            op: "mse_loss",
            msg: format!(
                "predictions and targets must have the same shape (predictions={:?}, targets={:?})",
                predictions.shape(),
                targets.shape()
            ),
        });
    }

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

    Ok(Tensor::new(vec![loss], &[1], grad_fn, requires_grad))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    try_mul(a, b).expect("mul: invalid inputs")
}

pub fn try_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    validate_input_layouts("mul", &[a, b])?;
    let shape = Tensor::try_broadcast_shape(a.shape(), b.shape(), "mul")?;

    let a_b = a.try_broadcast_to(&shape, "mul")?;
    let b_b = b.try_broadcast_to(&shape, "mul")?;
    let result_data = mul_kernel(a_b.storage().data.as_slice(), b_b.storage().data.as_slice());

    let requires_grad = a.requires_grad() || b.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_mul_grad(a, b, &shape))
    } else {
        None
    };
    Ok(Tensor::new(result_data, &shape, grad_fn, requires_grad))
}

pub fn relu(x: &Tensor) -> Tensor {
    try_relu(x).expect("relu: invalid inputs")
}

pub fn try_relu(x: &Tensor) -> Result<Tensor> {
    validate_input_layouts("relu", &[x])?;
    let data: Vec<f32> = x.storage().data.iter().map(|&v| v.max(0.0)).collect();
    let requires_grad = x.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_relu_grad(x))
    } else {
        None
    };
    Ok(Tensor::new(data, x.shape(), grad_fn, requires_grad))
}

pub fn dropout(x: &Tensor, p: f32, training: bool, seed: Option<u64>) -> Tensor {
    try_dropout(x, p, training, seed).expect("dropout: invalid inputs")
}

pub fn try_dropout(x: &Tensor, p: f32, training: bool, seed: Option<u64>) -> Result<Tensor> {
    validate_input_layouts("dropout", &[x])?;
    if !(0.0..1.0).contains(&p) {
        return Err(TorchError::InvalidArgument {
            op: "dropout",
            msg: "dropout p must be in [0, 1)".to_string(),
        });
    }
    let requires_grad = x.requires_grad();
    if !training {
        let grad_fn = if requires_grad {
            Some(make_dropout_grad(x, vec![1.0; x.numel()], 1.0))
        } else {
            None
        };
        return Ok(Tensor::new(
            x.storage().data.clone(),
            x.shape(),
            grad_fn,
            requires_grad,
        ));
    }

    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    let scale = 1.0 / (1.0 - p);
    let mut mask = Vec::with_capacity(x.numel());
    let mut out = Vec::with_capacity(x.numel());
    for &val in x.storage().data.iter() {
        let keep = rng.gen::<f32>() >= p;
        let m = if keep { 1.0 } else { 0.0 };
        mask.push(m);
        out.push(val * m * scale);
    }

    let grad_fn = if requires_grad {
        Some(make_dropout_grad(x, mask, scale))
    } else {
        None
    };
    Ok(Tensor::new(out, x.shape(), grad_fn, requires_grad))
}

fn try_normalize_dim(dim: isize, rank: usize, op: &'static str) -> Result<usize> {
    let mut dim = dim;
    if dim < 0 {
        dim += rank as isize;
    }
    if dim < 0 || dim as usize >= rank {
        return Err(TorchError::InvalidDim { op, dim, rank });
    }
    Ok(dim as usize)
}

pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    try_log_softmax(x, dim).expect("log_softmax: invalid inputs")
}

pub fn try_log_softmax(x: &Tensor, dim: isize) -> Result<Tensor> {
    validate_input_layouts("log_softmax", &[x])?;
    let shape = x.shape();
    let rank = shape.len();
    if rank == 0 {
        return Err(TorchError::InvalidArgument {
            op: "log_softmax",
            msg: "log_softmax requires a non-empty shape".to_string(),
        });
    }
    let dim = try_normalize_dim(dim, rank, "log_softmax")?;
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
    Ok(output)
}

pub fn max_pool2d(
    x: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Tensor {
    try_max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
        .expect("max_pool2d: invalid inputs")
}

pub fn try_max_pool2d(
    x: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<Tensor> {
    validate_input_layouts("max_pool2d", &[x])?;
    if x.shape().len() != 4 {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d expects NCHW input".to_string(),
        });
    }
    if padding != 0 {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d padding not supported".to_string(),
        });
    }
    if dilation != 1 {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d dilation not supported".to_string(),
        });
    }
    if ceil_mode {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d ceil_mode not supported".to_string(),
        });
    }
    let stride = stride.unwrap_or(kernel_size);
    let n = x.shape()[0];
    let c = x.shape()[1];
    let h = x.shape()[2];
    let w = x.shape()[3];
    if kernel_size == 0 {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d kernel_size must be > 0".to_string(),
        });
    }
    if kernel_size > h || kernel_size > w {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d kernel_size exceeds input".to_string(),
        });
    }
    if stride == 0 {
        return Err(TorchError::InvalidArgument {
            op: "max_pool2d",
            msg: "max_pool2d stride must be > 0".to_string(),
        });
    }
    let out_h = (h - kernel_size) / stride + 1;
    let out_w = (w - kernel_size) / stride + 1;

    let mut out = vec![0.0f32; n * c * out_h * out_w];
    let mut indices = vec![0usize; out.len()];

    for batch in 0..n {
        for channel in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let h_start = oh * stride;
                    let w_start = ow * stride;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = h_start + kh;
                            let iw = w_start + kw;
                            let idx = ((batch * c + channel) * h + ih) * w + iw;
                            let val = x.storage().data[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = idx;
                            }
                        }
                    }
                    let out_idx = ((batch * c + channel) * out_h + oh) * out_w + ow;
                    out[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }

    let requires_grad = x.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_max_pool2d_grad(x, indices))
    } else {
        None
    };
    Ok(Tensor::new(
        out,
        &[n, c, out_h, out_w],
        grad_fn,
        requires_grad,
    ))
}

pub fn sum(x: &Tensor) -> Tensor {
    try_sum(x).expect("sum: invalid inputs")
}

pub fn try_sum(x: &Tensor) -> Result<Tensor> {
    validate_input_layouts("sum", &[x])?;
    let total: f32 = x.storage().data.iter().sum();
    let requires_grad = x.requires_grad();
    let grad_fn = if requires_grad {
        Some(make_sum_grad(x))
    } else {
        None
    };
    Ok(Tensor::new(vec![total], &[1], grad_fn, requires_grad))
}

pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    // x: [batch, in_features], w: [in_features, out_features], b: [out_features]
    let out = matmul(x, w);
    add(&out, b)
}

pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    try_nll_loss(log_probs, targets).expect("nll_loss: invalid inputs")
}

pub fn try_nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    validate_input_layouts("nll_loss", &[log_probs, targets])?;
    let shape = log_probs.shape();
    if shape.len() != 2 {
        return Err(TorchError::InvalidArgument {
            op: "nll_loss",
            msg: "nll_loss expects [batch, classes] input".to_string(),
        });
    }
    let batch = shape[0];
    let classes = shape[1];
    if targets.shape() != &[batch] {
        return Err(TorchError::InvalidArgument {
            op: "nll_loss",
            msg: "targets must match batch size".to_string(),
        });
    }

    let mut loss = 0.0f32;
    for i in 0..batch {
        let target = targets.storage().data[i] as isize;
        if target < 0 || (target as usize) >= classes {
            return Err(TorchError::InvalidArgument {
                op: "nll_loss",
                msg: format!("target index out of range: {}", target),
            });
        }
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
    Ok(Tensor::new(vec![loss], &[1], grad_fn, requires_grad))
}

pub fn batch_norm(
    x: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    _momentum: f32,
    eps: f32,
) -> Tensor {
    try_batch_norm(x, running_mean, running_var, weight, bias, training, eps)
        .expect("batch_norm: invalid inputs")
}

pub fn try_batch_norm(
    x: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    eps: f32,
) -> Result<Tensor> {
    validate_input_layouts("batch_norm", &[x])?;
    if x.shape().len() != 4 {
        return Err(TorchError::InvalidArgument {
            op: "batch_norm",
            msg: "batch_norm expects NCHW input".to_string(),
        });
    }
    let n = x.shape()[0];
    let c = x.shape()[1];
    let h = x.shape()[2];
    let w = x.shape()[3];
    let channel_size = (n * h * w) as f32;

    for (name, tensor) in [
        ("running_mean", running_mean),
        ("running_var", running_var),
        ("weight", weight),
        ("bias", bias),
    ] {
        if let Some(tensor) = tensor {
            validate_input_layouts("batch_norm", &[tensor])?;
            if tensor.shape() != &[c] {
                return Err(TorchError::InvalidArgument {
                    op: "batch_norm",
                    msg: format!("{name} must have shape [{c}]"),
                });
            }
        }
    }

    let mut mean = vec![0.0f32; c];
    let mut var = vec![0.0f32; c];

    if training || running_mean.is_none() || running_var.is_none() {
        for channel in 0..c {
            let mut sum = 0.0f32;
            for batch in 0..n {
                for idx in 0..(h * w) {
                    let offset = ((batch * c + channel) * h * w) + idx;
                    sum += x.storage().data[offset];
                }
            }
            mean[channel] = sum / channel_size;
        }

        for channel in 0..c {
            let mut sum = 0.0f32;
            for batch in 0..n {
                for idx in 0..(h * w) {
                    let offset = ((batch * c + channel) * h * w) + idx;
                    let diff = x.storage().data[offset] - mean[channel];
                    sum += diff * diff;
                }
            }
            var[channel] = sum / channel_size;
        }
    } else {
        if running_mean.is_none() || running_var.is_none() {
            return Err(TorchError::InvalidArgument {
                op: "batch_norm",
                msg: "running_mean and running_var required when training is false".to_string(),
            });
        }
        mean.copy_from_slice(running_mean.unwrap().storage().data.as_slice());
        var.copy_from_slice(running_var.unwrap().storage().data.as_slice());
    }

    let mut inv_std = vec![0.0f32; c];
    for channel in 0..c {
        inv_std[channel] = 1.0 / (var[channel] + eps).sqrt();
    }

    let mut out = vec![0.0f32; x.numel()];
    for batch in 0..n {
        for channel in 0..c {
            let weight_val = weight.map(|w| w.storage().data[channel]).unwrap_or(1.0);
            let bias_val = bias.map(|b| b.storage().data[channel]).unwrap_or(0.0);
            for idx in 0..(h * w) {
                let offset = ((batch * c + channel) * h * w) + idx;
                let normalized = (x.storage().data[offset] - mean[channel]) * inv_std[channel];
                out[offset] = normalized * weight_val + bias_val;
            }
        }
    }

    let requires_grad = x.requires_grad()
        || weight.map(|w| w.requires_grad()).unwrap_or(false)
        || bias.map(|b| b.requires_grad()).unwrap_or(false);
    let grad_fn = if requires_grad {
        Some(make_batch_norm_grad(x, weight, bias, mean, inv_std))
    } else {
        None
    };
    Ok(Tensor::new(out, x.shape(), grad_fn, requires_grad))
}
