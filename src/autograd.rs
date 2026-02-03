use crate::tensor::GradFnRef;
use crate::tensor::Tensor;
use crate::{error::Result, TorchError};
use log::info;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Grad {
    pub data: Vec<f32>,
}
impl Grad {
    pub fn zeros_like_shape(shape: &[usize]) -> Self {
        Self {
            data: vec![0.0; shape.iter().product()],
        }
    }
}
pub trait GradFn {
    fn backward(&self, grad_out: &[f32]);
    fn parents(&self) -> Vec<&Tensor> {
        vec![]
    }
}

struct MatMulGrad {
    a: Tensor,
    b: Tensor,
}
impl GradFn for MatMulGrad {
    fn backward(&self, grad_out: &[f32]) {
        let grad_out_tensor = Tensor::new(
            grad_out.to_vec(),
            &[self.a.shape()[0], self.b.shape()[1]],
            None,
            false,
        );
        // Shapes: a: (m, k), b: (k, n), grad_out: (m, n)

        // Compute grad for a: grad_a = grad_out.matmul(b.T)
        if let Some(grad_a) = &self.a.grad_lock() {
            let mut grad_a_lock = grad_a.lock().unwrap();
            grad_a_lock.data = (&grad_out_tensor * &self.b.transpose()).storage().data;
            // TODO: handle unnessecary copy here
        }

        // Compute grad for b: grad_b = a.T.matmul(grad_out)
        if let Some(grad_b) = &self.b.grad_lock() {
            let mut grad_b_lock = grad_b.lock().unwrap();
            grad_b_lock.data = (&self.a.transpose() * &grad_out_tensor).storage().data;
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}
pub fn make_matmul_grad(a: &Tensor, b: &Tensor) -> GradFnRef {
    Arc::new(MatMulGrad {
        a: a.clone(),
        b: b.clone(),
    })
}

struct AddGrad {
    a: Tensor,
    b: Tensor,
    output_shape: Vec<usize>,
}
impl GradFn for AddGrad {
    fn backward(&self, grad_out: &[f32]) {
        let expected_len: usize = self.output_shape.iter().product();
        assert_eq!(
            grad_out.len(),
            expected_len,
            "add backward: grad_out len {} != output len {}",
            grad_out.len(),
            expected_len
        );
        let grad_out_tensor =
            Tensor::from_vec_f32(grad_out.to_vec(), &self.output_shape, None, false);

        if let Some(g) = &self.a.grad_lock() {
            let grad_input = if self.a.shape() == self.output_shape.as_slice() {
                grad_out_tensor.clone()
            } else {
                sum_to_shape(&grad_out_tensor, self.a.shape())
            };
            for (gi, &u) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(grad_input.storage().data.iter())
            {
                *gi += u;
            }
        }
        if let Some(g) = &self.b.grad_lock() {
            let grad_input = if self.b.shape() == self.output_shape.as_slice() {
                grad_out_tensor.clone()
            } else {
                sum_to_shape(&grad_out_tensor, self.b.shape())
            };
            for (gi, &u) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(grad_input.storage().data.iter())
            {
                *gi += u;
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}
pub fn make_add_grad(a: &Tensor, b: &Tensor, output_shape: &[usize]) -> GradFnRef {
    Arc::new(AddGrad {
        a: a.clone(),
        b: b.clone(),
        output_shape: output_shape.to_vec(),
    })
}

struct MulGrad {
    a: Tensor,
    b: Tensor,
    output_shape: Vec<usize>,
}
impl GradFn for MulGrad {
    fn backward(&self, grad_out: &[f32]) {
        let expected_len: usize = self.output_shape.iter().product();
        assert_eq!(
            grad_out.len(),
            expected_len,
            "mul backward: grad_out len {} != output len {}",
            grad_out.len(),
            expected_len
        );

        let a_b = Tensor::broadcast_to(&self.a, &self.output_shape);
        let b_b = Tensor::broadcast_to(&self.b, &self.output_shape);

        if let Some(g) = &self.a.grad_lock() {
            let grad_full: Vec<f32> = grad_out
                .iter()
                .zip(b_b.storage().data.iter())
                .map(|(&u, &b_val)| u * b_val)
                .collect();
            let grad_full_tensor = Tensor::from_vec_f32(grad_full, &self.output_shape, None, false);
            let grad_input = if self.a.shape() == self.output_shape.as_slice() {
                grad_full_tensor
            } else {
                sum_to_shape(&grad_full_tensor, self.a.shape())
            };
            for (gi, &u) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(grad_input.storage().data.iter())
            {
                *gi += u;
            }
        }
        if let Some(g) = &self.b.grad_lock() {
            let grad_full: Vec<f32> = grad_out
                .iter()
                .zip(a_b.storage().data.iter())
                .map(|(&u, &a_val)| u * a_val)
                .collect();
            let grad_full_tensor = Tensor::from_vec_f32(grad_full, &self.output_shape, None, false);
            let grad_input = if self.b.shape() == self.output_shape.as_slice() {
                grad_full_tensor
            } else {
                sum_to_shape(&grad_full_tensor, self.b.shape())
            };
            for (gi, &u) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(grad_input.storage().data.iter())
            {
                *gi += u;
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}
pub fn make_mul_grad(a: &Tensor, b: &Tensor, output_shape: &[usize]) -> GradFnRef {
    Arc::new(MulGrad {
        a: a.clone(),
        b: b.clone(),
        output_shape: output_shape.to_vec(),
    })
}

struct DropoutGrad {
    input: Tensor,
    mask: Vec<f32>,
    scale: f32,
}

impl GradFn for DropoutGrad {
    fn backward(&self, grad_out: &[f32]) {
        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, (&go, &mask)) in grad
                .data
                .iter_mut()
                .zip(grad_out.iter().zip(self.mask.iter()))
            {
                *gi += go * mask * self.scale;
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_dropout_grad(input: &Tensor, mask: Vec<f32>, scale: f32) -> GradFnRef {
    Arc::new(DropoutGrad {
        input: input.clone(),
        mask,
        scale,
    })
}

struct MaxPool2dGrad {
    input: Tensor,
    indices: Vec<usize>,
}

impl GradFn for MaxPool2dGrad {
    fn backward(&self, grad_out: &[f32]) {
        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (&go, &idx) in grad_out.iter().zip(self.indices.iter()) {
                grad.data[idx] += go;
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_max_pool2d_grad(input: &Tensor, indices: Vec<usize>) -> GradFnRef {
    Arc::new(MaxPool2dGrad {
        input: input.clone(),
        indices,
    })
}

struct BatchNormGrad {
    input: Tensor,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

impl GradFn for BatchNormGrad {
    fn backward(&self, grad_out: &[f32]) {
        let shape = self.input.shape();
        assert_eq!(shape.len(), 4, "batch_norm expects NCHW input");
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let hw = h * w;
        let channel_size = n * hw;

        let mut grad_input = vec![0.0f32; self.input.numel()];
        let mut grad_weight = vec![0.0f32; c];
        let mut grad_bias = vec![0.0f32; c];

        for channel in 0..c {
            let mean = self.mean[channel];
            let inv_std = self.inv_std[channel];
            let mut sum_dxhat = 0.0f32;
            let mut sum_dxhat_xmu = 0.0f32;
            let mut sum_xmu = 0.0f32;

            for batch in 0..n {
                for idx in 0..hw {
                    let offset = ((batch * c + channel) * hw) + idx;
                    let x = self.input.storage().data[offset];
                    let x_mu = x - mean;
                    let mut go = grad_out[offset];
                    if let Some(weight) = &self.weight {
                        go *= weight.storage().data[channel];
                    }
                    sum_dxhat += go;
                    sum_dxhat_xmu += go * x_mu;
                    sum_xmu += x_mu;
                }
            }

            let dvar = sum_dxhat_xmu * -0.5 * inv_std.powi(3);
            let dmean = sum_dxhat * -inv_std + dvar * (-2.0 * sum_xmu / channel_size as f32);
            for batch in 0..n {
                for idx in 0..hw {
                    let offset = ((batch * c + channel) * hw) + idx;
                    let x = self.input.storage().data[offset];
                    let x_mu = x - mean;
                    let mut go = grad_out[offset];
                    if let Some(weight) = &self.weight {
                        go *= weight.storage().data[channel];
                    }
                    let dx = go * inv_std
                        + dvar * 2.0 * x_mu / channel_size as f32
                        + dmean / channel_size as f32;
                    grad_input[offset] = dx;
                }
            }

            let mut sum_grad = 0.0f32;
            let mut sum_grad_xhat = 0.0f32;
            for batch in 0..n {
                for idx in 0..hw {
                    let offset = ((batch * c + channel) * hw) + idx;
                    let x = self.input.storage().data[offset];
                    let x_hat = (x - mean) * inv_std;
                    let go = grad_out[offset];
                    sum_grad += go;
                    sum_grad_xhat += go * x_hat;
                }
            }
            grad_bias[channel] = sum_grad;
            grad_weight[channel] = sum_grad_xhat;
        }

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, &dx) in grad.data.iter_mut().zip(grad_input.iter()) {
                *gi += dx;
            }
        }

        if let Some(weight) = &self.weight {
            if let Some(weight_grad) = &weight.grad_lock() {
                let mut grad = weight_grad.lock().unwrap();
                for (gi, &dw) in grad.data.iter_mut().zip(grad_weight.iter()) {
                    *gi += dw;
                }
            }
        }

        if let Some(bias) = &self.bias {
            if let Some(bias_grad) = &bias.grad_lock() {
                let mut grad = bias_grad.lock().unwrap();
                for (gi, &db) in grad.data.iter_mut().zip(grad_bias.iter()) {
                    *gi += db;
                }
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        let mut parents = vec![&self.input];
        if let Some(weight) = &self.weight {
            parents.push(weight);
        }
        if let Some(bias) = &self.bias {
            parents.push(bias);
        }
        parents
    }
}

pub fn make_batch_norm_grad(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
) -> GradFnRef {
    Arc::new(BatchNormGrad {
        input: input.clone(),
        weight: weight.cloned(),
        bias: bias.cloned(),
        mean,
        inv_std,
    })
}

struct LogSoftmaxGrad {
    input: Tensor,
    output_data: Vec<f32>,
    dim: usize,
    shape: Vec<usize>,
}

impl GradFn for LogSoftmaxGrad {
    fn backward(&self, grad_out: &[f32]) {
        let dim_size = self.shape[self.dim];
        let inner_size: usize = self.shape[self.dim + 1..].iter().product();
        let outer_size: usize = self.shape[..self.dim].iter().product();
        let mut grad_input = vec![0.0f32; grad_out.len()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum_grad = 0.0f32;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum_grad += grad_out[idx];
                }
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    let softmax = self.output_data[idx].exp();
                    grad_input[idx] = grad_out[idx] - softmax * sum_grad;
                }
            }
        }

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_input.iter()) {
                *gi += new_grad;
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_log_softmax_grad(
    input: &Tensor,
    output_data: Vec<f32>,
    dim: usize,
    shape: &[usize],
) -> GradFnRef {
    Arc::new(LogSoftmaxGrad {
        input: input.clone(),
        output_data,
        dim,
        shape: shape.to_vec(),
    })
}

struct NllLossGrad {
    input: Tensor,
    targets: Tensor,
    batch: usize,
    classes: usize,
}

impl GradFn for NllLossGrad {
    fn backward(&self, grad_out: &[f32]) {
        let scale = grad_out.get(0).copied().unwrap_or(1.0) / self.batch as f32;
        let mut grad_input = vec![0.0f32; self.input.numel()];
        for i in 0..self.batch {
            let target = self.targets.storage().data[i] as isize;
            if target < 0 || target as usize >= self.classes {
                continue;
            }
            let idx = i * self.classes + target as usize;
            grad_input[idx] = -scale;
        }

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_input.iter()) {
                *gi += new_grad;
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input, &self.targets]
    }
}

pub fn make_nll_loss_grad(
    input: &Tensor,
    targets: &Tensor,
    batch: usize,
    classes: usize,
) -> GradFnRef {
    Arc::new(NllLossGrad {
        input: input.clone(),
        targets: targets.clone(),
        batch,
        classes,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct BackwardStats {
    pub nodes: usize,
    pub edges: usize,
    pub steps: usize,
    pub max_ready_queue: usize,
    pub max_batch_size: usize,
    pub max_parallelism: usize,
    pub duration_ms: u128,
    pub total_node_time_ms: u128,
    pub max_node_time_ms: u128,
}

pub trait BackwardObserver: Send + Sync {
    fn on_batch_start(&self, _batch_size: usize) {}
    fn on_node_start(&self, _node_id: usize) {}
    fn on_node_end(&self, _node_id: usize, _duration: Duration) {}
    fn on_complete(&self, _stats: &BackwardStats) {}
}

#[derive(Default)]
pub struct NoopBackwardObserver;

impl BackwardObserver for NoopBackwardObserver {}

#[derive(Clone)]
pub struct BackwardConfig {
    pub max_parallelism: usize,
    pub observer: Arc<dyn BackwardObserver>,
}

impl Default for BackwardConfig {
    fn default() -> Self {
        Self {
            max_parallelism: 1,
            observer: Arc::new(NoopBackwardObserver),
        }
    }
}

impl BackwardConfig {
    pub fn new(max_parallelism: usize) -> Self {
        Self {
            max_parallelism: max_parallelism.max(1),
            observer: Arc::new(NoopBackwardObserver),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn BackwardObserver>) -> Self {
        self.observer = observer;
        self
    }
}

#[derive(Clone)]
struct GraphNode {
    tensor: Tensor,
    parents: Vec<Tensor>,
}

fn tensor_id(tensor: &Tensor) -> usize {
    Arc::as_ptr(tensor.inner()) as usize
}

fn collect_graph(
    loss: &Tensor,
) -> Result<(HashMap<usize, GraphNode>, HashMap<usize, usize>, usize)> {
    let mut nodes = HashMap::new();
    let mut child_counts: HashMap<usize, usize> = HashMap::new();
    let mut stack = vec![loss.clone()];
    let mut visited = HashSet::new();
    let mut edges = 0;

    while let Some(tensor) = stack.pop() {
        let id = tensor_id(&tensor);
        if !visited.insert(id) {
            continue;
        }
        let mut parents = Vec::new();
        if let Some(grad_fn) = tensor.grad_fn() {
            for parent in grad_fn.parents() {
                if parent.grad_lock().is_none() {
                    continue;
                }
                edges += 1;
                let parent_id = tensor_id(parent);
                *child_counts.entry(parent_id).or_insert(0) += 1;
                parents.push(parent.clone());
                if !visited.contains(&parent_id) {
                    stack.push(parent.clone());
                }
            }
        }
        nodes.insert(id, GraphNode { tensor, parents });
    }

    Ok((nodes, child_counts, edges))
}

pub fn backward(loss: &Tensor) -> Result<BackwardStats> {
    backward_with_config(loss, &BackwardConfig::default())
}

struct NodeExecution {
    id: usize,
    parents: Vec<usize>,
    grad_fn: Option<GradFnRef>,
    grad_data: Vec<f32>,
}

pub fn backward_with_config(loss: &Tensor, config: &BackwardConfig) -> Result<BackwardStats> {
    let loss_grad = loss.grad_lock().ok_or_else(|| TorchError::Autograd {
        op: "backward",
        msg: "loss tensor does not require gradients".to_string(),
    })?;
    {
        let mut grad = loss_grad.lock().unwrap();
        if grad.data.len() != 1 {
            return Err(TorchError::Autograd {
                op: "backward",
                msg: format!(
                    "loss tensor must be scalar, found {} elements",
                    grad.data.len()
                ),
            });
        }
        grad.data[0] = 1.0;
    }

    let (nodes, mut pending_children, edges) = collect_graph(loss)?;
    let loss_id = tensor_id(loss);
    let mut ready = VecDeque::from([loss_id]);
    let mut steps = 0;
    let mut max_ready_queue = ready.len();
    let mut max_batch_size = 0usize;
    let mut max_parallelism = 1usize;
    let observer = Arc::clone(&config.observer);
    let start_time = Instant::now();
    let total_node_time_ms = AtomicU64::new(0);
    let max_node_time_ms = AtomicU64::new(0);

    while !ready.is_empty() {
        let mut batch = Vec::new();
        while let Some(id) = ready.pop_front() {
            batch.push(id);
        }
        max_ready_queue = max_ready_queue.max(batch.len());
        max_batch_size = max_batch_size.max(batch.len());
        observer.on_batch_start(batch.len());

        let mut executions = Vec::with_capacity(batch.len());
        for id in batch {
            let node = nodes.get(&id).ok_or_else(|| TorchError::Autograd {
                op: "backward",
                msg: "graph node missing during traversal".to_string(),
            })?;
            let grad_fn = node.tensor.grad_fn();
            let grad_data = if grad_fn.is_some() {
                let grad = node.tensor.grad().ok_or_else(|| TorchError::Autograd {
                    op: "backward",
                    msg: "missing gradient buffer for tensor".to_string(),
                })?;
                grad.data
            } else {
                Vec::new()
            };
            let parents = node.parents.iter().map(tensor_id).collect();
            executions.push(NodeExecution {
                id,
                parents,
                grad_fn,
                grad_data,
            });
        }

        let requested_parallelism = config.max_parallelism.max(1);
        let batch_parallelism = requested_parallelism.min(executions.len().max(1));
        max_parallelism = max_parallelism.max(batch_parallelism);

        let run_node = |exec: &NodeExecution| {
            observer.on_node_start(exec.id);
            let node_start = Instant::now();
            if let Some(grad_fn) = &exec.grad_fn {
                grad_fn.backward(&exec.grad_data);
            }
            let elapsed = node_start.elapsed();
            let elapsed_ms = elapsed.as_millis() as u64;
            total_node_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
            let mut current_max = max_node_time_ms.load(Ordering::Relaxed);
            while elapsed_ms > current_max {
                match max_node_time_ms.compare_exchange(
                    current_max,
                    elapsed_ms,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(updated) => current_max = updated,
                }
            }
            observer.on_node_end(exec.id, elapsed);
        };

        if batch_parallelism <= 1 {
            for exec in &executions {
                run_node(exec);
            }
        } else {
            let chunk_size = (executions.len() + batch_parallelism - 1) / batch_parallelism;
            std::thread::scope(|scope| {
                for chunk in executions.chunks(chunk_size) {
                    scope.spawn(move || {
                        for exec in chunk {
                            run_node(exec);
                        }
                    });
                }
            });
        }

        steps += executions.len();
        for exec in &executions {
            for parent_id in &exec.parents {
                if let Some(count) = pending_children.get_mut(parent_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        ready.push_back(*parent_id);
                    }
                }
            }
        }
    }

    if steps != nodes.len() {
        return Err(TorchError::Autograd {
            op: "backward",
            msg: format!(
                "backward traversal incomplete: visited {steps} of {} nodes",
                nodes.len()
            ),
        });
    }

    let stats = BackwardStats {
        nodes: nodes.len(),
        edges,
        steps,
        max_ready_queue,
        max_batch_size,
        max_parallelism,
        duration_ms: start_time.elapsed().as_millis(),
        total_node_time_ms: total_node_time_ms.load(Ordering::Relaxed) as u128,
        max_node_time_ms: max_node_time_ms.load(Ordering::Relaxed) as u128,
    };
    info!(
        "autograd.backward completed: nodes={}, edges={}, steps={}, max_ready_queue={}, max_batch_size={}, max_parallelism={}, duration_ms={}, total_node_time_ms={}, max_node_time_ms={}",
        stats.nodes,
        stats.edges,
        stats.steps,
        stats.max_ready_queue,
        stats.max_batch_size,
        stats.max_parallelism,
        stats.duration_ms,
        stats.total_node_time_ms,
        stats.max_node_time_ms
    );
    observer.on_complete(&stats);
    Ok(stats)
}

pub struct BroadcastBackward {
    pub input: Tensor,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl GradFn for BroadcastBackward {
    fn backward(&self, grad_out: &[f32]) {
        // grad_out has the broadcasted (output) shape, we need to sum it back to input_shape
        let grad_out_tensor =
            Tensor::from_vec_f32(grad_out.to_vec(), &self.output_shape, None, false);
        let grad_input = sum_to_shape(&grad_out_tensor, &self.input_shape);

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_input.storage().data.iter()) {
                *gi += new_grad;
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_broadcast_grad(input: &Tensor, output_shape: &[usize]) -> GradFnRef {
    Arc::new(BroadcastBackward {
        input: input.clone(),
        input_shape: input.shape().to_vec(),
        output_shape: output_shape.to_vec(),
    })
}

pub struct ReshapeBackward {
    pub input: Tensor,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_out: &[f32]) {
        // grad_out has the reshaped (output) shape, but the data is the same
        // We just need to accumulate it back to the input tensor's gradient
        // The number of elements should be the same since reshape preserves total elements

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            // Both grad_out and grad.data should have the same number of elements
            assert_eq!(
                grad_out.len(),
                grad.data.len(),
                "Gradient output length should match input gradient length in reshape backward"
            );

            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_out.iter()) {
                *gi += new_grad;
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_reshape_grad(input: &Tensor, _out_shape: &[usize]) -> GradFnRef {
    Arc::new(ReshapeBackward {
        input: input.clone(),
    })
}

fn reduce_sum_along_axis(data: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let mut out_shape = shape.to_vec();
    let axis_dim = out_shape[axis];
    out_shape[axis] = 1;
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_len];

    let stride: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    for outer_idx in 0..outer {
        for inner_idx in 0..stride {
            let mut sum = 0.0;
            for a in 0..axis_dim {
                let idx = outer_idx * shape[axis] * stride + a * stride + inner_idx;
                sum += data[idx];
            }
            let out_idx = outer_idx * stride + inner_idx;
            out[out_idx] = sum;
        }
    }
    out
}

// sum_to_shape: pure helper (used only in backward of broadcast)
fn sum_to_shape(x: &Tensor, target_shape: &[usize]) -> Tensor {
    let x_shape = x.shape();
    if x_shape == target_shape {
        return x.clone();
    }

    // Compute which axes were broadcasted
    let mut tgt = vec![1; x_shape.len()];
    tgt[(x_shape.len() - target_shape.len())..].copy_from_slice(target_shape);

    let mut axes = Vec::new();
    for (i, (&xd, &td)) in x_shape.iter().zip(&tgt).enumerate() {
        if td == 1 && xd > 1 {
            axes.push(i);
        }
    }

    // Now reduce across those axes manually
    let mut reduced = x.storage().data.to_vec();
    let reduced_shape = x_shape.to_vec();

    for axis in axes.iter_mut().rev() {
        reduced = reduce_sum_along_axis(&reduced, &reduced_shape, *axis);
        *axis = 1;
    }

    // If target_shape is smaller (i.e., keepdim == false), reshape down
    let mut final_shape = reduced_shape.clone();
    while final_shape.len() > target_shape.len() {
        let start = final_shape.len() - target_shape.len();
        final_shape = final_shape[start..].to_vec();
    }

    Tensor::from_vec_f32(reduced, target_shape, None, false)
}

struct MseLossGrad {
    predictions: Tensor,
    targets: Tensor,
    n: f32,
}
impl GradFn for MseLossGrad {
    fn backward(&self, grad_out: &[f32]) {
        // For MSE = (1/n) * sum((pred - target)^2)
        // grad_pred = (2/n) * (pred - target) * grad_out
        // grad_target = -(2/n) * (pred - target) * grad_out
        let grad_scale = 2.0 / self.n * grad_out[0];

        if let Some(g) = &self.predictions.grad_lock() {
            for ((gi, &pred), &target) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(self.predictions.storage().data.iter())
                .zip(self.targets.storage().data.iter())
            {
                *gi += grad_scale * (pred - target);
            }
        }

        if let Some(g) = &self.targets.grad_lock() {
            for ((gi, &pred), &target) in g
                .lock()
                .unwrap()
                .data
                .iter_mut()
                .zip(self.predictions.storage().data.iter())
                .zip(self.targets.storage().data.iter())
            {
                *gi -= grad_scale * (pred - target);
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.predictions, &self.targets]
    }
}
pub fn make_mse_loss_grad(predictions: &Tensor, targets: &Tensor, n: f32) -> GradFnRef {
    Arc::new(MseLossGrad {
        predictions: predictions.clone(),
        targets: targets.clone(),
        n,
    })
}

pub struct ReluGrad {
    pub input: Tensor,
}

impl GradFn for ReluGrad {
    fn backward(&self, grad_out: &[f32]) {
        let input_storage = self.input.storage();
        let input_data = input_storage.data.as_slice();
        assert_eq!(
            grad_out.len(),
            input_data.len(),
            "relu backward: grad_out len {} != input len {}",
            grad_out.len(),
            input_data.len()
        );

        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for ((gi, &u), &x) in grad
                .data
                .iter_mut()
                .zip(grad_out.iter())
                .zip(input_data.iter())
            {
                if x > 0.0 {
                    *gi += u;
                }
            }
        }
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_relu_grad(input: &Tensor) -> GradFnRef {
    Arc::new(ReluGrad {
        input: input.clone(),
    })
}

pub struct SumGrad {
    pub input: Tensor,
}

impl GradFn for SumGrad {
    fn backward(&self, grad_out: &[f32]) {
        assert_eq!(
            grad_out.len(),
            1,
            "sum backward expects scalar grad_out, got {}",
            grad_out.len()
        );
        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for gi in grad.data.iter_mut() {
                *gi += grad_out[0];
            }
        }
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

pub fn make_sum_grad(input: &Tensor) -> GradFnRef {
    Arc::new(SumGrad {
        input: input.clone(),
    })
}
