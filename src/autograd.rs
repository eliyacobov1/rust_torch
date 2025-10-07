
use crate::tensor::Tensor;
use crate::tensor::GradFnRef;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Grad { pub data: Vec<f32> }
impl Grad {
    pub fn zeros_like_shape(shape: &[usize]) -> Self {
        Self { data: vec![0.0; shape.iter().product()] }
    }
}
pub trait GradFn { fn backward(&self, grad_out:&[f32]); }

struct MatMulGrad { a: Tensor, b: Tensor }
impl GradFn for MatMulGrad {
    fn backward(&self, grad_out: &[f32]) {
        let grad_out_tensor = Tensor::new(grad_out.to_vec(), &[self.a.shape()[0], self.b.shape()[1]], None, false);
        // Shapes: a: (m, k), b: (k, n), grad_out: (m, n)

        // Compute grad for a: grad_a = grad_out.matmul(b.T)
        if let Some(grad_a) = &self.a.grad_lock() {
            let mut grad_a_lock = grad_a.lock().unwrap();
            grad_a_lock.data = (&grad_out_tensor * &self.b.transpose()).storage().data; // TODO: handle unnessecary copy here
        }

        // Compute grad for b: grad_b = a.T.matmul(grad_out)
        if let Some(grad_b) = &self.b.grad_lock() {
            let mut grad_b_lock = grad_b.lock().unwrap();
            grad_b_lock.data = (&self.a.transpose() * &grad_out_tensor).storage().data;
        }
    }
}
pub fn make_matmul_grad(a:&Tensor, b:&Tensor)->GradFnRef{
    Arc::new(MatMulGrad { a: a.clone(), b: b.clone() })
}

struct AddGrad { a: Tensor, b: Tensor }
impl GradFn for AddGrad {
    fn backward(&self, grad_out:&[f32]) {
        if let Some(g)=&self.a.grad_lock()
        { 
            for (gi,&u) in g.lock().unwrap().data.iter_mut().zip(grad_out)
            {
                *gi+=u;
            } 
        }
        if let Some(g)=&self.b.grad_lock() 
        { 
            for (gi,&u) in g.lock().unwrap().data.iter_mut().zip(grad_out)
            {
                *gi+=u;
            } 
        }
    }
}
pub fn make_add_grad(a:&Tensor, b:&Tensor)->GradFnRef{
    Arc::new(AddGrad{a: a.clone(), b: b.clone()})
}

pub fn backward(loss:&Tensor){
    if let Some(g)=&loss.grad_lock() 
    {
         g.lock().unwrap().data[0]=1.0; 
    }
    if let Some(gf)=&loss.grad_fn()
    {
         gf.backward(&[1.0]);
    }
}

pub struct BroadcastBackward {
    pub input: Tensor,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl GradFn for BroadcastBackward {
    fn backward(&self, grad_out: &[f32]) {
        // grad_out has the broadcasted (output) shape, we need to sum it back to input_shape
        let grad_out_tensor = Tensor::from_vec_f32(grad_out.to_vec(), &self.output_shape, None, false);
        let grad_input = sum_to_shape(&grad_out_tensor, &self.input_shape);
        
        if let Some(grad_lock) = &self.input.grad_lock() {
            let mut grad = grad_lock.lock().unwrap();
            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_input.storage().data.iter()) {
                *gi += new_grad;
            }
        }
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
            assert_eq!(grad_out.len(), grad.data.len(), 
                "Gradient output length should match input gradient length in reshape backward");
            
            for (gi, &new_grad) in grad.data.iter_mut().zip(grad_out.iter()) {
                *gi += new_grad;
            }
        }
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
