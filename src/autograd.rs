
use crate::tensor::Tensor;
use crate::tensor::GradFnRef;
use std::sync::{Arc,Weak};

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
