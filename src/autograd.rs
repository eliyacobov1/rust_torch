
use crate::tensor::Tensor;
use crate::tensor::GradFnRef;
use std::sync::{Arc,Weak};

pub struct Grad { pub data: Vec<f32> }
impl Grad {
    pub fn zeros_like_shape(shape: &[usize]) -> Self {
        Self { data: vec![0.0; shape.iter().product()] }
    }
}
pub trait GradFn { fn backward(&self, grad_out:&[f32]); }

struct MatMulGrad { a: Weak<Tensor>, b: Weak<Tensor> }
impl GradFn for MatMulGrad {
    fn backward(&self, grad_out: &[f32]) {
        let grad_out_tensor = Tensor::new(grad_out.to_vec(), &[self.a.upgrade().unwrap().shape[0], self.b.upgrade().unwrap().shape[1]], None, false);
        // Assume a and b are 2D tensors (matrices)
        let a = match self.a.upgrade() {
            Some(a) => a,
            None => return,
        };
        let b = match self.b.upgrade() {
            Some(b) => b,
            None => return,
        };

        // Shapes: a: (m, k), b: (k, n), grad_out: (m, n)

        // Compute grad for a: grad_a = grad_out.matmul(b.T)
        if let Some(grad_a) = &a.grad {
            let mut grad_a_lock = grad_a.lock().unwrap();
            grad_a_lock.data = (&grad_out_tensor * &b.transpose()).storage.data;
        }

        // Compute grad for b: grad_b = a.T.matmul(grad_out)
        if let Some(grad_b) = &b.grad {
            let mut grad_b_lock = grad_b.lock().unwrap();
            grad_b_lock.data = (&a.transpose() * &grad_out_tensor).storage.data;
        }
    }
}
pub fn make_matmul_grad(a:Arc<Tensor>, b:Arc<Tensor>)->GradFnRef{
    Arc::new(MatMulGrad{a:Arc::downgrade(&a),b:Arc::downgrade(&b)})
}

struct AddGrad { a: Weak<Tensor>, b: Weak<Tensor> }
impl GradFn for AddGrad {
    fn backward(&self, grad_out:&[f32]) {
        if let Some(a)=self.a.upgrade() {
            if let Some(g)=&a.grad 
            { 
                for (gi,&u) in g.lock().unwrap().data.iter_mut().zip(grad_out)
                {
                    *gi+=u;
                } 
            }
        }
        if let Some(b)=self.b.upgrade() 
        {
            if let Some(g)=&b.grad 
            { 
                for (gi,&u) in g.lock().unwrap().data.iter_mut().zip(grad_out)
                {
                    *gi+=u;
                } 
            }
        }
    }
}
pub fn make_add_grad(a:Arc<Tensor>, b:Arc<Tensor>)->GradFnRef{
    Arc::new(AddGrad{a:Arc::downgrade(&a),b:Arc::downgrade(&b)})
}

pub fn backward(loss:&Arc<Tensor>){
    if let Some(g)=&loss.grad 
    {
         g.lock().unwrap().data[0]=1.0; 
    }
    if let Some(gf)=&loss.grad_fn 
    {
         gf.backward(&[1.0]);
    }
}
