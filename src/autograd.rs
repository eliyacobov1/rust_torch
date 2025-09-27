
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
