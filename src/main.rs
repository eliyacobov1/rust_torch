
use rustorch::{ops, autograd};
use rustorch::tensor::Tensor;

fn main() {
    let a = Tensor::from_vec_f32(vec![1.0,2.0,3.0,4.0], &[2,2], None, true);
    let b = Tensor::from_vec_f32(vec![5.0,6.0,7.0,8.0], &[2,2], None, true);
    let c = ops::add(&a, &b);
    let d = ops::matmul(&c, &c);
    println!("Forward numel={} shape={:?}", d.numel(), d.shape);
    autograd::backward(&d);
    if let Some(g)=&a.grad 
    { 
        println!("Grad a: {:?}", g.lock().unwrap().data); 
    }
}
