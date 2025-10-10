use rustorch::{ops, autograd};
use rustorch::tensor::Tensor;

fn main() {
    let a = Tensor::from_vec_f32(vec![1.0,2.0,3.0,4.0], &[2,2], None, true);
    let b = Tensor::from_vec_f32(vec![5.0,6.0,7.0,8.0], &[2,2], None, true);
    let c = ops::add(&a, &b);
    let d = ops::matmul(&c, &c);
    
    // Create some target values for the loss
    let targets = Tensor::from_vec_f32(vec![100.0, 200.0, 300.0, 400.0], &[2,2], None, false);
    
    // Compute MSE loss (this returns a scalar)
    let loss = ops::mse_loss(&d, &targets);
    
    println!("Forward: predictions shape={:?}, loss={:?}", d.shape(), loss.storage().data[0]);
    autograd::backward(&loss);
    
    if let Some(g) = &a.grad() { 
        println!("Grad a: {:?}", g.data); 
    }
}
