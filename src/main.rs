use rustorch::tensor::Tensor;
use rustorch::{autograd, ops};

fn main() {
    // Simulate a batch of 2 samples, 2 features each
    let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], None, true);
    // First layer: 2 in, 3 out
    let w1 = Tensor::from_vec_f32(vec![0.5, -0.3, 0.8, 0.2, 0.1, -0.5], &[2, 3], None, true);
    let b1 = Tensor::from_vec_f32(vec![0.1, -0.2, 0.3], &[3], None, true);
    // Second layer: 3 in, 1 out
    let w2 = Tensor::from_vec_f32(vec![0.7, -0.6, 0.2], &[3, 1], None, true);
    let b2 = Tensor::from_vec_f32(vec![0.05], &[1], None, true);

    // Forward pass: x -> linear -> relu -> linear -> scalar output
    let h1 = ops::linear(&x, &w1, &b1); // [2, 3]
    let h1_relu = ops::relu(&h1); // [2, 3]
    let logits = ops::linear(&h1_relu, &w2, &b2); // [2, 1]
                                                  // For simplicity, sum the logits to get a single scalar output (like reduction)
    let output = ops::sum(&logits);

    // Target: single scalar
    let target = Tensor::from_vec_f32(vec![1.0], &[1], None, false);
    let loss = ops::mse_loss(&output, &target);

    println!(
        "Network output: {:?}, Loss: {}",
        output.storage().data,
        loss.storage().data[0]
    );
    autograd::backward(&loss).expect("backward failed");

    if let Some(g) = &x.grad() {
        println!("Grad x: {:?}", g.data);
    }
    if let Some(g) = &w1.grad() {
        println!("Grad w1: {:?}", g.data);
    }
    if let Some(g) = &b1.grad() {
        println!("Grad b1: {:?}", g.data);
    }
    if let Some(g) = &w2.grad() {
        println!("Grad w2: {:?}", g.data);
    }
    if let Some(g) = &b2.grad() {
        println!("Grad b2: {:?}", g.data);
    }
}
