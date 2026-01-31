use crate::error::{Result, TorchError};
use crate::tensor::Tensor;

/// Mutable parameter view used by optimizers.
#[derive(Debug)]
pub struct ParameterView<'a> {
    pub name: &'a str,
    pub tensor: &'a mut Tensor,
}

/// Optimizer trait for updating trainable parameters.
pub trait Optimizer {
    fn step(&mut self, params: &mut [ParameterView<'_>]) -> Result<()>;
    fn zero_grad(&mut self, params: &mut [ParameterView<'_>]) -> Result<()>;
}

/// Stochastic Gradient Descent optimizer.
#[derive(Debug, Clone)]
pub struct Sgd {
    pub learning_rate: f32,
    pub weight_decay: f32,
}

impl Sgd {
    /// Create a new SGD optimizer with optional weight decay.
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(TorchError::Optimizer {
                op: "sgd.new",
                msg: "learning_rate must be > 0".to_string(),
            });
        }
        if weight_decay < 0.0 {
            return Err(TorchError::Optimizer {
                op: "sgd.new",
                msg: "weight_decay must be >= 0".to_string(),
            });
        }
        Ok(Self {
            learning_rate,
            weight_decay,
        })
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [ParameterView<'_>]) -> Result<()> {
        for param in params.iter_mut() {
            let grad = param.tensor.grad().ok_or_else(|| TorchError::Optimizer {
                op: "sgd.step",
                msg: format!("missing gradient for {}", param.name),
            })?;
            let mut data = param.tensor.storage().data.clone();
            if data.len() != grad.data.len() {
                return Err(TorchError::Optimizer {
                    op: "sgd.step",
                    msg: format!(
                        "gradient size mismatch for {} ({} vs {})",
                        param.name,
                        grad.data.len(),
                        data.len()
                    ),
                });
            }
            for (idx, value) in data.iter_mut().enumerate() {
                let decay = if self.weight_decay > 0.0 {
                    self.weight_decay * *value
                } else {
                    0.0
                };
                *value -= self.learning_rate * (grad.data[idx] + decay);
            }
            let shape = param.tensor.shape().to_vec();
            let requires_grad = param.tensor.requires_grad();
            *param.tensor = Tensor::from_vec_f32(data, &shape, None, requires_grad);
        }
        Ok(())
    }

    fn zero_grad(&mut self, params: &mut [ParameterView<'_>]) -> Result<()> {
        for param in params.iter_mut() {
            let data = param.tensor.storage().data.clone();
            let shape = param.tensor.shape().to_vec();
            let requires_grad = param.tensor.requires_grad();
            *param.tensor = Tensor::from_vec_f32(data, &shape, None, requires_grad);
        }
        Ok(())
    }
}
