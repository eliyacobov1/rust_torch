use std::collections::BTreeMap;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::checkpoint::StateDict;
use crate::error::{Result, TorchError};
use crate::ops;
use crate::optim::ParameterView;
use crate::tensor::Tensor;

/// Simple linear regression model (dense affine transform).
#[derive(Debug, Clone)]
pub struct LinearRegression {
    weights: Tensor,
    bias: Tensor,
}

impl LinearRegression {
    /// Initialize weights and bias with a seeded uniform distribution.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(TorchError::InvalidArgument {
                op: "linear_regression.new",
                msg: "in_features and out_features must be > 0".to_string(),
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Vec::with_capacity(in_features * out_features);
        for _ in 0..weights.capacity() {
            weights.push(rng.gen_range(-0.1..0.1));
        }
        let mut bias = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            bias.push(0.0);
        }
        Ok(Self {
            weights: Tensor::from_vec_f32(weights, &[in_features, out_features], None, true),
            bias: Tensor::from_vec_f32(bias, &[out_features], None, true),
        })
    }

    /// Forward pass: y = xW + b.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.validate_layout("linear_regression.forward")?;
        if input.shape().len() != 2 {
            return Err(TorchError::InvalidArgument {
                op: "linear_regression.forward",
                msg: "input must be a 2D tensor".to_string(),
            });
        }
        if input.shape()[1] != self.weights.shape()[0] {
            return Err(TorchError::InvalidArgument {
                op: "linear_regression.forward",
                msg: format!(
                    "input features {} do not match weights {}",
                    input.shape()[1],
                    self.weights.shape()[0]
                ),
            });
        }
        let out = ops::try_matmul(input, &self.weights)?;
        ops::try_add(&out, &self.bias)
    }

    /// Access trainable parameters as mutable views.
    pub fn parameters_mut(&mut self) -> Vec<ParameterView<'_>> {
        vec![
            ParameterView {
                name: "weights",
                tensor: &mut self.weights,
            },
            ParameterView {
                name: "bias",
                tensor: &mut self.bias,
            },
        ]
    }

    /// Build a state dict for persistence.
    pub fn state_dict(&self) -> StateDict {
        let mut state = BTreeMap::new();
        state.insert("weights".to_string(), self.weights.clone());
        state.insert("bias".to_string(), self.bias.clone());
        state
    }

    /// Load parameters from a state dict.
    pub fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        let weights = state
            .get("weights")
            .ok_or_else(|| TorchError::InvalidArgument {
                op: "linear_regression.load_state_dict",
                msg: "missing weights".to_string(),
            })?;
        let bias = state
            .get("bias")
            .ok_or_else(|| TorchError::InvalidArgument {
                op: "linear_regression.load_state_dict",
                msg: "missing bias".to_string(),
            })?;
        if weights.shape() != self.weights.shape() || bias.shape() != self.bias.shape() {
            return Err(TorchError::InvalidArgument {
                op: "linear_regression.load_state_dict",
                msg: "state dict shapes do not match model".to_string(),
            });
        }
        self.weights = weights.clone();
        self.bias = bias.clone();
        Ok(())
    }
}
