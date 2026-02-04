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

/// Simple MLP classifier (linear -> ReLU -> linear -> log_softmax).
#[derive(Debug, Clone)]
pub struct MlpClassifier {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl MlpClassifier {
    /// Initialize the MLP classifier with seeded uniform weights.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        seed: u64,
    ) -> Result<Self> {
        if in_features == 0 || hidden_features == 0 || out_features == 0 {
            return Err(TorchError::InvalidArgument {
                op: "mlp_classifier.new",
                msg: "in_features/hidden_features/out_features must be > 0".to_string(),
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w1 = Vec::with_capacity(in_features * hidden_features);
        for _ in 0..w1.capacity() {
            w1.push(rng.gen_range(-0.1..0.1));
        }
        let b1 = vec![0.0f32; hidden_features];
        let mut w2 = Vec::with_capacity(hidden_features * out_features);
        for _ in 0..w2.capacity() {
            w2.push(rng.gen_range(-0.1..0.1));
        }
        let b2 = vec![0.0f32; out_features];
        Ok(Self {
            w1: Tensor::from_vec_f32(w1, &[in_features, hidden_features], None, true),
            b1: Tensor::from_vec_f32(b1, &[hidden_features], None, true),
            w2: Tensor::from_vec_f32(w2, &[hidden_features, out_features], None, true),
            b2: Tensor::from_vec_f32(b2, &[out_features], None, true),
        })
    }

    /// Forward pass: log_softmax(W2 * relu(W1 * x + b1) + b2).
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.validate_layout("mlp_classifier.forward")?;
        if input.shape().len() != 2 {
            return Err(TorchError::InvalidArgument {
                op: "mlp_classifier.forward",
                msg: "input must be a 2D tensor".to_string(),
            });
        }
        if input.shape()[1] != self.w1.shape()[0] {
            return Err(TorchError::InvalidArgument {
                op: "mlp_classifier.forward",
                msg: format!(
                    "input features {} do not match weights {}",
                    input.shape()[1],
                    self.w1.shape()[0]
                ),
            });
        }
        let hidden = ops::linear(input, &self.w1, &self.b1);
        let activated = ops::relu(&hidden);
        let logits = ops::linear(&activated, &self.w2, &self.b2);
        Ok(ops::log_softmax(&logits, 1))
    }

    /// Access trainable parameters as mutable views.
    pub fn parameters_mut(&mut self) -> Vec<ParameterView<'_>> {
        vec![
            ParameterView {
                name: "w1",
                tensor: &mut self.w1,
            },
            ParameterView {
                name: "b1",
                tensor: &mut self.b1,
            },
            ParameterView {
                name: "w2",
                tensor: &mut self.w2,
            },
            ParameterView {
                name: "b2",
                tensor: &mut self.b2,
            },
        ]
    }

    /// Build a state dict for persistence.
    pub fn state_dict(&self) -> StateDict {
        let mut state = BTreeMap::new();
        state.insert("w1".to_string(), self.w1.clone());
        state.insert("b1".to_string(), self.b1.clone());
        state.insert("w2".to_string(), self.w2.clone());
        state.insert("b2".to_string(), self.b2.clone());
        state
    }

    /// Load parameters from a state dict.
    pub fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        let w1 = state.get("w1").ok_or_else(|| TorchError::InvalidArgument {
            op: "mlp_classifier.load_state_dict",
            msg: "missing w1".to_string(),
        })?;
        let b1 = state.get("b1").ok_or_else(|| TorchError::InvalidArgument {
            op: "mlp_classifier.load_state_dict",
            msg: "missing b1".to_string(),
        })?;
        let w2 = state.get("w2").ok_or_else(|| TorchError::InvalidArgument {
            op: "mlp_classifier.load_state_dict",
            msg: "missing w2".to_string(),
        })?;
        let b2 = state.get("b2").ok_or_else(|| TorchError::InvalidArgument {
            op: "mlp_classifier.load_state_dict",
            msg: "missing b2".to_string(),
        })?;
        if w1.shape() != self.w1.shape()
            || b1.shape() != self.b1.shape()
            || w2.shape() != self.w2.shape()
            || b2.shape() != self.b2.shape()
        {
            return Err(TorchError::InvalidArgument {
                op: "mlp_classifier.load_state_dict",
                msg: "state dict shapes do not match model".to_string(),
            });
        }
        self.w1 = w1.clone();
        self.b1 = b1.clone();
        self.w2 = w2.clone();
        self.b2 = b2.clone();
        Ok(())
    }
}
