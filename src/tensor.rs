
use crate::storage::Storage;
use crate::autograd::{GradFn, Grad};
use std::sync::{Arc, Mutex};
pub type GradFnRef = Arc<dyn GradFn + Send + Sync>;

pub struct Tensor {
    pub storage: Storage,
    pub requires_grad: bool,
    pub grad: Option<Arc<Mutex<Grad>>>,
    pub grad_fn: Option<GradFnRef>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_vec_f32(v: Vec<f32>, shape: &[usize], grad_fn: Option<GradFnRef>, requires_grad: bool) -> Arc<Self> {
        Arc::new(Self {
                        storage: Storage{data: v},
                        requires_grad,
                        grad: if requires_grad {Some(Arc::new(Mutex::new(Grad::zeros_like_shape(&shape))))} else {None},
                        grad_fn,
                        shape: shape.to_vec() })
    }
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Arc<Self> {
        Self::from_vec_f32(vec![0.0; shape.iter().product()], shape, None, requires_grad)
    }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}
