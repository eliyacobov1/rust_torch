
#![cfg(feature = "python-bindings")]

use pyo3::prelude::*;
use numpy::{PyArrayDyn, IntoPyArray};
use std::sync::Arc;
use crate::{tensor::Tensor, ops, autograd};

#[pyclass]
pub struct PyTensor { inner: Arc<Tensor> }

#[pymodule]
pub fn rustorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
