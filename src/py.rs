#![cfg(feature = "python-bindings")]

use crate::{ops, tensor::Tensor};
use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass]
pub struct PyTensor {
    inner: Tensor,
}

#[pymodule]
pub fn rustorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(run_fx, m)?)?;
    Ok(())
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (array, requires_grad=false))]
    fn new(array: PyReadonlyArrayDyn<f32>, requires_grad: bool) -> PyResult<Self> {
        let shape = array.shape().to_vec();
        let data = array.as_array().iter().cloned().collect();
        Ok(Self {
            inner: Tensor::from_vec_f32(data, &shape, None, requires_grad),
        })
    }

    fn add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: ops::add(&self.inner, &other.inner),
        }
    }

    fn mul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: ops::mul(&self.inner, &other.inner),
        }
    }

    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: ops::matmul(&self.inner, &other.inner),
        }
    }

    fn relu(&self) -> PyTensor {
        PyTensor {
            inner: ops::relu(&self.inner),
        }
    }

    #[pyo3(signature = (p=0.5, training=true, seed=None))]
    fn dropout(&self, p: f32, training: bool, seed: Option<u64>) -> PyTensor {
        PyTensor {
            inner: ops::dropout(&self.inner, p, training, seed),
        }
    }

    #[pyo3(signature = (kernel_size, stride=None, padding=0, dilation=1, ceil_mode=false))]
    fn max_pool2d(
        &self,
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> PyTensor {
        PyTensor {
            inner: ops::max_pool2d(
                &self.inner,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ),
        }
    }

    #[pyo3(signature = (running_mean=None, running_var=None, weight=None, bias=None, training=false, momentum=0.1, eps=1e-5))]
    fn batch_norm(
        &self,
        running_mean: Option<&PyTensor>,
        running_var: Option<&PyTensor>,
        weight: Option<&PyTensor>,
        bias: Option<&PyTensor>,
        training: bool,
        momentum: f32,
        eps: f32,
    ) -> PyTensor {
        PyTensor {
            inner: ops::batch_norm(
                &self.inner,
                running_mean.map(|t| &t.inner),
                running_var.map(|t| &t.inner),
                weight.map(|t| &t.inner),
                bias.map(|t| &t.inner),
                training,
                momentum,
                eps,
            ),
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> PyTensor {
        PyTensor {
            inner: self.inner.reshape(&shape),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f32>> {
        let array = ArrayD::from_shape_vec(
            self.inner.shape().to_vec(),
            self.inner.storage().data.clone(),
        )
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

#[pyfunction]
fn run_fx(py: Python<'_>, gm: PyObject, example_inputs: &PyAny) -> PyResult<PyObject> {
    if let Ok(module) = py.import("rust_backend.fx_runner") {
        if let Ok(func) = module.getattr("run_fx") {
            return func.call1((gm, example_inputs)).map(|obj| obj.into());
        }
    }
    Ok(gm)
}
