
#![cfg(feature = "python-bindings")]

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use crate::{ops, tensor::Tensor};
use ndarray::ArrayD;

#[pyclass]
pub struct PyTensor { inner: Tensor }

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
