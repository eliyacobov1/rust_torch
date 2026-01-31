#![cfg(feature = "python-bindings")]

use crate::{checkpoint, ops, tensor::Tensor, TorchError};
use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

pyo3::create_exception!(rustorch, RustorchError, pyo3::exceptions::PyException);
pyo3::create_exception!(rustorch, BroadcastError, RustorchError);
pyo3::create_exception!(rustorch, LayoutError, RustorchError);
pyo3::create_exception!(rustorch, InvalidDimError, RustorchError);
pyo3::create_exception!(rustorch, InvalidArgumentError, RustorchError);
pyo3::create_exception!(rustorch, CheckpointError, RustorchError);

fn map_torch_err(err: TorchError) -> PyErr {
    match err {
        TorchError::BroadcastMismatch { .. } => BroadcastError::new_err(err.to_string()),
        TorchError::NonContiguous { .. } | TorchError::InvalidLayout { .. } => {
            LayoutError::new_err(err.to_string())
        }
        TorchError::InvalidDim { .. } => InvalidDimError::new_err(err.to_string()),
        TorchError::InvalidArgument { .. } => InvalidArgumentError::new_err(err.to_string()),
        TorchError::CheckpointIo { .. }
        | TorchError::CheckpointFormat { .. }
        | TorchError::CheckpointDtypeMismatch { .. }
        | TorchError::CheckpointShapeMismatch { .. }
        | TorchError::CheckpointLayoutMismatch { .. } => CheckpointError::new_err(err.to_string()),
        TorchError::Autograd { .. }
        | TorchError::Experiment { .. }
        | TorchError::Data { .. }
        | TorchError::Optimizer { .. }
        | TorchError::Training { .. } => RustorchError::new_err(err.to_string()),
    }
}

#[pyclass]
pub struct PyTensor {
    inner: Tensor,
}

#[pymodule]
pub fn rustorch(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(run_fx, m)?)?;
    m.add_function(wrap_pyfunction!(save_state_dict, m)?)?;
    m.add_function(wrap_pyfunction!(load_state_dict, m)?)?;
    m.add("RustorchError", py.get_type::<RustorchError>())?;
    m.add("BroadcastError", py.get_type::<BroadcastError>())?;
    m.add("LayoutError", py.get_type::<LayoutError>())?;
    m.add("InvalidDimError", py.get_type::<InvalidDimError>())?;
    m.add(
        "InvalidArgumentError",
        py.get_type::<InvalidArgumentError>(),
    )?;
    m.add("CheckpointError", py.get_type::<CheckpointError>())?;
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

    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        ops::try_add(&self.inner, &other.inner)
            .map(|inner| PyTensor { inner })
            .map_err(map_torch_err)
    }

    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        ops::try_mul(&self.inner, &other.inner)
            .map(|inner| PyTensor { inner })
            .map_err(map_torch_err)
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        ops::try_matmul(&self.inner, &other.inner)
            .map(|inner| PyTensor { inner })
            .map_err(map_torch_err)
    }

    fn relu(&self) -> PyTensor {
        PyTensor {
            inner: ops::relu(&self.inner),
        }
    }

    #[pyo3(signature = (p=0.5, training=true, seed=None))]
    fn dropout(&self, p: f32, training: bool, seed: Option<u64>) -> PyResult<PyTensor> {
        ops::try_dropout(&self.inner, p, training, seed)
            .map(|inner| PyTensor { inner })
            .map_err(map_torch_err)
    }

    #[pyo3(signature = (kernel_size, stride=None, padding=0, dilation=1, ceil_mode=false))]
    fn max_pool2d(
        &self,
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> PyResult<PyTensor> {
        ops::try_max_pool2d(
            &self.inner,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        .map(|inner| PyTensor { inner })
        .map_err(map_torch_err)
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
    ) -> PyResult<PyTensor> {
        let _ = momentum;
        ops::try_batch_norm(
            &self.inner,
            running_mean.map(|t| &t.inner),
            running_var.map(|t| &t.inner),
            weight.map(|t| &t.inner),
            bias.map(|t| &t.inner),
            training,
            eps,
        )
        .map(|inner| PyTensor { inner })
        .map_err(map_torch_err)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.inner
            .try_reshape(&shape)
            .map(|inner| PyTensor { inner })
            .map_err(map_torch_err)
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

#[pyfunction]
fn save_state_dict(path: &str, state: &PyAny) -> PyResult<()> {
    let dict: &PyDict = state.downcast()?;
    let mut map = BTreeMap::new();
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let tensor: PyRef<PyTensor> = value.extract()?;
        map.insert(name, tensor.inner.clone());
    }
    checkpoint::save_state_dict(path, &map).map_err(map_torch_err)
}

#[pyfunction]
fn load_state_dict(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let state = checkpoint::load_state_dict(path).map_err(map_torch_err)?;
    let dict = PyDict::new(py);
    for (name, tensor) in state {
        let py_tensor = Py::new(py, PyTensor { inner: tensor })?;
        dict.set_item(name, py_tensor)?;
    }
    Ok(dict.into_py(py))
}
