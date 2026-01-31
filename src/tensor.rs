use std::array;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul};
use std::sync::{Arc, Mutex};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::autograd::{make_broadcast_grad, make_reshape_grad, Grad, GradFn};
use crate::error::{Result, TorchError};
use crate::storage::Storage;

pub type GradFnRef = Arc<dyn GradFn + Send + Sync>;

fn batch_iter<'a>(batch_shape: &'a [usize]) -> impl Iterator<Item = Vec<usize>> + 'a {
    let total: usize = batch_shape.iter().product();
    (0..total).map(move |mut idx| {
        let mut coord = vec![0; batch_shape.len()];
        for i in (0..batch_shape.len()).rev() {
            coord[i] = idx % batch_shape[i];
            idx /= batch_shape[i];
        }
        coord
    })
}

pub struct BatchIterMut<'a> {
    ptr: *mut f32,
    batch_size: usize,
    num_batches: usize,
    idx: usize,
    _marker: PhantomData<&'a mut [f32]>,
}

impl<'a> Iterator for BatchIterMut<'a> {
    type Item = &'a mut [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.num_batches {
            return None;
        }

        let start = self.idx * self.batch_size;
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), self.batch_size) };

        self.idx += 1;
        Some(slice)
    }
}

pub(crate) struct BatchTensorIter<'a, I, F>
where
    I: Iterator<Item = Vec<usize>>,
    F: FnMut(Vec<usize>) -> TensorInner + 'a,
{
    inner: std::iter::Map<I, F>,
    _marker: PhantomData<&'a F>,
}

impl<'a, I, F> Iterator for BatchTensorIter<'a, I, F>
where
    I: Iterator<Item = Vec<usize>>,
    F: FnMut(Vec<usize>) -> TensorInner + 'a,
{
    type Item = TensorInner;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

#[derive(Serialize, Deserialize)]
struct TensorSerde {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    requires_grad: bool,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("requires_grad", &self.requires_grad())
            .field("numel", &self.numel())
            .finish()
    }
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let payload = TensorSerde {
            data: self.storage().data.clone(),
            shape: self.shape().to_vec(),
            strides: self.strides().to_vec(),
            requires_grad: self.requires_grad(),
        };
        payload.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = TensorSerde::deserialize(deserializer)?;
        Tensor::try_from_vec_f32_with_strides(
            payload.data,
            &payload.shape,
            &payload.strides,
            None,
            payload.requires_grad,
        )
        .map_err(|err| D::Error::custom(err.to_string()))
    }
}

impl Tensor {
    pub fn new(
        v: Vec<f32>,
        shape: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Self {
        Self {
            inner: TensorInner::from_vec_f32(v, shape, grad_fn, requires_grad),
        }
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self {
            inner: Arc::new(TensorInner::zeros(shape, requires_grad)),
        }
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self {
            inner: Arc::new(TensorInner::ones(shape, requires_grad)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn transpose(&self) -> Self {
        self.try_transpose()
            .expect("transpose: invalid input layout")
    }

    pub fn try_transpose(&self) -> Result<Self> {
        self.validate_layout("transpose")?;
        Ok(Self {
            inner: Arc::new(self.inner.transpose()),
        })
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    pub fn grad_fn(&self) -> Option<GradFnRef> {
        self.inner.grad_fn.clone()
    }

    pub fn grad(&self) -> Option<Grad> {
        self.inner.grad.as_ref().map(|g| g.lock().unwrap().clone())
    }

    pub(crate) fn grad_lock(&self) -> Option<Arc<Mutex<Grad>>> {
        self.inner.grad.as_ref().map(Arc::clone)
    }

    pub fn storage(&self) -> Storage {
        self.inner.storage.clone()
    }

    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::get_mut(&mut self.inner)
            .expect("Cannot get mutable reference: Tensor is shared")
            .storage
            .as_mut()
    }

    pub fn from_vec_f32(
        v: Vec<f32>,
        shape: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Self {
        Self {
            inner: TensorInner::from_vec_f32(v, shape, grad_fn, requires_grad),
        }
    }

    pub fn try_from_vec_f32_with_strides(
        v: Vec<f32>,
        shape: &[usize],
        strides: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Result<Self> {
        let inner = TensorInner::new_with_strides(v, shape, strides, grad_fn, requires_grad)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        Self::try_broadcast_shape(a, b, "broadcast_shape")
            .expect("broadcast_shape: shapes not broadcastable")
    }

    pub fn try_broadcast_shape(a: &[usize], b: &[usize], op: &'static str) -> Result<Vec<usize>> {
        let mut out = Vec::new();
        let max_len = a.len().max(b.len());
        for i in 0..max_len {
            let ad = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let bd = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);
            if ad != bd && ad != 1 && bd != 1 {
                return Err(TorchError::BroadcastMismatch {
                    op,
                    lhs: a.to_vec(),
                    rhs: b.to_vec(),
                });
            }
            out.push(ad.max(bd));
        }
        out.reverse();
        Ok(out)
    }

    pub fn set_grad_fn(&mut self, grad_fn: Option<GradFnRef>) {
        Arc::get_mut(&mut self.inner)
            .expect("Cannot get mutable reference: Tensor is shared")
            .grad_fn = grad_fn;
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Tensor {
        self.try_broadcast_to(target_shape, "broadcast_to")
            .expect("broadcast_to: invalid target shape")
    }

    pub fn try_broadcast_to(&self, target_shape: &[usize], op: &'static str) -> Result<Tensor> {
        // TODO: broadcast optimizations (no copy, etc)
        let src_shape = self.shape();
        if target_shape.len() < src_shape.len() {
            return Err(TorchError::InvalidArgument {
                op,
                msg: "cannot broadcast to fewer dimensions".to_string(),
            });
        }

        // Align dimensions (pad with 1s on the left)
        let mut src_aligned = vec![1; target_shape.len()];
        src_aligned[(target_shape.len() - src_shape.len())..].copy_from_slice(src_shape);

        for (src_dim, target_dim) in src_aligned.iter().zip(target_shape.iter()) {
            if *src_dim != 1 && src_dim != target_dim {
                return Err(TorchError::BroadcastMismatch {
                    op,
                    lhs: src_shape.to_vec(),
                    rhs: target_shape.to_vec(),
                });
            }
        }

        // Compute total element count
        let out_len: usize = target_shape.iter().product();
        let mut out_data = Vec::with_capacity(out_len);

        // index broadcasting
        fn broadcast_index(
            src_data: &[f32],
            src_shape: &[usize],
            target_shape: &[usize],
            out_data: &mut Vec<f32>,
        ) {
            // TODO: this function goes iterates over all elements in the broadcasted tensor, highly inefficient
            let ndim = target_shape.len();
            let mut idx = vec![0; ndim];

            // Total number of elements in the broadcasted tensor
            let total_elems: usize = target_shape.iter().product();

            for linear_idx in 0..total_elems {
                // Decode linear index into multi-dimensional indices
                let mut rem = linear_idx;
                for d in (0..ndim).rev() {
                    idx[d] = rem % target_shape[d];
                    rem /= target_shape[d];
                }

                // Map broadcasted coordinates to source coordinates
                let mut src_idx = Vec::with_capacity(src_shape.len());
                let offset = ndim - src_shape.len();
                for i in 0..src_shape.len() {
                    let src_dim = src_shape[i];
                    let coord = idx[offset + i];
                    src_idx.push(if src_dim == 1 { 0 } else { coord });
                }

                // Compute flat index into source tensor
                let flat_src = flatten_index(&src_idx, src_shape);
                out_data.push(src_data[flat_src]);
            }
        }

        fn flatten_index(idxs: &[usize], shape: &[usize]) -> usize {
            // TODO: this broadcast operation is not efficient in terms of memory, speed, branch prediction
            let mut flat = 0;
            let mut stride = 1;
            for (&i, &dim) in idxs.iter().rev().zip(shape.iter().rev()) {
                flat += i * stride;
                stride *= dim;
            }
            flat
        }

        broadcast_index(
            &self.storage().data,
            &src_aligned,
            target_shape,
            &mut out_data,
        );
        let grad_fn = if self.requires_grad() {
            Some(make_broadcast_grad(self, target_shape))
        } else {
            None
        };
        Ok(Self::new(
            out_data,
            target_shape,
            grad_fn,
            self.requires_grad(),
        ))
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        self.try_reshape(new_shape)
            .expect("reshape: number of elements must remain the same")
    }

    pub fn try_reshape(&self, new_shape: &[usize]) -> Result<Self> {
        self.validate_layout("reshape")?;
        let old_numel: usize = self.shape().iter().product();
        let new_numel: usize = new_shape.iter().product();
        if old_numel != new_numel {
            return Err(TorchError::InvalidArgument {
                op: "reshape",
                msg: format!(
                    "cannot reshape from {:?} (numel={}) to {:?} (numel={})",
                    self.shape(),
                    old_numel,
                    new_shape,
                    new_numel
                ),
            });
        }

        let grad_fn = if self.requires_grad() {
            Some(make_reshape_grad(self, new_shape))
        } else {
            None
        };

        let inner = TensorInner {
            storage: Storage {
                data: self.storage().data.clone(),
            },
            requires_grad: self.requires_grad(),
            grad: self.inner.grad.as_ref().map(Arc::clone),
            grad_fn,
            shape: new_shape.to_vec(),
            strides: TensorInner::contiguous_strides(new_shape),
        };
        Ok(Tensor {
            inner: Arc::new(inner),
        })
    }

    pub(crate) fn inner(&self) -> &Arc<TensorInner> {
        &self.inner
    }

    pub(crate) fn from_inner(inner: TensorInner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn matrix_dims(&self) -> [usize; 2] {
        self.inner.matrix_dims()
    }

    pub(crate) fn batch_dims(&self) -> &[usize] {
        self.inner.batch_dims()
    }

    pub fn strides(&self) -> &[usize] {
        &self.inner.strides
    }

    pub fn validate_layout(&self, op: &'static str) -> Result<()> {
        self.inner.validate_layout(op)
    }
}

#[derive(Clone)]
pub(crate) struct TensorInner {
    pub storage: Storage,
    pub requires_grad: bool,
    pub grad: Option<Arc<Mutex<Grad>>>,
    pub grad_fn: Option<GradFnRef>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

struct Matrix {
    pub tensor: Tensor,
}

impl Matrix {
    fn new(v: Option<Vec<f32>>, rows: usize, cols: usize, requires_grad: bool) -> Self {
        let shape = vec![rows, cols];
        let data = match v {
            Some(vec) => vec,
            None => vec![0.0; rows * cols],
        };
        let tensor = Tensor::new(data, &shape, None, requires_grad);
        Self { tensor: tensor }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        assert_eq!(tensor.inner.shape.len(), 2);
        Self { tensor: tensor }
    }

    fn height(&self) -> usize {
        self.tensor.inner.shape[0]
    }

    fn width(&self) -> usize {
        self.tensor.inner.shape[1]
    }

    fn shape(&self) -> (usize, usize) {
        (self.tensor.inner.shape[0], self.tensor.inner.shape[1])
    }

    fn transpose(&self) -> Self {
        let (rows, cols) = self.shape();
        let mut transposed_data = Self::new(None, cols, rows, self.tensor.inner.requires_grad);
        for i in 0..rows {
            for j in 0..cols {
                transposed_data[(j, i)] = self[(i, j)];
            }
        }
        transposed_data
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let cols = self.width();
        let idx = row * cols + col;
        &self.tensor.inner.storage.data[idx]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let cols = self.width();
        let idx = row * cols + col;
        &mut self.tensor.storage_mut().data[idx]
    }
}

macro_rules! impl_mul_tensor {
    ($Lhs:ty, $Rhs:ty) => {
        impl Mul<$Rhs> for $Lhs {
            type Output = Matrix;
            fn mul(self, rhs: $Rhs) -> Self::Output {
                assert_eq!(self.width(), rhs.height());
                let m = self.height();
                let k = self.width();
                let n = rhs.width();
                let mut out_tensor = Matrix::new(None, m, n, false);
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for p in 0..k {
                            acc += self[(i, p)] * rhs[(p, j)];
                        }
                        out_tensor[(i, j)] = acc;
                    }
                }
                out_tensor
            }
        }
    };
}

impl_mul_tensor!(&Matrix, &Matrix);
impl_mul_tensor!(Matrix, &Matrix);
impl_mul_tensor!(&Matrix, Matrix);
impl_mul_tensor!(Matrix, Matrix);

impl TensorInner {
    pub fn new(
        v: Vec<f32>,
        shape: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Self {
        Self {
            storage: Storage { data: v },
            requires_grad,
            grad: if requires_grad {
                Some(Arc::new(Mutex::new(Grad::zeros_like_shape(&shape))))
            } else {
                None
            },
            grad_fn,
            shape: shape.to_vec(),
            strides: Self::contiguous_strides(shape),
        }
    }

    pub fn new_with_strides(
        v: Vec<f32>,
        shape: &[usize],
        strides: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Result<Self> {
        Self::validate_layout_fields(shape, strides, v.len(), "from_vec_f32_with_strides")?;
        Ok(Self {
            storage: Storage { data: v },
            requires_grad,
            grad: if requires_grad {
                Some(Arc::new(Mutex::new(Grad::zeros_like_shape(&shape))))
            } else {
                None
            },
            grad_fn,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        })
    }

    pub fn from_vec_f32(
        v: Vec<f32>,
        shape: &[usize],
        grad_fn: Option<GradFnRef>,
        requires_grad: bool,
    ) -> Arc<Self> {
        Arc::new(Self::new(v, shape, grad_fn, requires_grad))
    }
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::full(shape, 0.0, requires_grad)
    }
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::full(shape, 1.0, requires_grad)
    }
    fn full(shape: &[usize], value: f32, requires_grad: bool) -> Self {
        Self::new(
            vec![value; shape.iter().product()],
            shape,
            None,
            requires_grad,
        )
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return Vec::new();
        }
        let mut strides = vec![0; shape.len()];
        let mut stride: usize = 1;
        for (idx, dim) in shape.iter().enumerate().rev() {
            strides[idx] = stride;
            stride = stride.saturating_mul(*dim);
        }
        strides
    }

    fn validate_layout_fields(
        shape: &[usize],
        strides: &[usize],
        storage_len: usize,
        op: &'static str,
    ) -> Result<()> {
        if shape.len() != strides.len() {
            return Err(TorchError::InvalidLayout {
                op,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                msg: format!(
                    "shape rank {} does not match strides rank {}",
                    shape.len(),
                    strides.len()
                ),
            });
        }
        let numel: usize = shape.iter().product();
        if numel == 0 {
            if storage_len != 0 {
                return Err(TorchError::InvalidLayout {
                    op,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    msg: format!(
                        "zero-sized tensor must have empty storage (got length {storage_len})"
                    ),
                });
            }
            return Ok(());
        }
        let mut required_len: usize = 1;
        for (&dim, &stride) in shape.iter().zip(strides.iter()) {
            if dim > 1 && stride == 0 {
                return Err(TorchError::InvalidLayout {
                    op,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    msg: "stride must be > 0 for dimensions > 1".to_string(),
                });
            }
            if dim > 0 {
                let span = (dim - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| TorchError::InvalidLayout {
                        op,
                        shape: shape.to_vec(),
                        strides: strides.to_vec(),
                        msg: "layout span overflow".to_string(),
                    })?;
                required_len = required_len.checked_add(span).ok_or_else(|| {
                    TorchError::InvalidLayout {
                        op,
                        shape: shape.to_vec(),
                        strides: strides.to_vec(),
                        msg: "layout span overflow".to_string(),
                    }
                })?;
            }
        }
        if storage_len != required_len {
            return Err(TorchError::InvalidLayout {
                op,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                msg: format!(
                    "storage length {} does not match required layout length {}",
                    storage_len, required_len
                ),
            });
        }
        Ok(())
    }

    fn validate_layout(&self, op: &'static str) -> Result<()> {
        Self::validate_layout_fields(&self.shape, &self.strides, self.storage.data.len(), op)?;
        let expected = Self::contiguous_strides(&self.shape);
        if self.strides != expected.as_slice() {
            return Err(TorchError::NonContiguous {
                op,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
            });
        }
        Ok(())
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn transpose(&self) -> Self {
        // matrix shape
        let [m, n] = self.matrix_dims();

        // result shape = [n, m]
        let mut out = TensorInner::zeros(&[n, m], self.requires_grad);

        // iterate over batches
        for ([a_mat], [out_mat_buf]) in TensorInner::iter_batches(&[&self], &mut [&mut out]) {
            let result = Matrix::from_tensor(a_mat).transpose();
            out_mat_buf.copy_from_slice(result.tensor.inner.storage.data.as_slice());
        }
        out
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn num_of_matrices(&self) -> usize {
        self.shape.iter().rev().skip(2).product::<usize>()
    }

    pub fn as_mut(&mut self) -> &mut Self {
        self
    }

    fn batch_dims(&self) -> &[usize] {
        &self.shape[..self.shape.len().saturating_sub(2)]
    }

    fn matrix_dims(&self) -> [usize; 2] {
        let dims = &self.shape[(self.shape.len()).saturating_sub(2)..];
        dims.try_into()
            .expect("Expected exactly 2 matrix dimensions")
    }

    pub(crate) fn iter_batches<'a, const N: usize, const K: usize>(
        input_tensors: &'a [&TensorInner; N],
        output_tensors: &'a mut [&mut TensorInner; K],
    ) -> impl Iterator<Item = ([Tensor; N], [&'a mut [f32]; K])> + 'a {
        assert!(
            input_tensors.len() > 0,
            "Input tensor array must not be empty."
        );
        let batch_shape = &input_tensors[0].shape[..input_tensors[0].shape.len() - 2];
        for t in input_tensors.iter() {
            assert_eq!(
                t.batch_dims(),
                batch_shape,
                "All tensors must have identical batch dimensions"
            );
        }

        let mut input_batch_iters: Vec<_> = input_tensors
            .iter()
            .map(|t| t.batch_tensor_iter(batch_shape))
            .collect();
        let mut output_batch_iters: Vec<BatchIterMut> = output_tensors
            .iter_mut()
            .map(|t| t.batch_iter_mut())
            .collect();
        (0..input_tensors[0].num_of_matrices()).map(move |_| {
            (
                array::from_fn(|i| Tensor::from_inner(input_batch_iters[i].next().unwrap())),
                array::from_fn(|i| output_batch_iters[i].next().unwrap()),
            )
        })
    }

    fn batch_range(&self, batch_index: &[usize]) -> (usize, usize) {
        assert_eq!(batch_index.len(), self.shape.len() - 2);

        // compute strides for all dimensions
        let mut strides = Vec::with_capacity(self.shape.len()); // TODO: heap allocation here is bad
        let mut stride = 1;
        for &dim in self.shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse(); // now strides[i] = product of all dims after i

        // flat offset = dot product of index with stride
        let mut offset = 0;
        for (i, &idx) in batch_index.iter().enumerate() {
            offset += idx * strides[i];
        }

        // last two dims form the matrix
        let [m, n] = self.matrix_dims();
        let size = m * n;

        (offset, offset + size)
    }

    fn get_batch(&self, batch_index: &[usize]) -> TensorInner {
        let (batch_start, batch_end) = self.batch_range(batch_index);
        let [m, n] = self.matrix_dims();

        TensorInner {
            storage: Storage {
                data: self.storage.data[batch_start..batch_end].to_vec(),
            },
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            shape: [m, n].clone().to_vec(),
            strides: TensorInner::contiguous_strides(&[m, n]),
        }
    }

    fn batch_iter_mut(&mut self) -> BatchIterMut {
        BatchIterMut {
            ptr: self.storage.data.as_mut_ptr(),
            batch_size: self.matrix_dims().iter().product(),
            num_batches: self.num_of_matrices(),
            idx: 0,
            _marker: PhantomData,
        }
    }

    pub fn batch_tensor_iter<'a>(
        &'a self,
        batch_shape: &'a [usize],
    ) -> BatchTensorIter<
        'a,
        impl Iterator<Item = Vec<usize>> + 'a,
        impl FnMut(Vec<usize>) -> TensorInner + 'a,
    > {
        let iter = batch_iter(batch_shape).map(move |batch_index| self.get_batch(&batch_index));
        BatchTensorIter {
            inner: iter,
            _marker: PhantomData,
        }
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Self::Output {
        crate::ops::matmul(self, other)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        crate::ops::add(self, other)
    }
}
