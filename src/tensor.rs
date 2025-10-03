
use std::array;
use std::marker::PhantomData;
use crate::storage::Storage;
use crate::autograd::{GradFn, Grad};
use std::sync::{Arc, Mutex};
use std::ops::{Index, IndexMut, Mul};
pub type GradFnRef = Arc<dyn GradFn + Send + Sync>;

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let ad = *a.get(a.len().wrapping_sub(i+1)).unwrap_or(&1);
        let bd = *b.get(b.len().wrapping_sub(i+1)).unwrap_or(&1);
        if ad != bd && ad != 1 && bd != 1 {
            panic!("Shapes not broadcastable: {:?} vs {:?}", a, b);
        }
        out.push(ad.max(bd));
    }
    out.reverse();
    out
}

fn batch_iter<'a>(batch_shape: &'a[usize]) -> impl Iterator<Item = Vec<usize>> + 'a {
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
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(start),
                self.batch_size,
            )
        };

        self.idx += 1;
        Some(slice)
    }
}

pub struct BatchTensorIter<'a, I, F>
where
    I: Iterator<Item = Vec<usize>>,
    F: FnMut(Vec<usize>) -> Tensor + 'a,
{
    inner: std::iter::Map<I, F>,
    _marker: PhantomData<&'a F>,
}

impl<'a, I, F> Iterator for BatchTensorIter<'a, I, F>
where
    I: Iterator<Item = Vec<usize>>,
    F: FnMut(Vec<usize>) -> Tensor + 'a,
{
    type Item = Tensor;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub struct Tensor {
    pub storage: Storage,
    pub requires_grad: bool,
    pub grad: Option<Arc<Mutex<Grad>>>,
    pub grad_fn: Option<GradFnRef>,
    pub shape: Vec<usize>,
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
        Self {
            tensor: tensor
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        assert_eq!(tensor.shape.len(), 2);
        Self {
            tensor: tensor
        }
    }

    fn height(&self) -> usize {
        self.tensor.shape[0]
    }

    fn width(&self) -> usize {
        self.tensor.shape[1]
    }

    fn shape(&self) -> (usize, usize) {
        (self.tensor.shape[0], self.tensor.shape[1])
    }

    fn transpose(&self) -> Self {
        let (rows, cols) = self.shape();
        let mut transposed_data = Self::new(None, cols, rows, self.tensor.requires_grad);
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
        &self.tensor.storage.data[idx]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let cols = self.width();
        let idx = row * cols + col;
        &mut self.tensor.storage.data[idx]
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

impl Tensor {
    pub fn new(v: Vec<f32>, shape: &[usize], grad_fn: Option<GradFnRef>, requires_grad: bool) -> Self {
        Self {
                storage: Storage{data: v},
                requires_grad,
                grad: if requires_grad {Some(Arc::new(Mutex::new(Grad::zeros_like_shape(&shape))))} else {None},
                grad_fn,
                shape: shape.to_vec() }
    }

    pub fn from_vec_f32(v: Vec<f32>, shape: &[usize], grad_fn: Option<GradFnRef>, requires_grad: bool) -> Arc<Self> {
        Arc::new(Self::new(v, shape, grad_fn, requires_grad))
    }
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::full(shape, 0.0, requires_grad)
    }
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::full(shape, 1.0, requires_grad)
    }
    pub fn numel(&self) -> usize { 
        self.shape.iter().product() 
    }

    pub fn transpose(&self) -> Self {
        // matrix shape
        let [m, n] = self.matrix_dims();

        // result shape = [n, m]
        let mut out = Tensor::zeros(&[n, m], self.requires_grad);

        // iterate over batches
        for ([a_mat], [out_mat_buf]) in Tensor::iter_batches(&[&self], &mut[&mut out]) {
            let result = Matrix::from_tensor(a_mat).transpose();
            out_mat_buf.copy_from_slice(result.tensor.storage.data.as_slice());
        }
        out
    }

    fn full(shape: &[usize], value: f32, requires_grad: bool) -> Self {
        Self::new(vec![value; shape.iter().product()], shape, None, requires_grad)
    }
    pub fn shape(&self) -> &[usize] { &self.shape }

    pub fn num_of_matrices(&self) -> usize {
        self.shape.iter().rev().skip(2).product::<usize>()
    }

    fn batch_dims(&self) -> &[usize] {
        &self.shape[..self.shape.len().saturating_sub(2)]
    }

    fn matrix_dims(&self) -> [usize; 2] {
        let dims = &self.shape[(self.shape.len()).saturating_sub(2)..];
        dims.try_into().expect("Expected exactly 2 matrix dimensions")
    }

    fn iter_batches<'a, const N: usize, const K: usize>(
        input_tensors: &'a [&Tensor; N],
        output_tensors: &'a mut [&mut Tensor; K],
    ) -> impl Iterator<Item = ([Tensor; N], [&'a mut[f32]; K])> + 'a {
        assert!(input_tensors.len() > 0, "Input tensor array must not be empty.");
        let batch_shape = &input_tensors[0].shape[..input_tensors[0].shape.len() - 2];
        for t in input_tensors.iter() {
            assert_eq!(t.batch_dims(), batch_shape, "All tensors must have identical batch dimensions");
        }

        let mut input_batch_iters: Vec<_> = input_tensors.iter().map(|t| t.batch_tensor_iter(batch_shape)).collect();
        let mut output_batch_iters: Vec<BatchIterMut> = output_tensors.iter_mut().map(|t| t.batch_iter_mut()).collect();
        (0..input_tensors[0].num_of_matrices()).map(move |_| {
            (array::from_fn(|i| input_batch_iters[i].next().unwrap()), array::from_fn(|i| output_batch_iters[i].next().unwrap()))
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

    fn get_batch(&self, batch_index: &[usize]) -> Tensor {
        let (batch_start, batch_end) = self.batch_range(batch_index);
        let [m, n] = self.matrix_dims();
        
        Tensor {
            storage: Storage{data: self.storage.data[batch_start..batch_end].to_vec()},
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            shape: [m, n].clone().to_vec(),
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
    ) -> BatchTensorIter<'a, impl Iterator<Item = Vec<usize>> + 'a, impl FnMut(Vec<usize>) -> Tensor + 'a> {
        let iter = batch_iter(batch_shape)
            .map(move |batch_index| self.get_batch(&batch_index));
        BatchTensorIter { 
            inner: iter,
            _marker: PhantomData,
        }
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Self::Output {
        // matrix shape
        let [m, k1] = self.matrix_dims();
        let [k2, n] = other.matrix_dims();
        assert_eq!(k1, k2, "Inner dimensions must match");

        // batch shape = broadcast(a_batch, b_batch)
        let a_batch = self.batch_dims();
        let b_batch = other.batch_dims();
        let batch_shape = broadcast_shape(a_batch, b_batch);

        // result shape = batch_shape + [m, n]
        let mut out = Tensor::zeros(&[&batch_shape[..], &[m, n]].concat(), self.requires_grad || other.requires_grad);

        // iterate over batches
        for ([a_mat, b_mat], [out_mat_buf]) in Tensor::iter_batches(&[&self, &other], &mut[&mut out]) {
            let result = Matrix::from_tensor(a_mat) * Matrix::from_tensor(b_mat);
            out_mat_buf.copy_from_slice(result.tensor.storage.data.as_slice());
        }
        out
    }
}