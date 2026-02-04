use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};
use crate::tensor::Tensor;

/// Immutable dataset for paired feature/target tensors.
#[derive(Debug, Clone)]
pub struct TensorDataset {
    features: Tensor,
    targets: Tensor,
    len: usize,
}

impl TensorDataset {
    /// Create a dataset from dense feature and target tensors.
    pub fn new(features: Tensor, targets: Tensor) -> Result<Self> {
        validate_pair(&features, &targets, "tensor_dataset.new")?;
        let len = features.shape()[0];
        Ok(Self {
            features,
            targets,
            len,
        })
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Access the feature tensor.
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Access the target tensor.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// Create an iterator over mini-batches.
    pub fn batch_iter(&self, batch_size: usize) -> Result<TensorBatchIter<'_>> {
        if batch_size == 0 {
            return Err(TorchError::Data {
                op: "tensor_dataset.batch_iter",
                msg: "batch_size must be > 0".to_string(),
            });
        }
        Ok(TensorBatchIter {
            dataset: self,
            batch_size,
            cursor: 0,
        })
    }
}

/// Batch payload returned by dataset iterator.
#[derive(Debug)]
pub struct TensorBatch {
    pub index: usize,
    pub features: Tensor,
    pub targets: Tensor,
}

/// Iterator that materializes contiguous batches from a dataset.
pub struct TensorBatchIter<'a> {
    dataset: &'a TensorDataset,
    batch_size: usize,
    cursor: usize,
}

impl<'a> Iterator for TensorBatchIter<'a> {
    type Item = Result<TensorBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.dataset.len {
            return None;
        }
        let start = self.cursor;
        let end = (self.cursor + self.batch_size).min(self.dataset.len);
        self.cursor = end;
        let batch_index = start / self.batch_size;

        Some(
            build_batch(self.dataset, start, end).map(|(features, targets)| TensorBatch {
                index: batch_index,
                features,
                targets,
            }),
        )
    }
}

/// Dataset for classification tasks with integer class labels.
#[derive(Debug, Clone)]
pub struct ClassificationDataset {
    features: Tensor,
    labels: Tensor,
    len: usize,
}

impl ClassificationDataset {
    /// Create a dataset from feature tensors and 1D label tensors.
    pub fn new(features: Tensor, labels: Tensor) -> Result<Self> {
        validate_classification_pair(&features, &labels, "classification_dataset.new")?;
        let len = features.shape()[0];
        Ok(Self {
            features,
            labels,
            len,
        })
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Access the feature tensor.
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Access the label tensor.
    pub fn labels(&self) -> &Tensor {
        &self.labels
    }

    /// Create an iterator over mini-batches.
    pub fn batch_iter(&self, batch_size: usize) -> Result<ClassificationBatchIter<'_>> {
        if batch_size == 0 {
            return Err(TorchError::Data {
                op: "classification_dataset.batch_iter",
                msg: "batch_size must be > 0".to_string(),
            });
        }
        Ok(ClassificationBatchIter {
            dataset: self,
            batch_size,
            cursor: 0,
        })
    }
}

/// Batch payload for classification tasks.
#[derive(Debug)]
pub struct ClassificationBatch {
    pub index: usize,
    pub features: Tensor,
    pub targets: Tensor,
}

/// Iterator that materializes batches for classification datasets.
pub struct ClassificationBatchIter<'a> {
    dataset: &'a ClassificationDataset,
    batch_size: usize,
    cursor: usize,
}

impl<'a> Iterator for ClassificationBatchIter<'a> {
    type Item = Result<ClassificationBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.dataset.len {
            return None;
        }
        let start = self.cursor;
        let end = (self.cursor + self.batch_size).min(self.dataset.len);
        self.cursor = end;
        let batch_index = start / self.batch_size;

        Some(
            build_classification_batch(self.dataset, start, end).map(|(features, targets)| {
                ClassificationBatch {
                    index: batch_index,
                    features,
                    targets,
                }
            }),
        )
    }
}

/// Configuration for generating synthetic regression data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticRegressionConfig {
    pub samples: usize,
    pub features: usize,
    pub targets: usize,
    pub noise_std: f32,
    pub seed: u64,
}

/// Synthetic regression dataset with ground-truth parameters.
#[derive(Debug, Clone)]
pub struct RegressionData {
    pub dataset: TensorDataset,
    pub true_weights: Tensor,
    pub true_bias: Tensor,
}

/// Configuration for generating synthetic classification data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticClassificationConfig {
    pub samples: usize,
    pub features: usize,
    pub classes: usize,
    pub cluster_std: f32,
    pub seed: u64,
}

/// Synthetic classification dataset with class centroids.
#[derive(Debug, Clone)]
pub struct ClassificationData {
    pub dataset: ClassificationDataset,
    pub centroids: Tensor,
}

/// Generate synthetic regression data (y = xW + b + noise).
pub fn make_synthetic_regression(config: &SyntheticRegressionConfig) -> Result<RegressionData> {
    if config.samples == 0 || config.features == 0 || config.targets == 0 {
        return Err(TorchError::Data {
            op: "make_synthetic_regression",
            msg: "samples/features/targets must be > 0".to_string(),
        });
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut weights = Vec::with_capacity(config.features * config.targets);
    for _ in 0..weights.capacity() {
        weights.push(rng.gen_range(-1.0..1.0));
    }
    let mut bias = Vec::with_capacity(config.targets);
    for _ in 0..config.targets {
        bias.push(rng.gen_range(-0.5..0.5));
    }

    let mut features = Vec::with_capacity(config.samples * config.features);
    for _ in 0..features.capacity() {
        features.push(rng.gen_range(-1.0..1.0));
    }

    let mut targets = vec![0.0f32; config.samples * config.targets];
    for i in 0..config.samples {
        for j in 0..config.targets {
            let mut acc = bias[j];
            for k in 0..config.features {
                let x = features[i * config.features + k];
                let w = weights[k * config.targets + j];
                acc += x * w;
            }
            let noise = rng.gen_range(-config.noise_std..config.noise_std);
            targets[i * config.targets + j] = acc + noise;
        }
    }

    let features_tensor =
        Tensor::from_vec_f32(features, &[config.samples, config.features], None, false);
    let targets_tensor =
        Tensor::from_vec_f32(targets, &[config.samples, config.targets], None, false);
    let dataset = TensorDataset::new(features_tensor, targets_tensor)?;
    let true_weights =
        Tensor::from_vec_f32(weights, &[config.features, config.targets], None, false);
    let true_bias = Tensor::from_vec_f32(bias, &[config.targets], None, false);
    Ok(RegressionData {
        dataset,
        true_weights,
        true_bias,
    })
}

/// Generate synthetic classification data using Gaussian clusters.
pub fn make_synthetic_classification(
    config: &SyntheticClassificationConfig,
) -> Result<ClassificationData> {
    if config.samples == 0 || config.features == 0 || config.classes == 0 {
        return Err(TorchError::Data {
            op: "make_synthetic_classification",
            msg: "samples/features/classes must be > 0".to_string(),
        });
    }
    if config.cluster_std <= 0.0 {
        return Err(TorchError::Data {
            op: "make_synthetic_classification",
            msg: "cluster_std must be > 0".to_string(),
        });
    }
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut centroids = Vec::with_capacity(config.classes * config.features);
    for _ in 0..centroids.capacity() {
        centroids.push(rng.gen_range(-1.0..1.0));
    }

    let mut features = Vec::with_capacity(config.samples * config.features);
    let mut labels = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        let class = rng.gen_range(0..config.classes);
        labels.push(class as f32);
        for j in 0..config.features {
            let mean = centroids[class * config.features + j];
            let noise = rng.gen_range(-config.cluster_std..config.cluster_std);
            features.push(mean + noise);
        }
    }

    let features_tensor =
        Tensor::from_vec_f32(features, &[config.samples, config.features], None, false);
    let labels_tensor = Tensor::from_vec_f32(labels, &[config.samples], None, false);
    let dataset = ClassificationDataset::new(features_tensor, labels_tensor)?;
    let centroids_tensor =
        Tensor::from_vec_f32(centroids, &[config.classes, config.features], None, false);
    Ok(ClassificationData {
        dataset,
        centroids: centroids_tensor,
    })
}

fn validate_pair(features: &Tensor, targets: &Tensor, op: &'static str) -> Result<()> {
    features.validate_layout(op)?;
    targets.validate_layout(op)?;
    if features.shape().len() != 2 || targets.shape().len() != 2 {
        return Err(TorchError::Data {
            op,
            msg: "features and targets must be 2D tensors".to_string(),
        });
    }
    if features.shape()[0] != targets.shape()[0] {
        return Err(TorchError::Data {
            op,
            msg: format!(
                "features and targets must share sample dimension ({} vs {})",
                features.shape()[0],
                targets.shape()[0]
            ),
        });
    }
    Ok(())
}

fn validate_classification_pair(
    features: &Tensor,
    labels: &Tensor,
    op: &'static str,
) -> Result<()> {
    features.validate_layout(op)?;
    labels.validate_layout(op)?;
    if features.shape().len() != 2 || labels.shape().len() != 1 {
        return Err(TorchError::Data {
            op,
            msg: "features must be 2D and labels must be 1D".to_string(),
        });
    }
    if features.shape()[0] != labels.shape()[0] {
        return Err(TorchError::Data {
            op,
            msg: format!(
                "features and labels must share sample dimension ({} vs {})",
                features.shape()[0],
                labels.shape()[0]
            ),
        });
    }
    Ok(())
}

fn build_batch(dataset: &TensorDataset, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
    let features = slice_rows(dataset.features(), start, end, "tensor_dataset.batch")?;
    let targets = slice_rows(dataset.targets(), start, end, "tensor_dataset.batch")?;
    Ok((features, targets))
}

fn build_classification_batch(
    dataset: &ClassificationDataset,
    start: usize,
    end: usize,
) -> Result<(Tensor, Tensor)> {
    let features = slice_rows(
        dataset.features(),
        start,
        end,
        "classification_dataset.batch",
    )?;
    let labels = slice_labels(dataset.labels(), start, end, "classification_dataset.batch")?;
    Ok((features, labels))
}

fn slice_rows(tensor: &Tensor, start: usize, end: usize, op: &'static str) -> Result<Tensor> {
    tensor.validate_layout(op)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TorchError::Data {
            op,
            msg: "slice_rows requires a 2D tensor".to_string(),
        });
    }
    if start >= end || end > shape[0] {
        return Err(TorchError::Data {
            op,
            msg: format!("invalid slice rows {start}..{end} for len {}", shape[0]),
        });
    }
    let cols = shape[1];
    let storage = tensor.storage();
    let start_idx = start * cols;
    let end_idx = end * cols;
    if end_idx > storage.data.len() {
        return Err(TorchError::Data {
            op,
            msg: "slice_rows out of storage bounds".to_string(),
        });
    }
    let data = storage.data[start_idx..end_idx].to_vec();
    Ok(Tensor::from_vec_f32(
        data,
        &[end - start, cols],
        None,
        tensor.requires_grad(),
    ))
}

fn slice_labels(tensor: &Tensor, start: usize, end: usize, op: &'static str) -> Result<Tensor> {
    tensor.validate_layout(op)?;
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(TorchError::Data {
            op,
            msg: "slice_labels requires a 1D tensor".to_string(),
        });
    }
    if start >= end || end > shape[0] {
        return Err(TorchError::Data {
            op,
            msg: format!("invalid slice rows {start}..{end} for len {}", shape[0]),
        });
    }
    let storage = tensor.storage();
    if end > storage.data.len() {
        return Err(TorchError::Data {
            op,
            msg: "slice_labels out of storage bounds".to_string(),
        });
    }
    let data = storage.data[start..end].to_vec();
    Ok(Tensor::from_vec_f32(
        data,
        &[end - start],
        None,
        tensor.requires_grad(),
    ))
}
