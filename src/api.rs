use serde::{Deserialize, Serialize};

use crate::data::{
    make_synthetic_regression, RegressionData, SyntheticRegressionConfig, TensorDataset,
};
use crate::error::Result;
use crate::experiment::ExperimentStore;
use crate::models::LinearRegression;
use crate::training::{Trainer, TrainerConfig, TrainingReport};

/// API wrapper for orchestrating common training workflows.
#[derive(Debug, Clone)]
pub struct RustorchService {
    store: ExperimentStore,
}

impl RustorchService {
    /// Create a new service rooted at the experiment store path.
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Ok(Self {
            store: ExperimentStore::new(path)?,
        })
    }

    /// Run training on a provided dataset using the given trainer configuration.
    pub fn train_linear_regression(
        &self,
        dataset: &TensorDataset,
        config: TrainerConfig,
        seed: u64,
    ) -> Result<TrainingReport> {
        let mut model = LinearRegression::new(
            dataset.features().shape()[1],
            dataset.targets().shape()[1],
            seed,
        )?;
        let trainer = Trainer::new(self.store.clone(), config)?;
        trainer.train(&mut model, dataset)
    }

    /// Generate synthetic regression data and run a full training session.
    pub fn train_synthetic_regression(
        &self,
        data_config: SyntheticRegressionConfig,
        trainer_config: TrainerConfig,
        seed: u64,
    ) -> Result<SyntheticTrainingReport> {
        let RegressionData {
            dataset,
            true_weights,
            true_bias,
        } = make_synthetic_regression(&data_config)?;
        let report = self.train_linear_regression(&dataset, trainer_config, seed)?;
        Ok(SyntheticTrainingReport {
            report,
            true_weights,
            true_bias,
        })
    }
}

/// Rich response for synthetic regression runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticTrainingReport {
    pub report: TrainingReport,
    pub true_weights: crate::tensor::Tensor,
    pub true_bias: crate::tensor::Tensor,
}
