use serde::{Deserialize, Serialize};

use crate::data::{
    make_synthetic_classification, make_synthetic_regression, ClassificationData,
    ClassificationDataset, RegressionData, SyntheticClassificationConfig,
    SyntheticRegressionConfig, TensorDataset,
};
use crate::error::Result;
use crate::experiment::ExperimentStore;
use crate::models::{LinearRegression, MlpClassifier};
use crate::training::{
    ClassificationReport, ClassificationTrainer, ClassificationTrainerConfig, Trainer,
    TrainerConfig, TrainingReport,
};

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

    /// Run training on a provided classification dataset using the given trainer configuration.
    pub fn train_mlp_classifier(
        &self,
        dataset: &ClassificationDataset,
        config: ClassificationTrainerConfig,
        hidden_features: usize,
        classes: usize,
        seed: u64,
    ) -> Result<ClassificationReport> {
        let mut model = MlpClassifier::new(
            dataset.features().shape()[1],
            hidden_features,
            classes,
            seed,
        )?;
        let trainer = ClassificationTrainer::new(self.store.clone(), config)?;
        trainer.train(&mut model, dataset)
    }

    /// Generate synthetic classification data and run a full training session.
    pub fn train_synthetic_classification(
        &self,
        data_config: SyntheticClassificationConfig,
        trainer_config: ClassificationTrainerConfig,
        hidden_features: usize,
        seed: u64,
    ) -> Result<SyntheticClassificationReport> {
        let ClassificationData { dataset, centroids } =
            make_synthetic_classification(&data_config)?;
        let report = self.train_mlp_classifier(
            &dataset,
            trainer_config,
            hidden_features,
            data_config.classes,
            seed,
        )?;
        Ok(SyntheticClassificationReport { report, centroids })
    }
}

/// Rich response for synthetic regression runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticTrainingReport {
    pub report: TrainingReport,
    pub true_weights: crate::tensor::Tensor,
    pub true_bias: crate::tensor::Tensor,
}

/// Rich response for synthetic classification runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticClassificationReport {
    pub report: ClassificationReport,
    pub centroids: crate::tensor::Tensor,
}
