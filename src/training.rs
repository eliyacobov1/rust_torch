use std::collections::BTreeMap;
use std::sync::Arc;

use log::{info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::autograd;
use crate::data::TensorDataset;
use crate::error::{Result, TorchError};
use crate::experiment::{ExperimentStore, MetricsLoggerConfig};
use crate::models::LinearRegression;
use crate::ops;
use crate::optim::{Optimizer, Sgd};
use crate::telemetry::jsonl_recorder_from_env;

/// Training configuration for regression workloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub checkpoint_every: usize,
    pub run_name: String,
    pub tags: Vec<String>,
}

impl TrainerConfig {
    /// Serialize configuration to JSON for persistence.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "log_every": self.log_every,
            "checkpoint_every": self.checkpoint_every,
            "run_name": self.run_name,
            "tags": self.tags,
        })
    }
}

/// Report emitted after training completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub run_id: String,
    pub total_steps: usize,
    pub best_loss: f32,
    pub final_loss: f32,
}

/// Training orchestrator for experiments.
#[derive(Debug, Clone)]
pub struct Trainer {
    store: ExperimentStore,
    config: TrainerConfig,
}

impl Trainer {
    /// Create a new trainer bound to a persistence store.
    pub fn new(store: ExperimentStore, config: TrainerConfig) -> Result<Self> {
        if config.epochs == 0 || config.batch_size == 0 {
            return Err(TorchError::Training {
                op: "trainer.new",
                msg: "epochs and batch_size must be > 0".to_string(),
            });
        }
        Ok(Self { store, config })
    }

    /// Run the training loop and persist checkpoints/metrics.
    pub fn train(
        &self,
        model: &mut LinearRegression,
        dataset: &TensorDataset,
    ) -> Result<TrainingReport> {
        let mut run = self.store.create_run(
            &self.config.run_name,
            self.config.to_json(),
            self.config.tags.clone(),
        )?;
        let report = match self.train_inner(model, dataset, &mut run) {
            Ok(report) => {
                run.mark_completed()?;
                report
            }
            Err(err) => {
                run.mark_failed(err.to_string())?;
                return Err(err);
            }
        };
        Ok(report)
    }

    fn train_inner(
        &self,
        model: &mut LinearRegression,
        dataset: &TensorDataset,
        run: &mut crate::experiment::RunHandle,
    ) -> Result<TrainingReport> {
        let telemetry = match jsonl_recorder_from_env("RUSTORCH_RUN_TELEMETRY") {
            Ok(recorder) => recorder.map(Arc::new),
            Err(err) => {
                warn!("run telemetry disabled: {err:?}");
                None
            }
        };
        let metrics_logger = run.start_metrics_logger(MetricsLoggerConfig::default(), telemetry)?;
        let mut optimizer = Sgd::new(self.config.learning_rate, self.config.weight_decay)?;
        let mut best_loss = f32::MAX;
        let mut final_loss = f32::MAX;
        let mut total_steps = 0usize;

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0f32;
            let mut batches = 0usize;
            let mut batch_iter = dataset.batch_iter(self.config.batch_size)?;
            while let Some(batch_result) = batch_iter.next() {
                let batch = batch_result?;
                {
                    let mut params = model.parameters_mut();
                    optimizer.zero_grad(&mut params)?;
                }

                let preds = model.forward(&batch.features)?;
                let loss = ops::mse_loss(&preds, &batch.targets);
                let loss_value =
                    *loss
                        .storage()
                        .data
                        .first()
                        .ok_or_else(|| TorchError::Training {
                            op: "trainer.train",
                            msg: "loss tensor is empty".to_string(),
                        })?;

                let stats = autograd::backward(&loss)?;
                {
                    let mut params = model.parameters_mut();
                    optimizer.step(&mut params)?;
                }

                total_steps += 1;
                epoch_loss += loss_value;
                batches += 1;
                if loss_value < best_loss {
                    best_loss = loss_value;
                }
                final_loss = loss_value;

                if total_steps % self.config.log_every == 0 {
                    let mut metrics = BTreeMap::new();
                    metrics.insert("loss".to_string(), loss_value);
                    metrics.insert("epoch".to_string(), epoch as f32);
                    metrics.insert("batch".to_string(), batch.index as f32);
                    metrics.insert("autograd_nodes".to_string(), stats.nodes as f32);
                    metrics.insert("autograd_edges".to_string(), stats.edges as f32);
                    metrics_logger.log_metrics(total_steps, metrics)?;
                }
            }

            if batches == 0 {
                return Err(TorchError::Training {
                    op: "trainer.train",
                    msg: "no batches produced by dataset".to_string(),
                });
            }

            let avg_loss = epoch_loss / batches as f32;
            info!(
                "epoch {} completed: avg_loss={}, batches={}",
                epoch, avg_loss, batches
            );

            if self.config.checkpoint_every > 0 && (epoch + 1) % self.config.checkpoint_every == 0 {
                let checkpoint_name = format!("epoch_{epoch}");
                run.save_checkpoint(&checkpoint_name, &model.state_dict())?;
            }
        }

        if best_loss == f32::MAX {
            warn!("Training completed without recording loss values");
        }
        metrics_logger.flush()?;

        Ok(TrainingReport {
            run_id: run.metadata().id.clone(),
            total_steps,
            best_loss,
            final_loss,
        })
    }
}
