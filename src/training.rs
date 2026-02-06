use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use blake3::hash;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::autograd;
use crate::data::{ClassificationDataset, TensorDataset};
use crate::error::{Result, TorchError};
use crate::experiment::{ExperimentStore, MetricsLoggerConfig};
use crate::governance::{
    GovernanceAction, GovernanceGraph, GovernanceLedger, GovernanceLedgerEvent, GovernancePlan,
    GovernancePlanEntry, GovernancePlanner, GovernanceReplayCursor,
};
use crate::models::{LinearRegression, MlpClassifier};
use crate::ops;
use crate::optim::{Optimizer, Sgd};
use crate::telemetry::jsonl_recorder_from_env;
use crate::tensor::Tensor;

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

/// Training configuration for classification workloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationTrainerConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub checkpoint_every: usize,
    pub run_name: String,
    pub tags: Vec<String>,
}

impl ClassificationTrainerConfig {
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

/// Report emitted after classifier training completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub run_id: String,
    pub total_steps: usize,
    pub best_loss: f32,
    pub final_loss: f32,
    pub best_accuracy: f32,
    pub final_accuracy: f32,
}

/// Training orchestrator for experiments.
#[derive(Debug, Clone)]
pub struct Trainer {
    store: ExperimentStore,
    config: TrainerConfig,
}

#[derive(Debug)]
struct TrainingGovernance {
    cursor: GovernanceReplayCursor,
    ledger: GovernanceLedger,
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
        let start = Instant::now();
        let mut run = self.store.create_run(
            &self.config.run_name,
            self.config.to_json(),
            self.config.tags.clone(),
        )?;
        let mut governance =
            prepare_training_governance(&run, self.config.epochs, self.config.checkpoint_every)?;
        match self.train_inner(model, dataset, &mut run, &mut governance) {
            Ok(report) => {
                run.mark_completed()?;
                run.write_summary(Some(start.elapsed()))?;
                Ok(report)
            }
            Err(err) => {
                run.mark_failed(err.to_string())?;
                if let Err(summary_err) = run.write_summary(Some(start.elapsed())) {
                    warn!("failed to write run summary: {summary_err}");
                }
                Err(err)
            }
        }
    }

    fn train_inner(
        &self,
        model: &mut LinearRegression,
        dataset: &TensorDataset,
        run: &mut crate::experiment::RunHandle,
        governance: &mut TrainingGovernance,
    ) -> Result<TrainingReport> {
        let run_id = run.metadata().id.clone();
        let init_stage_id = format!("{run_id}:init");
        let init_entry = governance.cursor.expect_stage(&init_stage_id)?;
        record_governance_event(
            &mut governance.ledger,
            &init_entry,
            GovernanceAction::Started,
            "trainer_init_started",
        )?;
        let telemetry = match jsonl_recorder_from_env("RUSTORCH_RUN_TELEMETRY") {
            Ok(Some(recorder)) => Some(Arc::new(recorder)),
            Ok(None) => match run.create_telemetry_recorder() {
                Ok(recorder) => Some(Arc::new(recorder)),
                Err(err) => {
                    warn!("run telemetry disabled: {err:?}");
                    None
                }
            },
            Err(err) => {
                warn!("run telemetry disabled: {err:?}");
                None
            }
        };
        let metrics_logger =
            run.start_metrics_logger(MetricsLoggerConfig::default(), telemetry.clone())?;
        let mut optimizer = Sgd::new(self.config.learning_rate, self.config.weight_decay)?;
        let mut best_loss = f32::MAX;
        let mut final_loss = f32::MAX;
        let mut total_steps = 0usize;

        record_governance_event(
            &mut governance.ledger,
            &init_entry,
            GovernanceAction::Completed,
            "trainer_init_completed",
        )?;

        for epoch in 0..self.config.epochs {
            let epoch_stage_id = format!("{run_id}:epoch-{epoch}");
            let epoch_entry = governance.cursor.expect_stage(&epoch_stage_id)?;
            record_governance_event(
                &mut governance.ledger,
                &epoch_entry,
                GovernanceAction::Started,
                "epoch_started",
            )?;
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
                let checkpoint_stage_id = format!("{run_id}:checkpoint-{epoch}");
                let checkpoint_entry = governance.cursor.expect_stage(&checkpoint_stage_id)?;
                record_governance_event(
                    &mut governance.ledger,
                    &checkpoint_entry,
                    GovernanceAction::Started,
                    "checkpoint_started",
                )?;
                run.save_checkpoint(&checkpoint_name, &model.state_dict())?;
                record_governance_event(
                    &mut governance.ledger,
                    &checkpoint_entry,
                    GovernanceAction::Completed,
                    "checkpoint_completed",
                )?;
            }

            record_governance_event(
                &mut governance.ledger,
                &epoch_entry,
                GovernanceAction::Completed,
                "epoch_completed",
            )?;
        }

        let finalize_stage_id = format!("{run_id}:finalize");
        let finalize_entry = governance.cursor.expect_stage(&finalize_stage_id)?;
        record_governance_event(
            &mut governance.ledger,
            &finalize_entry,
            GovernanceAction::Started,
            "finalize_started",
        )?;

        if best_loss == f32::MAX {
            warn!("Training completed without recording loss values");
        }
        metrics_logger.flush()?;

        record_governance_event(
            &mut governance.ledger,
            &finalize_entry,
            GovernanceAction::Completed,
            "finalize_completed",
        )?;

        Ok(TrainingReport {
            run_id: run.metadata().id.clone(),
            total_steps,
            best_loss,
            final_loss,
        })
    }
}

/// Training orchestrator for classification experiments.
#[derive(Debug, Clone)]
pub struct ClassificationTrainer {
    store: ExperimentStore,
    config: ClassificationTrainerConfig,
}

impl ClassificationTrainer {
    /// Create a new classifier trainer bound to a persistence store.
    pub fn new(store: ExperimentStore, config: ClassificationTrainerConfig) -> Result<Self> {
        if config.epochs == 0 || config.batch_size == 0 {
            return Err(TorchError::Training {
                op: "classification_trainer.new",
                msg: "epochs and batch_size must be > 0".to_string(),
            });
        }
        Ok(Self { store, config })
    }

    /// Run the training loop and persist checkpoints/metrics.
    pub fn train(
        &self,
        model: &mut MlpClassifier,
        dataset: &ClassificationDataset,
    ) -> Result<ClassificationReport> {
        let start = Instant::now();
        let mut run = self.store.create_run(
            &self.config.run_name,
            self.config.to_json(),
            self.config.tags.clone(),
        )?;
        let mut governance =
            prepare_training_governance(&run, self.config.epochs, self.config.checkpoint_every)?;
        match self.train_inner(model, dataset, &mut run, &mut governance) {
            Ok(report) => {
                run.mark_completed()?;
                run.write_summary(Some(start.elapsed()))?;
                Ok(report)
            }
            Err(err) => {
                run.mark_failed(err.to_string())?;
                if let Err(summary_err) = run.write_summary(Some(start.elapsed())) {
                    warn!("failed to write run summary: {summary_err}");
                }
                Err(err)
            }
        }
    }

    fn train_inner(
        &self,
        model: &mut MlpClassifier,
        dataset: &ClassificationDataset,
        run: &mut crate::experiment::RunHandle,
        governance: &mut TrainingGovernance,
    ) -> Result<ClassificationReport> {
        let run_id = run.metadata().id.clone();
        let init_stage_id = format!("{run_id}:init");
        let init_entry = governance.cursor.expect_stage(&init_stage_id)?;
        record_governance_event(
            &mut governance.ledger,
            &init_entry,
            GovernanceAction::Started,
            "classification_init_started",
        )?;
        let telemetry = match jsonl_recorder_from_env("RUSTORCH_RUN_TELEMETRY") {
            Ok(Some(recorder)) => Some(Arc::new(recorder)),
            Ok(None) => match run.create_telemetry_recorder() {
                Ok(recorder) => Some(Arc::new(recorder)),
                Err(err) => {
                    warn!("run telemetry disabled: {err:?}");
                    None
                }
            },
            Err(err) => {
                warn!("run telemetry disabled: {err:?}");
                None
            }
        };
        let metrics_logger =
            run.start_metrics_logger(MetricsLoggerConfig::default(), telemetry.clone())?;
        let mut optimizer = Sgd::new(self.config.learning_rate, self.config.weight_decay)?;
        let mut best_loss = f32::MAX;
        let mut final_loss = f32::MAX;
        let mut best_accuracy = 0.0f32;
        let mut final_accuracy = 0.0f32;
        let mut total_steps = 0usize;

        record_governance_event(
            &mut governance.ledger,
            &init_entry,
            GovernanceAction::Completed,
            "classification_init_completed",
        )?;

        for epoch in 0..self.config.epochs {
            let epoch_stage_id = format!("{run_id}:epoch-{epoch}");
            let epoch_entry = governance.cursor.expect_stage(&epoch_stage_id)?;
            record_governance_event(
                &mut governance.ledger,
                &epoch_entry,
                GovernanceAction::Started,
                "classification_epoch_started",
            )?;
            let mut epoch_loss = 0.0f32;
            let mut epoch_accuracy = 0.0f32;
            let mut batches = 0usize;
            let mut batch_iter = dataset.batch_iter(self.config.batch_size)?;
            while let Some(batch_result) = batch_iter.next() {
                let batch = batch_result?;
                {
                    let mut params = model.parameters_mut();
                    optimizer.zero_grad(&mut params)?;
                }

                let log_probs = if let Some(recorder) = telemetry.as_ref() {
                    let _timer = recorder
                        .timer("classification_forward")
                        .with_tag("epoch", epoch.to_string());
                    model.forward(&batch.features)?
                } else {
                    model.forward(&batch.features)?
                };
                let loss = ops::nll_loss(&log_probs, &batch.targets);
                let loss_value =
                    *loss
                        .storage()
                        .data
                        .first()
                        .ok_or_else(|| TorchError::Training {
                            op: "classification_trainer.train",
                            msg: "loss tensor is empty".to_string(),
                        })?;
                let accuracy = batch_accuracy(&log_probs, &batch.targets)?;

                let stats = if let Some(recorder) = telemetry.as_ref() {
                    let _timer = recorder
                        .timer("classification_backward")
                        .with_tag("epoch", epoch.to_string());
                    autograd::backward(&loss)?
                } else {
                    autograd::backward(&loss)?
                };
                {
                    let mut params = model.parameters_mut();
                    optimizer.step(&mut params)?;
                }

                total_steps += 1;
                epoch_loss += loss_value;
                epoch_accuracy += accuracy;
                batches += 1;
                if loss_value < best_loss {
                    best_loss = loss_value;
                }
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                }
                final_loss = loss_value;
                final_accuracy = accuracy;

                if total_steps % self.config.log_every == 0 {
                    let mut metrics = BTreeMap::new();
                    metrics.insert("loss".to_string(), loss_value);
                    metrics.insert("accuracy".to_string(), accuracy);
                    metrics.insert("epoch".to_string(), epoch as f32);
                    metrics.insert("batch".to_string(), batch.index as f32);
                    metrics.insert("autograd_nodes".to_string(), stats.nodes as f32);
                    metrics.insert("autograd_edges".to_string(), stats.edges as f32);
                    metrics_logger.log_metrics(total_steps, metrics)?;
                }
            }

            if batches == 0 {
                return Err(TorchError::Training {
                    op: "classification_trainer.train",
                    msg: "no batches produced by dataset".to_string(),
                });
            }

            let avg_loss = epoch_loss / batches as f32;
            let avg_accuracy = epoch_accuracy / batches as f32;
            info!(
                "epoch {} completed: avg_loss={}, avg_accuracy={}, batches={}",
                epoch, avg_loss, avg_accuracy, batches
            );

            if self.config.checkpoint_every > 0 && (epoch + 1) % self.config.checkpoint_every == 0 {
                let checkpoint_name = format!("epoch_{epoch}");
                let checkpoint_stage_id = format!("{run_id}:checkpoint-{epoch}");
                let checkpoint_entry = governance.cursor.expect_stage(&checkpoint_stage_id)?;
                record_governance_event(
                    &mut governance.ledger,
                    &checkpoint_entry,
                    GovernanceAction::Started,
                    "classification_checkpoint_started",
                )?;
                run.save_checkpoint(&checkpoint_name, &model.state_dict())?;
                record_governance_event(
                    &mut governance.ledger,
                    &checkpoint_entry,
                    GovernanceAction::Completed,
                    "classification_checkpoint_completed",
                )?;
            }

            record_governance_event(
                &mut governance.ledger,
                &epoch_entry,
                GovernanceAction::Completed,
                "classification_epoch_completed",
            )?;
        }

        let finalize_stage_id = format!("{run_id}:finalize");
        let finalize_entry = governance.cursor.expect_stage(&finalize_stage_id)?;
        record_governance_event(
            &mut governance.ledger,
            &finalize_entry,
            GovernanceAction::Started,
            "classification_finalize_started",
        )?;

        if best_loss == f32::MAX {
            warn!("Training completed without recording loss values");
        }
        metrics_logger.flush()?;

        record_governance_event(
            &mut governance.ledger,
            &finalize_entry,
            GovernanceAction::Completed,
            "classification_finalize_completed",
        )?;

        Ok(ClassificationReport {
            run_id: run.metadata().id.clone(),
            total_steps,
            best_loss,
            final_loss,
            best_accuracy,
            final_accuracy,
        })
    }
}

fn batch_accuracy(log_probs: &Tensor, targets: &Tensor) -> Result<f32> {
    log_probs.validate_layout("batch_accuracy")?;
    targets.validate_layout("batch_accuracy")?;
    if log_probs.shape().len() != 2 {
        return Err(TorchError::Training {
            op: "batch_accuracy",
            msg: "log_probs must be 2D".to_string(),
        });
    }
    if targets.shape().len() != 1 {
        return Err(TorchError::Training {
            op: "batch_accuracy",
            msg: "targets must be 1D".to_string(),
        });
    }
    let batch = log_probs.shape()[0];
    let classes = log_probs.shape()[1];
    if targets.shape()[0] != batch {
        return Err(TorchError::Training {
            op: "batch_accuracy",
            msg: "targets must match batch size".to_string(),
        });
    }
    let log_data = &log_probs.storage().data;
    let target_data = &targets.storage().data;
    let mut correct = 0usize;
    for i in 0..batch {
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        let offset = i * classes;
        for j in 0..classes {
            let val = log_data[offset + j];
            if val > best_val {
                best_val = val;
                best_idx = j;
            }
        }
        let target = target_data[i] as isize;
        if target >= 0 && (target as usize) == best_idx {
            correct += 1;
        }
    }
    Ok(correct as f32 / batch as f32)
}

fn prepare_training_governance(
    run: &crate::experiment::RunHandle,
    epochs: usize,
    checkpoint_every: usize,
) -> Result<TrainingGovernance> {
    let plan = build_training_governance_plan(&run.metadata().id, epochs, checkpoint_every)?;
    run.write_governance_plan(&plan)?;
    let ledger = run.create_governance_ledger()?;
    Ok(TrainingGovernance {
        cursor: GovernanceReplayCursor::new(plan),
        ledger,
    })
}

fn build_training_governance_plan(
    run_id: &str,
    epochs: usize,
    checkpoint_every: usize,
) -> Result<GovernancePlan> {
    let seed = seed_from_run_id(run_id);
    let mut graph = GovernanceGraph::new();
    let init_stage_id = graph.add_stage(run_id.to_string(), "init", Vec::new())?;
    let mut previous = init_stage_id;
    for epoch in 0..epochs {
        let epoch_stage = format!("epoch-{epoch}");
        let epoch_id = graph.add_stage(run_id.to_string(), &epoch_stage, vec![previous.clone()])?;
        previous = epoch_id;
        if checkpoint_every > 0 && (epoch + 1) % checkpoint_every == 0 {
            let checkpoint_stage = format!("checkpoint-{epoch}");
            let checkpoint_id =
                graph.add_stage(run_id.to_string(), &checkpoint_stage, vec![previous.clone()])?;
            previous = checkpoint_id;
        }
    }
    graph.add_stage(run_id.to_string(), "finalize", vec![previous])?;
    GovernancePlanner::new(seed, graph).plan()
}

fn seed_from_run_id(run_id: &str) -> u64 {
    let hash = hash(run_id.as_bytes());
    let bytes = hash.as_bytes();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[0..8]);
    u64::from_le_bytes(buf)
}

fn record_governance_event(
    ledger: &mut GovernanceLedger,
    entry: &GovernancePlanEntry,
    action: GovernanceAction,
    message: &str,
) -> Result<()> {
    ledger
        .record(GovernanceLedgerEvent::new(
            entry.stage_id.clone(),
            entry.run_id.clone(),
            entry.stage.clone(),
            action,
            message.to_string(),
            entry.wave,
            entry.lane,
        ))
        .map(|_| ())
}
