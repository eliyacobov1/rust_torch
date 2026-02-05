pub mod api;
pub mod audit;
pub mod autograd;
pub mod checkpoint;
pub mod data;
pub mod error;
pub mod experiment;
pub mod governance;
pub mod models;
pub mod ops;
pub mod optim;
pub mod storage;
pub mod telemetry;
pub mod tensor;
pub mod training;

#[cfg(feature = "python-bindings")]
pub mod py;

pub use audit::{
    verify_audit_log, AuditChainIssue, AuditChainIssueKind, AuditEvent, AuditLog, AuditProof,
    AuditProofDirection, AuditScope, AuditStatus, AuditVerificationConfig, AuditVerificationReport,
    AuditVerificationStatus, MerkleAccumulator,
};
pub use checkpoint::{load_state_dict, save_state_dict, StateDict};
pub use data::{
    ClassificationData, ClassificationDataset, RegressionData, SyntheticClassificationConfig,
    SyntheticRegressionConfig, TensorDataset,
};
pub use error::{Result, TorchError};
pub use experiment::{
    ComparisonEdge, CsvExportReport, ExperimentStore, LayoutSummary, MetricAggregation,
    MetricDelta, MetricStats, MetricsLogger, MetricsLoggerConfig, MetricsSummary,
    RemediationSeverity, RunComparisonConfig, RunComparisonGraph, RunComparisonReport,
    RunDeltaReport, RunDeltaSummary, RunFilter, RunGovernanceConfig, RunGovernanceReport,
    RunGovernanceSummary, RunHandle, RunMetadata, RunOverview, RunRemediationReport,
    RunRemediationTicket, RunStatus, RunSummary, RunValidationCategory, RunValidationFinding,
    RunValidationLevel, RunValidationResult, RunValidationStatus, TelemetryStats, TelemetrySummary,
};
pub use governance::{
    build_governance_schedule, DeterministicScheduler, GovernanceSchedule,
    GovernanceScheduleEntry,
};
pub use models::{LinearRegression, MlpClassifier};
pub use optim::{Optimizer, Sgd};
pub use training::{
    ClassificationReport, ClassificationTrainer, ClassificationTrainerConfig, Trainer,
    TrainerConfig, TrainingReport,
};
