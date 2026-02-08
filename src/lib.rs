pub mod api;
pub mod audit;
pub mod autograd;
pub mod checkpoint;
pub mod data;
pub mod error;
pub mod execution;
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
    execution_plan_digest, governance_plan_digest, record_execution_plan, record_governance_plan,
    verify_audit_log, AuditChainIssue, AuditChainIssueKind, AuditEvent, AuditLog, AuditProof,
    AuditProofDirection, AuditScope, AuditStatus, AuditVerificationConfig, AuditVerificationReport,
    AuditVerificationStatus, MerkleAccumulator, PlanDigest,
};
pub use checkpoint::{load_state_dict, save_state_dict, StateDict};
pub use data::{
    ClassificationData, ClassificationDataset, RegressionData, SyntheticClassificationConfig,
    SyntheticRegressionConfig, TensorDataset,
};
pub use error::{Result, TorchError};
pub use experiment::{
    ComparisonEdge, CsvExportReport, ExperimentStore, LayoutSummary, MetricAggregation,
    DeltaIndex, MetricDelta, MetricStats, MetricsLogger, MetricsLoggerConfig, MetricsSummary,
    RegressionGateConfig, RegressionGateFinding, RegressionGateReport, RegressionGateStatus,
    RegressionPolicy, RegressionSeverity, RemediationSeverity, RunComparisonConfig,
    RunComparisonGraph, RunComparisonReport, RunDeltaReport, RunDeltaSummary, RunFilter,
    RunGovernanceConfig, RunGovernanceReport, RunGovernanceSummary, RunHandle, RunMetadata,
    RunOverview, RunRemediationReport, RunRemediationTicket, RunStatus, RunSummary,
    RunValidationCategory, RunValidationFinding, RunValidationLevel, RunValidationResult,
    RunValidationStatus,
};
pub use execution::{
    ExecutionAction, ExecutionCostProfile, ExecutionEngine, ExecutionGraph, ExecutionLedger,
    ExecutionLedgerEvent, ExecutionLedgerIssue, ExecutionLedgerIssueKind,
    ExecutionLedgerVerificationConfig, ExecutionLedgerVerificationReport,
    ExecutionLedgerVerificationStatus, ExecutionPlan, ExecutionPlanEntry, ExecutionPlanner,
    ExecutionRegistry, ExecutionReplayCursor, ExecutionRunReport, ExecutionRunStatus,
    ExecutionStage, ExecutionStageReport, ExecutionStatus, ExecutionTask,
    verify_execution_ledger,
};
pub use governance::{
    build_governance_schedule, deterministic_run_order, DeterministicScheduler,
    GovernanceAction, GovernanceGraph, GovernanceLedger, GovernanceLedgerEvent, GovernancePlan,
    GovernancePlanEntry, GovernancePlanner, GovernanceReplayCursor, GovernanceSchedule,
    GovernanceScheduleEntry,
};
pub use models::{LinearRegression, MlpClassifier};
pub use optim::{Optimizer, Sgd};
pub use training::{
    ClassificationReport, ClassificationTrainer, ClassificationTrainerConfig, Trainer,
    TrainerConfig, TrainingReport,
};
pub use telemetry::{
    TelemetryBudget, TelemetryBudgetReport, TelemetryBudgetStatus, TelemetryEvent, TelemetryStats,
    TelemetrySummary,
};
