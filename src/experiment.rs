use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{info, warn};
use rand::{distributions::Alphanumeric, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::audit::{
    verify_audit_log, AuditEvent, AuditLog, AuditScope, AuditStatus, AuditVerificationConfig,
    AuditVerificationReport,
};
use crate::checkpoint::{save_state_dict, StateDict};
use crate::error::{Result, TorchError};
use crate::telemetry::{JsonlSink, TelemetryEvent, TelemetryRecorder};
use crate::tensor::{layout_stats, reset_layout_stats};

const MIN_RUN_SCHEMA_VERSION: u32 = 1;
const CURRENT_RUN_SCHEMA_VERSION: u32 = 2;

fn default_schema_version() -> u32 {
    MIN_RUN_SCHEMA_VERSION
}

/// High-level status for an experiment run stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
}

/// Metadata persisted alongside each run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub id: String,
    pub name: String,
    pub created_at_unix: u64,
    pub status: RunStatus,
    pub status_message: Option<String>,
    pub config: Value,
    pub tags: Vec<String>,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// Metric payload appended as JSONL per training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub step: usize,
    pub metrics: BTreeMap<String, f32>,
    pub timestamp_unix: u64,
}

/// Aggregate statistics for a single metric series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub count: u64,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub last: f32,
}

/// Rollup summary for metrics recorded during a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_records: u64,
    pub first_step: Option<usize>,
    pub last_step: Option<usize>,
    pub first_timestamp_unix: Option<u64>,
    pub last_timestamp_unix: Option<u64>,
    pub metrics: BTreeMap<String, MetricStats>,
}

/// Aggregate statistics for telemetry events of a given name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStats {
    pub count: u64,
    pub mean_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub max_duration_ms: f64,
}

/// Rollup summary for telemetry events recorded during a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub total_events: u64,
    pub events: BTreeMap<String, TelemetryStats>,
}

/// Rollup summary for tensor layout validation telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSummary {
    pub validations: u64,
    pub failures: u64,
    pub overlap_failures: u64,
}

/// Summary persisted after training completes for comparative analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub name: String,
    pub created_at_unix: u64,
    pub completed_at_unix: u64,
    pub status: RunStatus,
    pub status_message: Option<String>,
    pub duration_secs: Option<f64>,
    pub metrics: MetricsSummary,
    pub telemetry: Option<TelemetrySummary>,
    pub layout: LayoutSummary,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
}

/// Metric aggregation to compare summaries across runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetricAggregation {
    Min,
    Max,
    Mean,
    P50,
    P95,
    Last,
}

impl MetricAggregation {
    pub fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "mean" => Ok(Self::Mean),
            "p50" => Ok(Self::P50),
            "p95" => Ok(Self::P95),
            "last" => Ok(Self::Last),
            other => Err(TorchError::InvalidArgument {
                op: "metric_aggregation.from_str",
                msg: format!("unknown aggregation {other}"),
            }),
        }
    }

    pub fn value(self, stats: &MetricStats) -> f32 {
        match self {
            Self::Min => stats.min,
            Self::Max => stats.max,
            Self::Mean => stats.mean,
            Self::P50 => stats.p50,
            Self::P95 => stats.p95,
            Self::Last => stats.last,
        }
    }
}

/// Configuration for comparing multiple runs.
#[derive(Debug, Clone)]
pub struct RunComparisonConfig {
    pub run_ids: Vec<String>,
    pub filter: RunFilter,
    pub baseline_id: Option<String>,
    pub metric_aggregation: MetricAggregation,
    pub top_k: usize,
    pub build_graph: bool,
}

impl Default for RunComparisonConfig {
    fn default() -> Self {
        Self {
            run_ids: Vec::new(),
            filter: RunFilter::default(),
            baseline_id: None,
            metric_aggregation: MetricAggregation::Last,
            top_k: 5,
            build_graph: true,
        }
    }
}

/// Per-metric delta between the baseline and a comparison run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub metric: String,
    pub baseline: f32,
    pub candidate: f32,
    pub delta: f32,
    pub delta_pct: Option<f32>,
}

/// Aggregated delta statistics for a comparison run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunDeltaSummary {
    pub total_metrics: usize,
    pub compared_metrics: usize,
    pub missing_metrics: usize,
    pub mean_abs_delta: f32,
    pub mean_delta: f32,
}

/// Comparison output for a single run against a baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunDeltaReport {
    pub run_id: String,
    pub name: String,
    pub status: RunStatus,
    pub deltas: Vec<MetricDelta>,
    pub top_deltas: Vec<MetricDelta>,
    pub missing_metrics: Vec<String>,
    pub summary: RunDeltaSummary,
}

/// Pairwise edge describing how one run compares to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonEdge {
    pub from_run: String,
    pub to_run: String,
    pub compared_metrics: usize,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub mean_delta: f32,
}

/// Graph summary of pairwise run comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparisonGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<ComparisonEdge>,
}

/// Summary report for comparing multiple runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparisonReport {
    pub baseline_id: String,
    pub baseline_name: String,
    pub generated_at_unix: u64,
    pub metric_aggregation: MetricAggregation,
    pub comparisons: Vec<RunDeltaReport>,
    pub graph: Option<RunComparisonGraph>,
    pub top_deltas: Vec<MetricDelta>,
}

/// Filter criteria for querying runs within an experiment store.
#[derive(Debug, Clone, Default)]
pub struct RunFilter {
    pub tags: Vec<String>,
    pub statuses: Option<Vec<RunStatus>>,
}

impl RunFilter {
    pub fn matches(&self, metadata: &RunMetadata) -> bool {
        if let Some(statuses) = &self.statuses {
            if !statuses.contains(&metadata.status) {
                return false;
            }
        }
        for tag in &self.tags {
            if !metadata.tags.iter().any(|existing| existing == tag) {
                return false;
            }
        }
        true
    }
}

/// Combined metadata + optional summary for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOverview {
    pub metadata: RunMetadata,
    pub summary: Option<RunSummary>,
}

/// Outcome details for a CSV export of run summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvExportReport {
    pub output_path: PathBuf,
    pub rows: usize,
    pub bytes_written: u64,
    pub duration_ms: f64,
    pub validation_checks: usize,
    pub validation_ms: f64,
}

/// Validation severity for run governance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunValidationLevel {
    Error,
    Warning,
}

/// Classification for run governance findings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunValidationCategory {
    Metadata,
    Summary,
    Metrics,
    Telemetry,
    Artifacts,
    Schema,
    OrphanedFiles,
    Audit,
}

/// Finding emitted during run governance validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunValidationFinding {
    pub level: RunValidationLevel,
    pub category: RunValidationCategory,
    pub message: String,
}

/// Per-run validation outcome from governance checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunValidationStatus {
    Valid,
    Invalid,
}

/// Validation result for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunValidationResult {
    pub run_id: String,
    pub status: RunValidationStatus,
    pub findings: Vec<RunValidationFinding>,
    pub duration_ms: f64,
    pub checked_at_unix: u64,
    pub schema_version: Option<u32>,
    pub quarantined: bool,
    pub quarantine_path: Option<PathBuf>,
}

/// Aggregate summary for governance validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunGovernanceSummary {
    pub total_runs: usize,
    pub valid_runs: usize,
    pub invalid_runs: usize,
    pub quarantined_runs: usize,
    pub warning_count: usize,
}

/// Governance report covering an entire experiment store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunGovernanceReport {
    pub store_root: PathBuf,
    pub generated_at_unix: u64,
    pub audit_log_path: Option<PathBuf>,
    pub audit_merkle_root: Option<String>,
    pub audit_verification: Option<AuditVerificationReport>,
    pub summary: RunGovernanceSummary,
    pub results: Vec<RunValidationResult>,
    pub remediation: RunRemediationReport,
}

/// Severity for remediation tickets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RemediationSeverity {
    High,
    Medium,
    Low,
}

/// Remediation ticket generated for an invalid or quarantined run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRemediationTicket {
    pub ticket_id: String,
    pub run_id: String,
    pub severity: RemediationSeverity,
    pub findings: Vec<RunValidationFinding>,
    pub recommended_actions: Vec<String>,
    pub quarantine_path: Option<PathBuf>,
    pub created_at_unix: u64,
}

/// Aggregate remediation report for a governance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRemediationReport {
    pub total_tickets: usize,
    pub high_severity: usize,
    pub quarantined: usize,
    pub tickets: Vec<RunRemediationTicket>,
}

impl RunRemediationReport {
    fn empty() -> Self {
        Self {
            total_tickets: 0,
            high_severity: 0,
            quarantined: 0,
            tickets: Vec::new(),
        }
    }
}

/// Configuration for the governance validation pipeline.
#[derive(Debug, Clone)]
pub struct RunGovernanceConfig {
    pub max_workers: usize,
    pub strict_schema: bool,
    pub quarantine: bool,
    pub quarantine_dir: Option<PathBuf>,
    pub include_orphaned_files: bool,
    pub check_metrics: bool,
    pub check_telemetry: bool,
    pub audit_log: bool,
    pub audit_log_path: Option<PathBuf>,
    pub audit_verify: bool,
    pub audit_expected_root: Option<String>,
    pub audit_include_proofs: bool,
    pub audit_max_proofs: usize,
    pub emit_remediation: bool,
}

impl Default for RunGovernanceConfig {
    fn default() -> Self {
        let workers = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(4);
        Self {
            max_workers: workers,
            strict_schema: true,
            quarantine: false,
            quarantine_dir: None,
            include_orphaned_files: true,
            check_metrics: true,
            check_telemetry: true,
            audit_log: false,
            audit_log_path: None,
            audit_verify: true,
            audit_expected_root: None,
            audit_include_proofs: false,
            audit_max_proofs: 5,
            emit_remediation: true,
        }
    }
}

/// Configuration for asynchronous metrics logging.
#[derive(Debug, Clone)]
pub struct MetricsLoggerConfig {
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub max_queue: usize,
}

impl Default for MetricsLoggerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            flush_interval_ms: 250,
            max_queue: 1024,
        }
    }
}

enum MetricsMessage {
    Record(MetricRecord),
    Flush(mpsc::Sender<Result<()>>),
    Shutdown,
}

/// Asynchronous logger for high-frequency metrics streams.
#[derive(Debug)]
pub struct MetricsLogger {
    sender: SyncSender<MetricsMessage>,
    handle: Option<JoinHandle<()>>,
    error: Arc<Mutex<Option<TorchError>>>,
}

impl MetricsLogger {
    pub fn log_metrics(&self, step: usize, metrics: BTreeMap<String, f32>) -> Result<()> {
        self.check_error()?;
        let record = MetricRecord {
            step,
            metrics,
            timestamp_unix: now_unix_seconds()?,
        };
        self.sender
            .try_send(MetricsMessage::Record(record))
            .map_err(|err| TorchError::Experiment {
                op: "metrics_logger.log_metrics",
                msg: format!("failed to enqueue metrics: {err}"),
            })?;
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        self.check_error()?;
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(MetricsMessage::Flush(sender))
            .map_err(|err| TorchError::Experiment {
                op: "metrics_logger.flush",
                msg: format!("failed to signal flush: {err}"),
            })?;
        receiver.recv().map_err(|err| TorchError::Experiment {
            op: "metrics_logger.flush",
            msg: format!("failed to await flush: {err}"),
        })?
    }

    fn check_error(&self) -> Result<()> {
        let error = self.error.lock().map_err(|_| TorchError::Experiment {
            op: "metrics_logger.check_error",
            msg: "metrics logger error lock poisoned".to_string(),
        })?;
        if let Some(err) = error.as_ref() {
            return Err(err.clone());
        }
        Ok(())
    }
}

impl Drop for MetricsLogger {
    fn drop(&mut self) {
        let _ = self.sender.send(MetricsMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            if let Err(err) = handle.join() {
                warn!("metrics logger join failed: {err:?}");
            }
        }
    }
}

/// Artifact record referencing payloads saved inside a run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub name: String,
    pub path: String,
    pub kind: ArtifactKind,
    pub created_at_unix: u64,
    pub metadata: BTreeMap<String, String>,
}

/// Artifact types available to the persistence layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    Checkpoint,
    Dataset,
    Report,
    Other,
}

/// Persistent, file-backed store for experiments and artifacts.
#[derive(Debug, Clone)]
pub struct ExperimentStore {
    root: PathBuf,
}

impl ExperimentStore {
    /// Create or open a persistent experiment store rooted at `path`.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|err| TorchError::Experiment {
            op: "experiment_store.new",
            msg: format!("failed to create root {}: {err}", root.display()),
        })?;
        Ok(Self { root })
    }

    /// Return the root directory for this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Create a new run directory with metadata persisted to disk.
    pub fn create_run(&self, name: &str, config: Value, tags: Vec<String>) -> Result<RunHandle> {
        reset_layout_stats();
        let run_id = generate_run_id(name)?;
        let run_dir = self.root.join(&run_id);
        fs::create_dir_all(&run_dir).map_err(|err| TorchError::Experiment {
            op: "experiment_store.create_run",
            msg: format!("failed to create run dir {}: {err}", run_dir.display()),
        })?;
        let metadata = RunMetadata {
            id: run_id.clone(),
            name: name.to_string(),
            created_at_unix: now_unix_seconds()?,
            status: RunStatus::Running,
            status_message: None,
            config,
            tags,
            schema_version: CURRENT_RUN_SCHEMA_VERSION,
        };
        validate_run_metadata(&metadata)?;
        write_metadata(&run_dir, &metadata)?;
        info!("Created run {} ({})", metadata.id, metadata.name);
        Ok(RunHandle {
            store: self.clone(),
            metadata,
        })
    }

    /// List all run metadata found in the store root.
    pub fn list_runs(&self) -> Result<Vec<RunMetadata>> {
        let mut runs = Vec::new();
        for entry in fs::read_dir(&self.root).map_err(|err| TorchError::Experiment {
            op: "experiment_store.list_runs",
            msg: format!("failed to read root {}: {err}", self.root.display()),
        })? {
            let entry = entry.map_err(|err| TorchError::Experiment {
                op: "experiment_store.list_runs",
                msg: format!("failed to read directory entry: {err}"),
            })?;
            if !entry
                .file_type()
                .map_err(|err| TorchError::Experiment {
                    op: "experiment_store.list_runs",
                    msg: format!("failed to read file type: {err}"),
                })?
                .is_dir()
            {
                continue;
            }
            let run_dir = entry.path();
            match read_metadata(&run_dir) {
                Ok(metadata) => runs.push(metadata),
                Err(err) => warn!("Skipping run dir {}: {err}", run_dir.display()),
            }
        }
        runs.sort_by(|a, b| b.created_at_unix.cmp(&a.created_at_unix));
        Ok(runs)
    }

    /// Open an existing run by id.
    pub fn open_run(&self, run_id: &str) -> Result<RunHandle> {
        let run_dir = self.root.join(run_id);
        let metadata = read_metadata(&run_dir)?;
        Ok(RunHandle {
            store: self.clone(),
            metadata,
        })
    }

    /// Read a run summary by id.
    pub fn read_summary(&self, run_id: &str) -> Result<RunSummary> {
        let run_dir = self.root.join(run_id);
        read_summary(&run_dir)
    }

    /// List run metadata with optional summaries for reporting.
    pub fn list_overviews(&self, filter: &RunFilter) -> Result<Vec<RunOverview>> {
        let runs = self.list_runs()?;
        let mut overviews = Vec::new();
        for metadata in runs {
            if !filter.matches(&metadata) {
                continue;
            }
            let summary = read_summary_optional(&self.root.join(&metadata.id))?;
            overviews.push(RunOverview { metadata, summary });
        }
        Ok(overviews)
    }

    /// Export run summaries into a CSV file with dynamic metric/telemetry columns.
    pub fn export_run_summaries_csv<P: AsRef<Path>>(
        &self,
        output: P,
        filter: &RunFilter,
    ) -> Result<CsvExportReport> {
        let start = Instant::now();
        let overviews = self.list_overviews(filter)?;
        let validation_start = Instant::now();
        for overview in &overviews {
            validate_run_overview(overview)?;
        }
        let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
        let mut metric_names = BTreeSet::new();
        let mut telemetry_names = BTreeSet::new();
        for overview in &overviews {
            if let Some(summary) = &overview.summary {
                metric_names.extend(summary.metrics.metrics.keys().cloned());
                if let Some(telemetry) = &summary.telemetry {
                    telemetry_names.extend(telemetry.events.keys().cloned());
                }
            }
        }

        let output_path = output.as_ref().to_path_buf();
        let file = File::create(&output_path).map_err(|err| TorchError::Experiment {
            op: "experiment_store.export_run_summaries_csv",
            msg: format!(
                "failed to create export file {}: {err}",
                output_path.display()
            ),
        })?;
        let mut writer = BufWriter::new(file);
        let header = build_csv_header(&metric_names, &telemetry_names);
        let mut bytes_written = write_csv_row(&mut writer, &header)?;

        for overview in &overviews {
            let row = build_csv_row(overview, &metric_names, &telemetry_names);
            if row.len() != header.len() {
                return Err(TorchError::Experiment {
                    op: "experiment_store.export_run_summaries_csv",
                    msg: format!(
                        "csv row length mismatch (expected {}, got {})",
                        header.len(),
                        row.len()
                    ),
                });
            }
            bytes_written += write_csv_row(&mut writer, &row)?;
        }

        writer.flush().map_err(|err| TorchError::Experiment {
            op: "experiment_store.export_run_summaries_csv",
            msg: format!("failed to flush export {}: {err}", output_path.display()),
        })?;

        Ok(CsvExportReport {
            output_path,
            rows: overviews.len(),
            bytes_written: bytes_written as u64,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            validation_checks: overviews.len(),
            validation_ms,
        })
    }

    /// Validate runs with governance checks and optional quarantine.
    pub fn validate_runs(&self, config: &RunGovernanceConfig) -> Result<RunGovernanceReport> {
        let start = Instant::now();
        let audit_log = prepare_audit_log(config, &self.root)?;
        if let Err(err) = record_audit_event(
            &audit_log,
            AuditEvent::new(
                AuditScope::Store,
                None,
                None,
                AuditStatus::Started,
                "governance_validation_started",
                0,
            ),
        ) {
            warn!("failed to record audit start event: {err}");
        }
        let run_dirs = list_run_dirs(&self.root)?;
        let total_runs = run_dirs.len();
        if total_runs == 0 {
            if let Err(err) = record_audit_event(
                &audit_log,
                AuditEvent::new(
                    AuditScope::Store,
                    None,
                    None,
                    AuditStatus::Completed,
                    "governance_validation_completed",
                    0,
                ),
            ) {
                warn!("failed to record audit completion event: {err}");
            }
            let (audit_log_path, audit_merkle_root) = finalize_audit_log(&audit_log)?;
            let audit_verification = build_audit_verification(&audit_log_path, config);
            return Ok(RunGovernanceReport {
                store_root: self.root.clone(),
                generated_at_unix: now_unix_seconds()?,
                audit_log_path,
                audit_merkle_root,
                audit_verification,
                summary: RunGovernanceSummary {
                    total_runs: 0,
                    valid_runs: 0,
                    invalid_runs: 0,
                    quarantined_runs: 0,
                    warning_count: 0,
                },
                results: Vec::new(),
                remediation: RunRemediationReport::empty(),
            });
        }

        let worker_count = config.max_workers.max(1).min(total_runs.max(1));
        let graph = build_validation_graph(config)?;
        let mut receiver = spawn_validation_workers(
            worker_count,
            Arc::new(graph),
            config.clone(),
            self.root.clone(),
            audit_log.clone(),
        )?;

        for run_dir in run_dirs {
            receiver.send(run_dir)?;
        }
        receiver.close_sender();
        let results = receiver.collect_results()?;

        let mut valid_runs = 0;
        let mut invalid_runs = 0;
        let mut quarantined_runs = 0;
        let mut warning_count = 0;
        for result in &results {
            match result.status {
                RunValidationStatus::Valid => valid_runs += 1,
                RunValidationStatus::Invalid => invalid_runs += 1,
            }
            if result.quarantined {
                quarantined_runs += 1;
            }
            warning_count += result
                .findings
                .iter()
                .filter(|finding| finding.level == RunValidationLevel::Warning)
                .count();
        }

        let remediation = if config.emit_remediation {
            build_remediation_report(&results)?
        } else {
            RunRemediationReport::empty()
        };

        if config.emit_remediation {
            for ticket in &remediation.tickets {
                if let Err(err) = record_audit_event(
                    &audit_log,
                    AuditEvent::new(
                        AuditScope::Run,
                        Some(ticket.run_id.clone()),
                        Some("remediation".to_string()),
                        AuditStatus::Completed,
                        "run_remediation_ticket_created",
                        ticket.findings.len(),
                    ),
                ) {
                    warn!("failed to record remediation audit event: {err}");
                }
            }
        }

        info!(
            "Governance validation completed: total={}, valid={}, invalid={}, quarantined={}, duration_ms={:.2}",
            total_runs,
            valid_runs,
            invalid_runs,
            quarantined_runs,
            start.elapsed().as_secs_f64() * 1000.0
        );

        if let Err(err) = record_audit_event(
            &audit_log,
            AuditEvent::new(
                AuditScope::Store,
                None,
                None,
                AuditStatus::Completed,
                "governance_validation_completed",
                invalid_runs,
            ),
        ) {
            warn!("failed to record audit completion event: {err}");
        }
        let (audit_log_path, audit_merkle_root) = finalize_audit_log(&audit_log)?;
        let audit_verification = build_audit_verification(&audit_log_path, config);

        Ok(RunGovernanceReport {
            store_root: self.root.clone(),
            generated_at_unix: now_unix_seconds()?,
            audit_log_path,
            audit_merkle_root,
            audit_verification,
            summary: RunGovernanceSummary {
                total_runs,
                valid_runs,
                invalid_runs,
                quarantined_runs,
                warning_count,
            },
            results,
            remediation,
        })
    }

    /// Compare multiple runs and return a structured delta report.
    pub fn compare_runs(&self, config: &RunComparisonConfig) -> Result<RunComparisonReport> {
        let mut run_ids = if config.run_ids.is_empty() {
            let overviews = self.list_overviews(&config.filter)?;
            overviews
                .into_iter()
                .map(|overview| overview.metadata.id)
                .collect::<Vec<String>>()
        } else {
            config.run_ids.clone()
        };

        run_ids.sort();
        run_ids.dedup();
        if run_ids.len() < 2 {
            return Err(TorchError::InvalidArgument {
                op: "experiment_store.compare_runs",
                msg: "at least two run IDs are required for comparison".to_string(),
            });
        }

        let baseline_id = config
            .baseline_id
            .clone()
            .unwrap_or_else(|| run_ids[0].clone());
        if !run_ids.iter().any(|id| id == &baseline_id) {
            return Err(TorchError::InvalidArgument {
                op: "experiment_store.compare_runs",
                msg: format!("baseline id {baseline_id} not found in run set"),
            });
        }

        info!(
            "Comparing {} runs (baseline={})",
            run_ids.len(),
            baseline_id
        );

        let summary_results = run_ids
            .par_iter()
            .map(|run_id| (run_id.clone(), self.read_summary(run_id)))
            .collect::<Vec<(String, Result<RunSummary>)>>();

        let mut summaries = BTreeMap::new();
        for (run_id, result) in summary_results {
            let summary = result?;
            if summary.run_id != run_id {
                return Err(TorchError::Experiment {
                    op: "experiment_store.compare_runs",
                    msg: format!(
                        "run id mismatch for summary: expected {run_id}, got {}",
                        summary.run_id
                    ),
                });
            }
            summaries.insert(run_id, summary);
        }

        let baseline = summaries.get(&baseline_id).ok_or(TorchError::Experiment {
            op: "experiment_store.compare_runs",
            msg: format!("baseline summary {baseline_id} not found"),
        })?;

        if baseline.metrics.metrics.is_empty() {
            return Err(TorchError::Experiment {
                op: "experiment_store.compare_runs",
                msg: "baseline run contains no metrics to compare".to_string(),
            });
        }

        let mut comparisons = Vec::new();
        let mut global_top = TopK::new(config.top_k);
        for (run_id, summary) in &summaries {
            if run_id == &baseline_id {
                continue;
            }
            let report =
                build_run_delta_report(baseline, summary, config.metric_aggregation, config.top_k);
            for entry in &report.deltas {
                global_top.insert(entry.clone());
            }
            comparisons.push(report);
        }

        comparisons.sort_by(|a, b| {
            a.summary
                .mean_abs_delta
                .total_cmp(&b.summary.mean_abs_delta)
        });
        let graph = if config.build_graph {
            Some(build_comparison_graph(
                &summaries,
                config.metric_aggregation,
            ))
        } else {
            None
        };

        Ok(RunComparisonReport {
            baseline_id: baseline_id.clone(),
            baseline_name: baseline.name.clone(),
            generated_at_unix: now_unix_seconds()?,
            metric_aggregation: config.metric_aggregation,
            comparisons,
            graph,
            top_deltas: global_top.into_sorted_vec_desc(),
        })
    }
}

/// Handle to a live experiment run, used to record metrics and artifacts.
#[derive(Debug, Clone)]
pub struct RunHandle {
    store: ExperimentStore,
    metadata: RunMetadata,
}

impl RunHandle {
    /// Return current metadata snapshot.
    pub fn metadata(&self) -> &RunMetadata {
        &self.metadata
    }

    /// Append a metrics record to the run metrics file.
    pub fn log_metrics(&self, step: usize, metrics: BTreeMap<String, f32>) -> Result<()> {
        let record = MetricRecord {
            step,
            metrics,
            timestamp_unix: now_unix_seconds()?,
        };
        let metrics_path = self.run_dir().join("metrics.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .map_err(|err| TorchError::Experiment {
                op: "run_handle.log_metrics",
                msg: format!("failed to open metrics {}: {err}", metrics_path.display()),
            })?;
        let json = serde_json::to_string(&record).map_err(|err| TorchError::Experiment {
            op: "run_handle.log_metrics",
            msg: format!("failed to serialize metrics: {err}"),
        })?;
        writeln!(file, "{json}").map_err(|err| TorchError::Experiment {
            op: "run_handle.log_metrics",
            msg: format!("failed to append metrics: {err}"),
        })?;
        Ok(())
    }

    /// Start an asynchronous metrics logger that batches writes for this run.
    pub fn start_metrics_logger(
        &self,
        config: MetricsLoggerConfig,
        telemetry: Option<Arc<TelemetryRecorder<JsonlSink>>>,
    ) -> Result<MetricsLogger> {
        if config.batch_size == 0 || config.max_queue == 0 {
            return Err(TorchError::Experiment {
                op: "run_handle.start_metrics_logger",
                msg: "batch_size and max_queue must be > 0".to_string(),
            });
        }
        if config.flush_interval_ms == 0 {
            return Err(TorchError::Experiment {
                op: "run_handle.start_metrics_logger",
                msg: "flush_interval_ms must be > 0".to_string(),
            });
        }
        let (sender, receiver) = mpsc::sync_channel(config.max_queue);
        let error = Arc::new(Mutex::new(None));
        let handle = spawn_metrics_worker(
            self.run_dir().join("metrics.jsonl"),
            config,
            telemetry,
            Arc::clone(&error),
            receiver,
        );
        Ok(MetricsLogger {
            sender,
            handle: Some(handle),
            error,
        })
    }

    /// Create a telemetry recorder that writes into this run directory.
    pub fn create_telemetry_recorder(&self) -> Result<TelemetryRecorder<JsonlSink>> {
        let path = self.run_dir().join("telemetry.jsonl");
        JsonlSink::new(&path)
            .map(TelemetryRecorder::new)
            .map_err(|err| TorchError::Experiment {
                op: "run_handle.create_telemetry_recorder",
                msg: format!("failed to create telemetry sink {}: {err}", path.display()),
            })
    }

    /// Record an artifact entry in the run metadata directory.
    pub fn record_artifact(&self, record: ArtifactRecord) -> Result<()> {
        let artifacts_path = self.run_dir().join("artifacts.json");
        let mut records = read_artifacts(&artifacts_path)?;
        records.push(record);
        let json =
            serde_json::to_string_pretty(&records).map_err(|err| TorchError::Experiment {
                op: "run_handle.record_artifact",
                msg: format!("failed to serialize artifacts: {err}"),
            })?;
        fs::write(&artifacts_path, json).map_err(|err| TorchError::Experiment {
            op: "run_handle.record_artifact",
            msg: format!("failed to write artifacts: {err}"),
        })?;
        Ok(())
    }

    /// Save a checkpoint inside the run directory and record it as an artifact.
    pub fn save_checkpoint(&self, name: &str, state: &StateDict) -> Result<PathBuf> {
        let file_name = format!("{name}.rtch");
        let path = self.run_dir().join(&file_name);
        save_state_dict(&path, state)?;
        let mut metadata = BTreeMap::new();
        metadata.insert("format".to_string(), "RTCH".to_string());
        metadata.insert("version".to_string(), "1".to_string());
        self.record_artifact(ArtifactRecord {
            name: name.to_string(),
            path: file_name,
            kind: ArtifactKind::Checkpoint,
            created_at_unix: now_unix_seconds()?,
            metadata,
        })?;
        Ok(path)
    }

    /// Mark the run as completed.
    pub fn mark_completed(&mut self) -> Result<()> {
        self.metadata.status = RunStatus::Completed;
        self.metadata.status_message = None;
        write_metadata(&self.run_dir(), &self.metadata)?;
        info!("Run {} completed", self.metadata.id);
        Ok(())
    }

    /// Mark the run as failed with a status message.
    pub fn mark_failed(&mut self, message: String) -> Result<()> {
        self.metadata.status = RunStatus::Failed;
        self.metadata.status_message = Some(message.clone());
        write_metadata(&self.run_dir(), &self.metadata)?;
        warn!("Run {} failed: {}", self.metadata.id, message);
        Ok(())
    }

    /// Persist rollup summaries for metrics and telemetry into the run directory.
    pub fn write_summary(&self, duration: Option<Duration>) -> Result<RunSummary> {
        let run_dir = self.run_dir();
        let metrics_path = run_dir.join("metrics.jsonl");
        let telemetry_path = run_dir.join("telemetry.jsonl");
        let metrics = summarize_metrics(&metrics_path)?;
        let telemetry = summarize_telemetry(&telemetry_path)?;
        let layout = summarize_layout();
        let summary = RunSummary {
            run_id: self.metadata.id.clone(),
            name: self.metadata.name.clone(),
            created_at_unix: self.metadata.created_at_unix,
            completed_at_unix: now_unix_seconds()?,
            status: self.metadata.status.clone(),
            status_message: self.metadata.status_message.clone(),
            duration_secs: duration.map(|d| d.as_secs_f64()),
            metrics,
            telemetry,
            layout,
            schema_version: self.metadata.schema_version,
        };
        write_summary(&run_dir, &summary)?;
        Ok(summary)
    }

    fn run_dir(&self) -> PathBuf {
        self.store.root.join(&self.metadata.id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ValidationStage {
    Metadata,
    Summary,
    Artifacts,
    Metrics,
    Telemetry,
    OrphanedFiles,
}

impl ValidationStage {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Summary => "summary",
            Self::Artifacts => "artifacts",
            Self::Metrics => "metrics",
            Self::Telemetry => "telemetry",
            Self::OrphanedFiles => "orphaned_files",
        }
    }
}

#[derive(Debug, Clone)]
struct ValidationGraph {
    nodes: BTreeSet<ValidationStage>,
    edges: BTreeMap<ValidationStage, BTreeSet<ValidationStage>>,
    order: Vec<ValidationStage>,
}

impl ValidationGraph {
    fn new() -> Self {
        Self {
            nodes: BTreeSet::new(),
            edges: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    fn add_node(&mut self, node: ValidationStage) {
        self.nodes.insert(node);
        self.edges.entry(node).or_default();
    }

    fn add_edge(&mut self, from: ValidationStage, to: ValidationStage) {
        self.edges.entry(from).or_default().insert(to);
        self.nodes.insert(from);
        self.nodes.insert(to);
    }

    fn finalize_order(&mut self) -> Result<()> {
        let mut incoming = BTreeMap::new();
        for node in &self.nodes {
            incoming.insert(*node, 0usize);
        }
        for (_node, deps) in &self.edges {
            for dep in deps {
                if let Some(count) = incoming.get_mut(dep) {
                    *count += 1;
                }
            }
        }
        let mut ready = incoming
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(node, _)| *node)
            .collect::<BTreeSet<ValidationStage>>();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(node) = ready.iter().next().cloned() {
            ready.remove(&node);
            order.push(node);
            if let Some(deps) = self.edges.get(&node) {
                for dep in deps {
                    if let Some(count) = incoming.get_mut(dep) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            ready.insert(*dep);
                        }
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(TorchError::Experiment {
                op: "experiment_store.validation_graph",
                msg: "validation graph contains a cycle".to_string(),
            });
        }
        self.order = order;
        Ok(())
    }
}

struct ValidationReceiver {
    sender: Option<mpsc::Sender<PathBuf>>,
    results: Arc<Mutex<Vec<RunValidationResult>>>,
    handles: Vec<JoinHandle<()>>,
}

impl ValidationReceiver {
    fn send(&self, run_dir: PathBuf) -> Result<()> {
        let sender = self.sender.as_ref().ok_or(TorchError::Experiment {
            op: "experiment_store.validation_send",
            msg: "validation sender already closed".to_string(),
        })?;
        sender.send(run_dir).map_err(|err| TorchError::Experiment {
            op: "experiment_store.validation_send",
            msg: format!("failed to queue run for validation: {err}"),
        })
    }

    fn close_sender(&mut self) {
        self.sender.take();
    }

    fn collect_results(mut self) -> Result<Vec<RunValidationResult>> {
        self.sender.take();
        for handle in self.handles.drain(..) {
            if let Err(err) = handle.join() {
                warn!("validation worker join failed: {err:?}");
            }
        }
        let mut results = self.results.lock().map_err(|_| TorchError::Experiment {
            op: "experiment_store.collect_results",
            msg: "validation results lock poisoned".to_string(),
        })?;
        results.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        Ok(std::mem::take(&mut *results))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricDirection {
    HigherBetter,
    LowerBetter,
}

fn metric_direction(metric: &str) -> MetricDirection {
    let key = metric.to_ascii_lowercase();
    let lower_better_tokens = [
        "loss", "error", "mse", "rmse", "mae", "l1", "l2", "latency", "time",
    ];
    if lower_better_tokens.iter().any(|token| key.contains(token)) {
        MetricDirection::LowerBetter
    } else {
        MetricDirection::HigherBetter
    }
}

#[derive(Debug, Clone)]
struct DeltaEntry {
    delta_abs: f32,
    delta: MetricDelta,
}

impl PartialEq for DeltaEntry {
    fn eq(&self, other: &Self) -> bool {
        self.delta_abs.total_cmp(&other.delta_abs) == Ordering::Equal
    }
}

impl Eq for DeltaEntry {}

impl PartialOrd for DeltaEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeltaEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.delta_abs.total_cmp(&other.delta_abs)
    }
}

#[derive(Debug)]
struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<DeltaEntry>>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
        }
    }

    fn insert(&mut self, delta: MetricDelta) {
        if self.k == 0 {
            return;
        }
        let entry = DeltaEntry {
            delta_abs: delta.delta.abs(),
            delta,
        };
        if self.heap.len() < self.k {
            self.heap.push(Reverse(entry));
            return;
        }
        if let Some(Reverse(min)) = self.heap.peek() {
            if entry.delta_abs > min.delta_abs {
                self.heap.pop();
                self.heap.push(Reverse(entry));
            }
        }
    }

    fn into_sorted_vec_desc(mut self) -> Vec<MetricDelta> {
        let mut values = Vec::with_capacity(self.heap.len());
        while let Some(Reverse(entry)) = self.heap.pop() {
            values.push(entry.delta);
        }
        values.sort_by(|a, b| b.delta.abs().total_cmp(&a.delta.abs()));
        values
    }
}

fn build_run_delta_report(
    baseline: &RunSummary,
    candidate: &RunSummary,
    aggregation: MetricAggregation,
    top_k: usize,
) -> RunDeltaReport {
    let mut deltas = Vec::new();
    let mut missing_metrics = Vec::new();
    let mut abs_sum = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut compared = 0_usize;
    for (metric, baseline_stats) in &baseline.metrics.metrics {
        match candidate.metrics.metrics.get(metric) {
            Some(candidate_stats) => {
                let baseline_value = aggregation.value(baseline_stats);
                let candidate_value = aggregation.value(candidate_stats);
                let delta = candidate_value - baseline_value;
                let delta_pct = if baseline_value.abs() > f32::EPSILON {
                    Some(delta / baseline_value * 100.0)
                } else {
                    None
                };
                deltas.push(MetricDelta {
                    metric: metric.clone(),
                    baseline: baseline_value,
                    candidate: candidate_value,
                    delta,
                    delta_pct,
                });
                abs_sum += delta.abs();
                sum += delta;
                compared += 1;
            }
            None => missing_metrics.push(metric.clone()),
        }
    }

    let mut top = TopK::new(top_k);
    for delta in &deltas {
        top.insert(delta.clone());
    }
    let mean_abs_delta = if compared > 0 {
        abs_sum / compared as f32
    } else {
        0.0
    };
    let mean_delta = if compared > 0 {
        sum / compared as f32
    } else {
        0.0
    };
    RunDeltaReport {
        run_id: candidate.run_id.clone(),
        name: candidate.name.clone(),
        status: candidate.status.clone(),
        top_deltas: top.into_sorted_vec_desc(),
        missing_metrics,
        summary: RunDeltaSummary {
            total_metrics: baseline.metrics.metrics.len(),
            compared_metrics: compared,
            missing_metrics: baseline.metrics.metrics.len().saturating_sub(compared),
            mean_abs_delta,
            mean_delta,
        },
        deltas,
    }
}

fn build_comparison_graph(
    summaries: &BTreeMap<String, RunSummary>,
    aggregation: MetricAggregation,
) -> RunComparisonGraph {
    let nodes = summaries.keys().cloned().collect::<Vec<String>>();
    let mut edges = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let from_id = &nodes[i];
            let to_id = &nodes[j];
            let from = summaries.get(from_id);
            let to = summaries.get(to_id);
            if let (Some(from_summary), Some(to_summary)) = (from, to) {
                let (edge_a, edge_b) = build_pairwise_edges(from_summary, to_summary, aggregation);
                edges.push(edge_a);
                edges.push(edge_b);
            }
        }
    }
    RunComparisonGraph { nodes, edges }
}

fn build_pairwise_edges(
    from: &RunSummary,
    to: &RunSummary,
    aggregation: MetricAggregation,
) -> (ComparisonEdge, ComparisonEdge) {
    let mut compared_metrics = 0_usize;
    let mut wins = 0_usize;
    let mut losses = 0_usize;
    let mut ties = 0_usize;
    let mut delta_sum = 0.0_f32;

    for (metric, from_stats) in &from.metrics.metrics {
        if let Some(to_stats) = to.metrics.metrics.get(metric) {
            compared_metrics += 1;
            let from_value = aggregation.value(from_stats);
            let to_value = aggregation.value(to_stats);
            let directional_delta = match metric_direction(metric) {
                MetricDirection::HigherBetter => from_value - to_value,
                MetricDirection::LowerBetter => to_value - from_value,
            };
            if directional_delta > 0.0 {
                wins += 1;
            } else if directional_delta < 0.0 {
                losses += 1;
            } else {
                ties += 1;
            }
            delta_sum += directional_delta;
        }
    }

    let mean_delta = if compared_metrics > 0 {
        delta_sum / compared_metrics as f32
    } else {
        0.0
    };
    let edge_a = ComparisonEdge {
        from_run: from.run_id.clone(),
        to_run: to.run_id.clone(),
        compared_metrics,
        wins,
        losses,
        ties,
        mean_delta,
    };
    let edge_b = ComparisonEdge {
        from_run: to.run_id.clone(),
        to_run: from.run_id.clone(),
        compared_metrics,
        wins: losses,
        losses: wins,
        ties,
        mean_delta: -mean_delta,
    };
    (edge_a, edge_b)
}

fn build_validation_graph(config: &RunGovernanceConfig) -> Result<ValidationGraph> {
    let mut graph = ValidationGraph::new();
    graph.add_node(ValidationStage::Metadata);
    graph.add_node(ValidationStage::Summary);
    graph.add_node(ValidationStage::Artifacts);
    if config.check_metrics {
        graph.add_node(ValidationStage::Metrics);
    }
    if config.check_telemetry {
        graph.add_node(ValidationStage::Telemetry);
    }
    if config.include_orphaned_files {
        graph.add_node(ValidationStage::OrphanedFiles);
    }

    graph.add_edge(ValidationStage::Metadata, ValidationStage::Summary);
    graph.add_edge(ValidationStage::Metadata, ValidationStage::Artifacts);
    if config.check_metrics {
        graph.add_edge(ValidationStage::Metadata, ValidationStage::Metrics);
    }
    if config.check_telemetry {
        graph.add_edge(ValidationStage::Metadata, ValidationStage::Telemetry);
    }
    if config.include_orphaned_files {
        graph.add_edge(ValidationStage::Artifacts, ValidationStage::OrphanedFiles);
        graph.add_edge(ValidationStage::Metadata, ValidationStage::OrphanedFiles);
    }

    graph.finalize_order()?;
    Ok(graph)
}

fn spawn_validation_workers(
    worker_count: usize,
    graph: Arc<ValidationGraph>,
    config: RunGovernanceConfig,
    store_root: PathBuf,
    audit_log: Option<Arc<Mutex<AuditLog>>>,
) -> Result<ValidationReceiver> {
    let (sender, receiver) = mpsc::channel::<PathBuf>();
    let receiver = Arc::new(Mutex::new(receiver));
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::with_capacity(worker_count);
    for worker_id in 0..worker_count {
        let receiver = Arc::clone(&receiver);
        let results = Arc::clone(&results);
        let graph = Arc::clone(&graph);
        let config = config.clone();
        let store_root = store_root.clone();
        let audit_log = audit_log.clone();
        let handle = thread::Builder::new()
            .name(format!("run-validate-{worker_id}"))
            .spawn(move || loop {
                let next = {
                    let lock = receiver.lock();
                    match lock {
                        Ok(guard) => guard.recv(),
                        Err(_) => {
                            warn!("validation receiver lock poisoned");
                            break;
                        }
                    }
                };
                let run_dir = match next {
                    Ok(dir) => dir,
                    Err(_) => break,
                };
                let result = validate_run_dir(&store_root, &run_dir, &config, &graph, &audit_log);
                let mut guard = match results.lock() {
                    Ok(guard) => guard,
                    Err(_) => {
                        warn!("validation results lock poisoned");
                        break;
                    }
                };
                guard.push(result);
            })
            .map_err(|err| TorchError::Experiment {
                op: "experiment_store.spawn_validation_workers",
                msg: format!("failed to spawn worker: {err}"),
            })?;
        handles.push(handle);
    }
    Ok(ValidationReceiver {
        sender: Some(sender),
        results,
        handles,
    })
}

fn validate_run_dir(
    store_root: &Path,
    run_dir: &Path,
    config: &RunGovernanceConfig,
    graph: &ValidationGraph,
    audit_log: &Option<Arc<Mutex<AuditLog>>>,
) -> RunValidationResult {
    let start = Instant::now();
    let run_id = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();
    let mut findings = Vec::new();
    let mut metadata: Option<RunMetadata> = None;
    let mut summary: Option<RunSummary> = None;
    let mut schema_version = None;

    if let Err(err) = record_audit_event(
        audit_log,
        AuditEvent::new(
            AuditScope::Run,
            Some(run_id.clone()),
            None,
            AuditStatus::Started,
            "run_validation_started",
            0,
        ),
    ) {
        findings.push(RunValidationFinding {
            level: RunValidationLevel::Warning,
            category: RunValidationCategory::Audit,
            message: format!("audit log failure (run start): {err}"),
        });
    }

    for stage in &graph.order {
        let before_len = findings.len();
        match stage {
            ValidationStage::Metadata => match read_metadata(run_dir) {
                Ok(value) => {
                    if value.schema_version < MIN_RUN_SCHEMA_VERSION
                        || value.schema_version > CURRENT_RUN_SCHEMA_VERSION
                    {
                        findings.push(RunValidationFinding {
                            level: RunValidationLevel::Error,
                            category: RunValidationCategory::Schema,
                            message: format!(
                                "metadata schema version {} outside supported range {}-{}",
                                value.schema_version,
                                MIN_RUN_SCHEMA_VERSION,
                                CURRENT_RUN_SCHEMA_VERSION
                            ),
                        });
                    }
                    schema_version = Some(value.schema_version);
                    metadata = Some(value);
                }
                Err(err) => {
                    findings.push(RunValidationFinding {
                        level: RunValidationLevel::Error,
                        category: RunValidationCategory::Metadata,
                        message: format!("failed to read metadata: {err}"),
                    });
                    break;
                }
            },
            ValidationStage::Summary => {
                let Some(meta) = metadata.as_ref() else {
                    findings.push(RunValidationFinding {
                        level: RunValidationLevel::Error,
                        category: RunValidationCategory::Summary,
                        message: "summary validation skipped: metadata missing".to_string(),
                    });
                    continue;
                };
                match read_summary_optional(run_dir) {
                    Ok(value) => {
                        if let Some(summary_value) = &value {
                            if summary_value.schema_version < MIN_RUN_SCHEMA_VERSION
                                || summary_value.schema_version > CURRENT_RUN_SCHEMA_VERSION
                            {
                                findings.push(RunValidationFinding {
                                    level: RunValidationLevel::Error,
                                    category: RunValidationCategory::Schema,
                                    message: format!(
                                        "summary schema version {} outside supported range {}-{}",
                                        summary_value.schema_version,
                                        MIN_RUN_SCHEMA_VERSION,
                                        CURRENT_RUN_SCHEMA_VERSION
                                    ),
                                });
                            }
                            if summary_value.schema_version != meta.schema_version {
                                findings.push(RunValidationFinding {
                                    level: RunValidationLevel::Warning,
                                    category: RunValidationCategory::Schema,
                                    message: format!(
                                        "summary schema version {} does not match metadata {}",
                                        summary_value.schema_version, meta.schema_version
                                    ),
                                });
                            }
                        }
                        if value.is_none() && meta.status == RunStatus::Completed {
                            let level = if config.strict_schema {
                                RunValidationLevel::Error
                            } else {
                                RunValidationLevel::Warning
                            };
                            findings.push(RunValidationFinding {
                                level,
                                category: RunValidationCategory::Summary,
                                message: "completed run missing run_summary.json".to_string(),
                            });
                        }
                        summary = value;
                    }
                    Err(err) => {
                        findings.push(RunValidationFinding {
                            level: RunValidationLevel::Error,
                            category: RunValidationCategory::Summary,
                            message: format!("failed to read summary: {err}"),
                        });
                    }
                }
            }
            ValidationStage::Artifacts => match validate_artifacts(run_dir) {
                Ok(mut artifact_findings) => findings.append(&mut artifact_findings),
                Err(err) => findings.push(RunValidationFinding {
                    level: RunValidationLevel::Error,
                    category: RunValidationCategory::Artifacts,
                    message: format!("failed to read artifacts: {err}"),
                }),
            },
            ValidationStage::Metrics => {
                if config.check_metrics {
                    if let Err(err) = validate_metrics_file(run_dir) {
                        findings.push(RunValidationFinding {
                            level: RunValidationLevel::Error,
                            category: RunValidationCategory::Metrics,
                            message: format!("metrics validation failed: {err}"),
                        });
                    }
                }
            }
            ValidationStage::Telemetry => {
                if config.check_telemetry {
                    if let Err(err) = validate_telemetry_file(run_dir) {
                        findings.push(RunValidationFinding {
                            level: RunValidationLevel::Error,
                            category: RunValidationCategory::Telemetry,
                            message: format!("telemetry validation failed: {err}"),
                        });
                    }
                }
            }
            ValidationStage::OrphanedFiles => {
                if config.include_orphaned_files {
                    match find_orphaned_files(run_dir) {
                        Ok(orphaned) if !orphaned.is_empty() => {
                            let level = if config.strict_schema {
                                RunValidationLevel::Error
                            } else {
                                RunValidationLevel::Warning
                            };
                            findings.push(RunValidationFinding {
                                level,
                                category: RunValidationCategory::OrphanedFiles,
                                message: format!("orphaned files present: {}", orphaned.join(", ")),
                            });
                        }
                        Ok(_) => {}
                        Err(err) => findings.push(RunValidationFinding {
                            level: RunValidationLevel::Error,
                            category: RunValidationCategory::OrphanedFiles,
                            message: format!("failed to scan run dir: {err}"),
                        }),
                    }
                }
            }
        }

        let stage_findings = &findings[before_len..];
        let stage_status = if stage_findings
            .iter()
            .any(|finding| finding.level == RunValidationLevel::Error)
        {
            AuditStatus::Failed
        } else {
            AuditStatus::Completed
        };
        if let Err(err) = record_audit_event(
            audit_log,
            AuditEvent::new(
                AuditScope::Run,
                Some(run_id.clone()),
                Some(stage.as_str().to_string()),
                stage_status,
                "run_validation_stage",
                stage_findings.len(),
            ),
        ) {
            findings.push(RunValidationFinding {
                level: RunValidationLevel::Warning,
                category: RunValidationCategory::Audit,
                message: format!("audit log failure (stage {}): {err}", stage.as_str()),
            });
        }
    }

    if let (Some(meta), Some(summary)) = (metadata.as_ref(), summary.as_ref()) {
        if meta.id != summary.run_id {
            findings.push(RunValidationFinding {
                level: RunValidationLevel::Error,
                category: RunValidationCategory::Schema,
                message: format!(
                    "metadata id {} does not match summary {}",
                    meta.id, summary.run_id
                ),
            });
        }
    }

    let status = if findings
        .iter()
        .any(|finding| finding.level == RunValidationLevel::Error)
    {
        RunValidationStatus::Invalid
    } else {
        RunValidationStatus::Valid
    };

    let mut quarantine_path = None;
    let mut quarantined = false;
    if status == RunValidationStatus::Invalid && config.quarantine {
        let root = config
            .quarantine_dir
            .clone()
            .unwrap_or_else(|| store_root.join("quarantine"));
        match quarantine_run(run_dir, &root) {
            Ok(path) => {
                quarantine_path = Some(path);
                quarantined = true;
            }
            Err(err) => findings.push(RunValidationFinding {
                level: RunValidationLevel::Error,
                category: RunValidationCategory::Schema,
                message: format!("failed to quarantine run: {err}"),
            }),
        }
    }

    if let Err(err) = record_audit_event(
        audit_log,
        AuditEvent::new(
            AuditScope::Run,
            Some(run_id.clone()),
            None,
            match status {
                RunValidationStatus::Valid => AuditStatus::Completed,
                RunValidationStatus::Invalid => AuditStatus::Failed,
            },
            "run_validation_completed",
            findings.len(),
        ),
    ) {
        findings.push(RunValidationFinding {
            level: RunValidationLevel::Warning,
            category: RunValidationCategory::Audit,
            message: format!("audit log failure (run completed): {err}"),
        });
    }

    if quarantined {
        if let Err(err) = record_audit_event(
            audit_log,
            AuditEvent::new(
                AuditScope::Run,
                Some(run_id.clone()),
                Some("quarantine".to_string()),
                AuditStatus::Completed,
                "run_quarantined",
                0,
            ),
        ) {
            findings.push(RunValidationFinding {
                level: RunValidationLevel::Warning,
                category: RunValidationCategory::Audit,
                message: format!("audit log failure (quarantine): {err}"),
            });
        }
    }

    RunValidationResult {
        run_id,
        status,
        findings,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        checked_at_unix: now_unix_seconds().unwrap_or_default(),
        schema_version,
        quarantined,
        quarantine_path,
    }
}

fn prepare_audit_log(
    config: &RunGovernanceConfig,
    store_root: &Path,
) -> Result<Option<Arc<Mutex<AuditLog>>>> {
    if !config.audit_log {
        return Ok(None);
    }
    let path = config
        .audit_log_path
        .clone()
        .unwrap_or_else(|| store_root.join("audit").join("run_governance_audit.jsonl"));
    let log = AuditLog::open(path)?;
    Ok(Some(Arc::new(Mutex::new(log))))
}

fn record_audit_event(
    audit_log: &Option<Arc<Mutex<AuditLog>>>,
    event: AuditEvent,
) -> Result<Option<AuditEvent>> {
    let Some(log) = audit_log else {
        return Ok(None);
    };
    let mut guard = log.lock().map_err(|_| TorchError::Experiment {
        op: "experiment_store.record_audit_event",
        msg: "audit log lock poisoned".to_string(),
    })?;
    let recorded = guard.record(event)?;
    Ok(Some(recorded))
}

fn finalize_audit_log(
    audit_log: &Option<Arc<Mutex<AuditLog>>>,
) -> Result<(Option<PathBuf>, Option<String>)> {
    let Some(log) = audit_log else {
        return Ok((None, None));
    };
    let guard = log.lock().map_err(|_| TorchError::Experiment {
        op: "experiment_store.finalize_audit_log",
        msg: "audit log lock poisoned".to_string(),
    })?;
    Ok((Some(guard.path().to_path_buf()), guard.merkle_root_hex()))
}

fn build_audit_verification(
    audit_log_path: &Option<PathBuf>,
    config: &RunGovernanceConfig,
) -> Option<AuditVerificationReport> {
    if !config.audit_verify {
        return None;
    }
    let Some(path) = audit_log_path else {
        return None;
    };
    let mut verify_config = AuditVerificationConfig::default();
    verify_config.include_proofs = config.audit_include_proofs;
    verify_config.max_proofs = config.audit_max_proofs;
    verify_config.expected_root = config.audit_expected_root.clone();
    match verify_audit_log(path, &verify_config) {
        Ok(report) => Some(report),
        Err(err) => Some(AuditVerificationReport::from_error(format!(
            "audit verification failed: {err}"
        ))),
    }
}

fn list_run_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut runs = Vec::new();
    for entry in fs::read_dir(root).map_err(|err| TorchError::Experiment {
        op: "experiment_store.list_run_dirs",
        msg: format!("failed to read root {}: {err}", root.display()),
    })? {
        let entry = entry.map_err(|err| TorchError::Experiment {
            op: "experiment_store.list_run_dirs",
            msg: format!("failed to read dir entry: {err}"),
        })?;
        let file_type = entry.file_type().map_err(|err| TorchError::Experiment {
            op: "experiment_store.list_run_dirs",
            msg: format!("failed to read dir entry type: {err}"),
        })?;
        if !file_type.is_dir() {
            continue;
        }
        let run_dir = entry.path();
        match read_metadata(&run_dir) {
            Ok(_) => runs.push(run_dir),
            Err(err) => warn!("Skipping non-run dir {}: {err}", run_dir.display()),
        }
    }
    runs.sort();
    Ok(runs)
}

fn quarantine_run(run_dir: &Path, quarantine_root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(quarantine_root).map_err(|err| TorchError::Experiment {
        op: "experiment_store.quarantine_run",
        msg: format!(
            "failed to create quarantine dir {}: {err}",
            quarantine_root.display()
        ),
    })?;
    let run_name = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("run");
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(4)
        .map(char::from)
        .collect();
    let target = quarantine_root.join(format!(
        "{}-{}-{}",
        run_name,
        now_unix_seconds().unwrap_or_default(),
        suffix
    ));
    fs::rename(run_dir, &target).map_err(|err| TorchError::Experiment {
        op: "experiment_store.quarantine_run",
        msg: format!(
            "failed to move {} to quarantine {}: {err}",
            run_dir.display(),
            target.display()
        ),
    })?;
    Ok(target)
}

fn build_remediation_report(results: &[RunValidationResult]) -> Result<RunRemediationReport> {
    let mut tickets = Vec::new();
    let mut high_severity = 0;
    let mut quarantined = 0;
    for result in results {
        if result.status == RunValidationStatus::Valid {
            continue;
        }
        let severity = remediation_severity(result);
        if severity == RemediationSeverity::High {
            high_severity += 1;
        }
        if result.quarantined {
            quarantined += 1;
        }
        let ticket_id = format!("remediate-{}-{}", result.run_id, now_unix_seconds()?);
        let actions = remediation_actions(&result.findings, result.quarantined);
        tickets.push(RunRemediationTicket {
            ticket_id,
            run_id: result.run_id.clone(),
            severity,
            findings: result.findings.clone(),
            recommended_actions: actions,
            quarantine_path: result.quarantine_path.clone(),
            created_at_unix: now_unix_seconds()?,
        });
    }
    Ok(RunRemediationReport {
        total_tickets: tickets.len(),
        high_severity,
        quarantined,
        tickets,
    })
}

fn remediation_severity(result: &RunValidationResult) -> RemediationSeverity {
    if result.quarantined {
        return RemediationSeverity::High;
    }
    let has_error = result
        .findings
        .iter()
        .any(|finding| finding.level == RunValidationLevel::Error);
    if !has_error {
        return RemediationSeverity::Low;
    }
    let critical_categories = [
        RunValidationCategory::Schema,
        RunValidationCategory::Audit,
        RunValidationCategory::Artifacts,
    ];
    if result.findings.iter().any(|finding| {
        finding.level == RunValidationLevel::Error
            && critical_categories.contains(&finding.category)
    }) {
        RemediationSeverity::High
    } else {
        RemediationSeverity::Medium
    }
}

fn remediation_actions(findings: &[RunValidationFinding], quarantined: bool) -> Vec<String> {
    let mut actions = BTreeSet::new();
    if quarantined {
        actions.insert(
            "Inspect quarantined run contents and restore to the main store after remediation."
                .to_string(),
        );
    }
    for finding in findings {
        match finding.category {
            RunValidationCategory::Schema => {
                actions.insert(
                    "Regenerate run.json and run_summary.json with a supported schema version."
                        .to_string(),
                );
            }
            RunValidationCategory::Metadata => {
                actions.insert(
                    "Repair run metadata: ensure status, timestamps, and tags are consistent."
                        .to_string(),
                );
            }
            RunValidationCategory::Summary => {
                actions.insert(
                    "Recompute run summary statistics and write run_summary.json.".to_string(),
                );
            }
            RunValidationCategory::Metrics => {
                actions.insert("Recompute metrics.jsonl with valid JSONL payloads.".to_string());
            }
            RunValidationCategory::Telemetry => {
                actions.insert("Re-export telemetry.jsonl with valid JSONL payloads.".to_string());
            }
            RunValidationCategory::Artifacts => {
                actions.insert(
                    "Restore missing artifacts or update artifacts.json references.".to_string(),
                );
            }
            RunValidationCategory::OrphanedFiles => {
                actions.insert("Remove orphaned files or add them to artifacts.json.".to_string());
            }
            RunValidationCategory::Audit => {
                actions
                    .insert("Re-run governance validation with audit logging enabled.".to_string());
            }
        }
    }
    actions.into_iter().collect()
}

fn validate_artifacts(run_dir: &Path) -> Result<Vec<RunValidationFinding>> {
    let artifacts_path = run_dir.join("artifacts.json");
    if !artifacts_path.exists() {
        return Ok(Vec::new());
    }
    let records = read_artifacts(&artifacts_path)?;
    let mut findings = Vec::new();
    for record in records {
        let artifact_path = run_dir.join(&record.path);
        if !artifact_path.exists() {
            findings.push(RunValidationFinding {
                level: RunValidationLevel::Error,
                category: RunValidationCategory::Artifacts,
                message: format!(
                    "artifact {} missing at {}",
                    record.name,
                    artifact_path.display()
                ),
            });
        }
    }
    Ok(findings)
}

fn validate_metrics_file(run_dir: &Path) -> Result<()> {
    let metrics_path = run_dir.join("metrics.jsonl");
    if !metrics_path.exists() {
        return Ok(());
    }
    let file = File::open(&metrics_path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.validate_metrics_file",
        msg: format!("failed to open metrics {}: {err}", metrics_path.display()),
    })?;
    let reader = std::io::BufReader::new(file);
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| TorchError::Experiment {
            op: "experiment_store.validate_metrics_file",
            msg: format!("failed to read metrics line: {err}"),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        serde_json::from_str::<MetricRecord>(&line).map_err(|err| TorchError::Experiment {
            op: "experiment_store.validate_metrics_file",
            msg: format!("invalid metrics json at line {}: {err}", idx + 1),
        })?;
    }
    Ok(())
}

fn validate_telemetry_file(run_dir: &Path) -> Result<()> {
    let telemetry_path = run_dir.join("telemetry.jsonl");
    if !telemetry_path.exists() {
        return Ok(());
    }
    let file = File::open(&telemetry_path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.validate_telemetry_file",
        msg: format!(
            "failed to open telemetry {}: {err}",
            telemetry_path.display()
        ),
    })?;
    let reader = std::io::BufReader::new(file);
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| TorchError::Experiment {
            op: "experiment_store.validate_telemetry_file",
            msg: format!("failed to read telemetry line: {err}"),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        serde_json::from_str::<TelemetryEvent>(&line).map_err(|err| TorchError::Experiment {
            op: "experiment_store.validate_telemetry_file",
            msg: format!("invalid telemetry json at line {}: {err}", idx + 1),
        })?;
    }
    Ok(())
}

fn find_orphaned_files(run_dir: &Path) -> Result<Vec<String>> {
    let mut allowed = BTreeSet::new();
    allowed.insert("run.json".to_string());
    allowed.insert("run_summary.json".to_string());
    allowed.insert("metrics.jsonl".to_string());
    allowed.insert("telemetry.jsonl".to_string());
    allowed.insert("artifacts.json".to_string());

    let artifacts_path = run_dir.join("artifacts.json");
    if artifacts_path.exists() {
        if let Ok(records) = read_artifacts(&artifacts_path) {
            for record in records {
                allowed.insert(record.path);
            }
        }
    }

    let mut orphaned = Vec::new();
    for entry in fs::read_dir(run_dir).map_err(|err| TorchError::Experiment {
        op: "experiment_store.find_orphaned_files",
        msg: format!("failed to read run dir {}: {err}", run_dir.display()),
    })? {
        let entry = entry.map_err(|err| TorchError::Experiment {
            op: "experiment_store.find_orphaned_files",
            msg: format!("failed to read run dir entry: {err}"),
        })?;
        let file_type = entry.file_type().map_err(|err| TorchError::Experiment {
            op: "experiment_store.find_orphaned_files",
            msg: format!("failed to read run dir entry type: {err}"),
        })?;
        if !file_type.is_file() {
            let name = entry.file_name().to_string_lossy().to_string();
            orphaned.push(name);
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !allowed.contains(&name) {
            orphaned.push(name);
        }
    }
    orphaned.sort();
    Ok(orphaned)
}

fn generate_run_id(name: &str) -> Result<String> {
    let now = now_unix_seconds()?;
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(6)
        .map(char::from)
        .collect();
    let slug = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    Ok(format!("{slug}-{now}-{suffix}"))
}

fn now_unix_seconds() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|err| TorchError::Experiment {
            op: "time",
            msg: format!("system time error: {err}"),
        })
}

fn metadata_path(run_dir: &Path) -> PathBuf {
    run_dir.join("run.json")
}

fn summary_path(run_dir: &Path) -> PathBuf {
    run_dir.join("run_summary.json")
}

fn write_metadata(run_dir: &Path, metadata: &RunMetadata) -> Result<()> {
    validate_run_metadata(metadata)?;
    let json = serde_json::to_string_pretty(metadata).map_err(|err| TorchError::Experiment {
        op: "experiment_store.write_metadata",
        msg: format!("failed to serialize metadata: {err}"),
    })?;
    fs::write(metadata_path(run_dir), json).map_err(|err| TorchError::Experiment {
        op: "experiment_store.write_metadata",
        msg: format!("failed to write metadata: {err}"),
    })?;
    Ok(())
}

fn write_summary(run_dir: &Path, summary: &RunSummary) -> Result<()> {
    validate_run_summary(summary)?;
    let json = serde_json::to_string_pretty(summary).map_err(|err| TorchError::Experiment {
        op: "experiment_store.write_summary",
        msg: format!("failed to serialize summary: {err}"),
    })?;
    fs::write(summary_path(run_dir), json).map_err(|err| TorchError::Experiment {
        op: "experiment_store.write_summary",
        msg: format!("failed to write summary: {err}"),
    })?;
    Ok(())
}

fn read_metadata(run_dir: &Path) -> Result<RunMetadata> {
    let path = metadata_path(run_dir);
    let mut file = File::open(&path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_metadata",
        msg: format!("failed to open {}: {err}", path.display()),
    })?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|err| TorchError::Experiment {
            op: "experiment_store.read_metadata",
            msg: format!("failed to read {}: {err}", path.display()),
        })?;
    let metadata: RunMetadata =
        serde_json::from_str(&buf).map_err(|err| TorchError::Experiment {
            op: "experiment_store.read_metadata",
            msg: format!("failed to parse metadata: {err}"),
        })?;
    validate_run_metadata(&metadata)?;
    Ok(metadata)
}

fn read_artifacts(path: &Path) -> Result<Vec<ArtifactRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let buf = fs::read_to_string(path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_artifacts",
        msg: format!("failed to read {}: {err}", path.display()),
    })?;
    serde_json::from_str(&buf).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_artifacts",
        msg: format!("failed to parse artifacts: {err}"),
    })
}

fn read_summary(run_dir: &Path) -> Result<RunSummary> {
    let path = summary_path(run_dir);
    let buf = fs::read_to_string(&path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_summary",
        msg: format!("failed to read {}: {err}", path.display()),
    })?;
    let summary: RunSummary = serde_json::from_str(&buf).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_summary",
        msg: format!("failed to parse summary: {err}"),
    })?;
    validate_run_summary(&summary)?;
    Ok(summary)
}

fn read_summary_optional(run_dir: &Path) -> Result<Option<RunSummary>> {
    let path = summary_path(run_dir);
    if !path.exists() {
        return Ok(None);
    }
    read_summary(run_dir).map(Some)
}

fn validate_run_metadata(metadata: &RunMetadata) -> Result<()> {
    if metadata.id.trim().is_empty() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: "metadata.id is required".to_string(),
        });
    }
    if metadata.name.trim().is_empty() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: "metadata.name is required".to_string(),
        });
    }
    if metadata.created_at_unix == 0 {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: "metadata.created_at_unix must be non-zero".to_string(),
        });
    }
    if let RunStatus::Failed = metadata.status {
        if metadata
            .status_message
            .as_ref()
            .map(String::as_str)
            .unwrap_or("")
            .is_empty()
        {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_metadata",
                msg: "metadata.status_message required when status is failed".to_string(),
            });
        }
    }
    if !metadata.config.is_object() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: "metadata.config must be a JSON object".to_string(),
        });
    }
    if metadata.tags.iter().any(|tag| tag.trim().is_empty()) {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: "metadata.tags cannot contain empty values".to_string(),
        });
    }
    if metadata.schema_version < MIN_RUN_SCHEMA_VERSION
        || metadata.schema_version > CURRENT_RUN_SCHEMA_VERSION
    {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metadata",
            msg: format!(
                "metadata schema version {} outside supported range {}-{}",
                metadata.schema_version, MIN_RUN_SCHEMA_VERSION, CURRENT_RUN_SCHEMA_VERSION
            ),
        });
    }
    Ok(())
}

fn validate_run_summary(summary: &RunSummary) -> Result<()> {
    if summary.run_id.trim().is_empty() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_summary",
            msg: "summary.run_id is required".to_string(),
        });
    }
    if summary.name.trim().is_empty() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_summary",
            msg: "summary.name is required".to_string(),
        });
    }
    if summary.created_at_unix == 0 || summary.completed_at_unix == 0 {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_summary",
            msg: "summary timestamps must be non-zero".to_string(),
        });
    }
    if summary.completed_at_unix < summary.created_at_unix {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_summary",
            msg: "summary.completed_at_unix must be >= created_at_unix".to_string(),
        });
    }
    if let Some(duration) = summary.duration_secs {
        if !duration.is_finite() || duration < 0.0 {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_summary",
                msg: "summary.duration_secs must be finite and non-negative".to_string(),
            });
        }
    }
    if summary.schema_version < MIN_RUN_SCHEMA_VERSION
        || summary.schema_version > CURRENT_RUN_SCHEMA_VERSION
    {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_summary",
            msg: format!(
                "summary schema version {} outside supported range {}-{}",
                summary.schema_version, MIN_RUN_SCHEMA_VERSION, CURRENT_RUN_SCHEMA_VERSION
            ),
        });
    }
    validate_metrics_summary(&summary.metrics)?;
    if let Some(telemetry) = &summary.telemetry {
        validate_telemetry_summary(telemetry)?;
    }
    validate_layout_summary(&summary.layout)?;
    Ok(())
}

fn validate_metrics_summary(metrics: &MetricsSummary) -> Result<()> {
    if metrics.total_records == 0 {
        if metrics.first_step.is_some()
            || metrics.last_step.is_some()
            || metrics.first_timestamp_unix.is_some()
            || metrics.last_timestamp_unix.is_some()
            || !metrics.metrics.is_empty()
        {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_metrics_summary",
                msg: "metrics summary contains data but total_records is zero".to_string(),
            });
        }
        return Ok(());
    }
    if metrics.first_step.is_none()
        || metrics.last_step.is_none()
        || metrics.first_timestamp_unix.is_none()
        || metrics.last_timestamp_unix.is_none()
    {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metrics_summary",
            msg: "metrics summary missing step/timestamp bounds".to_string(),
        });
    }
    if let (Some(first), Some(last)) = (metrics.first_step, metrics.last_step) {
        if first > last {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_metrics_summary",
                msg: "metrics summary first_step must be <= last_step".to_string(),
            });
        }
    }
    for (name, stats) in &metrics.metrics {
        if name.trim().is_empty() {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_metrics_summary",
                msg: "metrics summary contains empty metric name".to_string(),
            });
        }
        validate_metric_stats(stats)?;
    }
    Ok(())
}

fn validate_metric_stats(stats: &MetricStats) -> Result<()> {
    if stats.count == 0 {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats count must be > 0".to_string(),
        });
    }
    let values = [
        stats.min, stats.max, stats.mean, stats.p50, stats.p95, stats.last,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats must be finite".to_string(),
        });
    }
    if stats.min > stats.max {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats min must be <= max".to_string(),
        });
    }
    if stats.mean < stats.min || stats.mean > stats.max {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats mean must be within [min, max]".to_string(),
        });
    }
    if stats.p50 < stats.min || stats.p50 > stats.max {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats p50 must be within [min, max]".to_string(),
        });
    }
    if stats.p95 < stats.min || stats.p95 > stats.max {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats p95 must be within [min, max]".to_string(),
        });
    }
    if stats.last < stats.min || stats.last > stats.max {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_metric_stats",
            msg: "metric stats last must be within [min, max]".to_string(),
        });
    }
    Ok(())
}

fn validate_telemetry_summary(telemetry: &TelemetrySummary) -> Result<()> {
    if telemetry.total_events == 0 && !telemetry.events.is_empty() {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_telemetry_summary",
            msg: "telemetry summary contains events but total_events is zero".to_string(),
        });
    }
    for (name, stats) in &telemetry.events {
        if name.trim().is_empty() {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_telemetry_summary",
                msg: "telemetry summary contains empty event name".to_string(),
            });
        }
        if stats.count == 0 {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_telemetry_summary",
                msg: "telemetry stats count must be > 0".to_string(),
            });
        }
        let durations = [
            stats.mean_duration_ms,
            stats.p95_duration_ms,
            stats.max_duration_ms,
        ];
        if durations
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_telemetry_summary",
                msg: "telemetry durations must be finite and non-negative".to_string(),
            });
        }
        if stats.mean_duration_ms > stats.max_duration_ms
            || stats.p95_duration_ms > stats.max_duration_ms
        {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_telemetry_summary",
                msg: "telemetry max_duration_ms must be the max".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_layout_summary(layout: &LayoutSummary) -> Result<()> {
    if layout.failures > layout.validations {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_layout_summary",
            msg: "layout failures cannot exceed validations".to_string(),
        });
    }
    if layout.overlap_failures > layout.failures {
        return Err(TorchError::Experiment {
            op: "experiment_store.validate_layout_summary",
            msg: "layout overlap_failures cannot exceed failures".to_string(),
        });
    }
    Ok(())
}

fn validate_run_overview(overview: &RunOverview) -> Result<()> {
    validate_run_metadata(&overview.metadata)?;
    if let Some(summary) = &overview.summary {
        validate_run_summary(summary)?;
        if summary.schema_version != overview.metadata.schema_version {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_overview",
                msg: format!(
                    "summary schema version {} does not match metadata {}",
                    summary.schema_version, overview.metadata.schema_version
                ),
            });
        }
        if summary.run_id != overview.metadata.id {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_overview",
                msg: "summary run_id does not match metadata id".to_string(),
            });
        }
        if summary.name != overview.metadata.name {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_overview",
                msg: "summary name does not match metadata name".to_string(),
            });
        }
        if summary.status != overview.metadata.status {
            return Err(TorchError::Experiment {
                op: "experiment_store.validate_overview",
                msg: "summary status does not match metadata status".to_string(),
            });
        }
    }
    Ok(())
}

fn summarize_metrics(path: &Path) -> Result<MetricsSummary> {
    if !path.exists() {
        return Ok(MetricsSummary {
            total_records: 0,
            first_step: None,
            last_step: None,
            first_timestamp_unix: None,
            last_timestamp_unix: None,
            metrics: BTreeMap::new(),
        });
    }
    let file = File::open(path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.summarize_metrics",
        msg: format!("failed to open {}: {err}", path.display()),
    })?;
    let reader = std::io::BufReader::new(file);
    let mut total_records = 0u64;
    let mut first_step = None;
    let mut last_step = None;
    let mut first_timestamp_unix = None;
    let mut last_timestamp_unix = None;
    let mut accumulators: BTreeMap<String, MetricAccumulator> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.map_err(|err| TorchError::Experiment {
            op: "experiment_store.summarize_metrics",
            msg: format!("failed to read metrics line: {err}"),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: MetricRecord =
            serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                op: "experiment_store.summarize_metrics",
                msg: format!("failed to parse metrics: {err}"),
            })?;
        total_records += 1;
        first_step.get_or_insert(record.step);
        last_step = Some(record.step);
        first_timestamp_unix.get_or_insert(record.timestamp_unix);
        last_timestamp_unix = Some(record.timestamp_unix);
        for (name, value) in record.metrics {
            accumulators
                .entry(name)
                .or_insert_with(MetricAccumulator::default)
                .push(value);
        }
    }

    let metrics = accumulators
        .into_iter()
        .map(|(name, acc)| (name, acc.into_stats()))
        .collect();

    Ok(MetricsSummary {
        total_records,
        first_step,
        last_step,
        first_timestamp_unix,
        last_timestamp_unix,
        metrics,
    })
}

fn summarize_telemetry(path: &Path) -> Result<Option<TelemetrySummary>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path).map_err(|err| TorchError::Experiment {
        op: "experiment_store.summarize_telemetry",
        msg: format!("failed to open {}: {err}", path.display()),
    })?;
    let reader = std::io::BufReader::new(file);
    let mut total_events = 0u64;
    let mut accumulators: BTreeMap<String, TelemetryAccumulator> = BTreeMap::new();
    for line in reader.lines() {
        let line = line.map_err(|err| TorchError::Experiment {
            op: "experiment_store.summarize_telemetry",
            msg: format!("failed to read telemetry line: {err}"),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let event: TelemetryEvent =
            serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                op: "experiment_store.summarize_telemetry",
                msg: format!("failed to parse telemetry: {err}"),
            })?;
        total_events += 1;
        accumulators
            .entry(event.name)
            .or_insert_with(TelemetryAccumulator::default)
            .push(event.duration_ms);
    }
    let events = accumulators
        .into_iter()
        .map(|(name, acc)| (name, acc.into_stats()))
        .collect();
    Ok(Some(TelemetrySummary {
        total_events,
        events,
    }))
}

fn summarize_layout() -> LayoutSummary {
    let stats = layout_stats();
    LayoutSummary {
        validations: stats.validations,
        failures: stats.failures,
        overlap_failures: stats.overlap_failures,
    }
}

const GK_EPSILON: f64 = 0.01;
const GK_EXACT_LIMIT: usize = 512;

#[derive(Clone, Debug)]
struct GkEntry {
    value: f64,
    g: u64,
    delta: u64,
}

#[derive(Debug)]
struct GkSummary {
    entries: Vec<GkEntry>,
    exact: Vec<f64>,
    count: u64,
    epsilon: f64,
}

impl Default for GkSummary {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            exact: Vec::new(),
            count: 0,
            epsilon: GK_EPSILON,
        }
    }
}

impl GkSummary {
    fn insert(&mut self, value: f64) {
        self.count += 1;
        if self.entries.is_empty() {
            self.exact.push(value);
            if self.exact.len() <= GK_EXACT_LIMIT {
                return;
            }
            let mut seeded = std::mem::take(&mut self.exact);
            seeded.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.entries = seeded
                .into_iter()
                .map(|value| GkEntry {
                    value,
                    g: 1,
                    delta: 0,
                })
                .collect();
        }

        let insert_idx = self.entries.partition_point(|entry| entry.value < value);
        let delta = if insert_idx == 0 || insert_idx == self.entries.len() {
            0
        } else {
            ((2.0 * self.epsilon * self.count as f64).floor().max(0.0) as u64).saturating_sub(1)
        };
        self.entries
            .insert(insert_idx, GkEntry { value, g: 1, delta });
        self.compress();
    }

    fn compress(&mut self) {
        if self.entries.len() < 2 {
            return;
        }
        let threshold = (2.0 * self.epsilon * self.count as f64).floor() as u64;
        let mut idx = 0;
        while idx + 1 < self.entries.len() {
            let next = &self.entries[idx + 1];
            let current = &self.entries[idx];
            if current.g + next.g + next.delta <= threshold {
                let merged = GkEntry {
                    value: next.value,
                    g: current.g + next.g,
                    delta: next.delta,
                };
                self.entries[idx + 1] = merged;
                self.entries.remove(idx);
            } else {
                idx += 1;
            }
        }
    }

    fn query(&self, quantile: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        if self.entries.is_empty() {
            let mut values = self.exact.clone();
            if values.is_empty() {
                return 0.0;
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            return percentile_f64_exact(&values, quantile);
        }
        let clamped = quantile.clamp(0.0, 1.0);
        if clamped <= 0.0 {
            return self.entries.first().map(|e| e.value).unwrap_or(0.0);
        }
        if clamped >= 1.0 {
            return self.entries.last().map(|e| e.value).unwrap_or(0.0);
        }
        let rank = (clamped * self.count as f64).ceil() as u64;
        let allowed = (self.epsilon * self.count as f64).ceil() as u64;
        let mut rmin = 0u64;
        for entry in &self.entries {
            rmin += entry.g;
            let rmax = rmin + entry.delta;
            if rmax + allowed >= rank {
                return entry.value;
            }
        }
        self.entries.last().map(|entry| entry.value).unwrap_or(0.0)
    }
}

#[derive(Default)]
struct MetricAccumulator {
    quantiles: GkSummary,
    sum: f64,
    min: f32,
    max: f32,
    last: f32,
    count: u64,
    initialized: bool,
}

impl MetricAccumulator {
    fn push(&mut self, value: f32) {
        self.quantiles.insert(value as f64);
        self.sum += value as f64;
        self.count += 1;
        if !self.initialized {
            self.min = value;
            self.max = value;
            self.last = value;
            self.initialized = true;
            return;
        }
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.last = value;
    }

    fn into_stats(self) -> MetricStats {
        if self.count == 0 {
            return MetricStats {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                last: 0.0,
            };
        }
        let mean = (self.sum / self.count as f64) as f32;
        let p50 = self.quantiles.query(0.50) as f32;
        let p95 = self.quantiles.query(0.95) as f32;
        MetricStats {
            count: self.count,
            min: self.min,
            max: self.max,
            mean,
            p50,
            p95,
            last: self.last,
        }
    }
}

#[derive(Default)]
struct TelemetryAccumulator {
    quantiles: GkSummary,
    sum: f64,
    max: f64,
    count: u64,
    initialized: bool,
}

impl TelemetryAccumulator {
    fn push(&mut self, value: f64) {
        self.quantiles.insert(value);
        self.sum += value;
        self.count += 1;
        if !self.initialized {
            self.max = value;
            self.initialized = true;
            return;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn into_stats(self) -> TelemetryStats {
        if self.count == 0 {
            return TelemetryStats {
                count: 0,
                mean_duration_ms: 0.0,
                p95_duration_ms: 0.0,
                max_duration_ms: 0.0,
            };
        }
        let mean_duration_ms = self.sum / self.count as f64;
        let p95_duration_ms = self.quantiles.query(0.95);
        TelemetryStats {
            count: self.count,
            mean_duration_ms,
            p95_duration_ms,
            max_duration_ms: self.max,
        }
    }
}

fn percentile_f64_exact(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * clamped).round() as usize;
    values[idx]
}

fn spawn_metrics_worker(
    path: PathBuf,
    config: MetricsLoggerConfig,
    telemetry: Option<Arc<TelemetryRecorder<JsonlSink>>>,
    error: Arc<Mutex<Option<TorchError>>>,
    receiver: mpsc::Receiver<MetricsMessage>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let file = match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => file,
            Err(err) => {
                record_logger_error(
                    &error,
                    TorchError::Experiment {
                        op: "metrics_logger.worker",
                        msg: format!("failed to open {}: {err}", path.display()),
                    },
                );
                return;
            }
        };
        let mut writer = BufWriter::new(file);
        let mut buffer = Vec::with_capacity(config.batch_size);
        let flush_interval = Duration::from_millis(config.flush_interval_ms);

        loop {
            match receiver.recv_timeout(flush_interval) {
                Ok(message) => match message {
                    MetricsMessage::Record(record) => {
                        buffer.push(record);
                        if buffer.len() >= config.batch_size {
                            if let Err(err) =
                                flush_metrics(&mut writer, &mut buffer, &config, telemetry.as_ref())
                            {
                                record_logger_error(&error, err);
                                break;
                            }
                        }
                    }
                    MetricsMessage::Flush(sender) => {
                        let result =
                            flush_metrics(&mut writer, &mut buffer, &config, telemetry.as_ref());
                        if let Err(err) = result.as_ref() {
                            record_logger_error(&error, err.clone());
                        }
                        let _ = sender.send(result);
                    }
                    MetricsMessage::Shutdown => {
                        let _ =
                            flush_metrics(&mut writer, &mut buffer, &config, telemetry.as_ref());
                        break;
                    }
                },
                Err(RecvTimeoutError::Timeout) => {
                    if !buffer.is_empty() {
                        if let Err(err) =
                            flush_metrics(&mut writer, &mut buffer, &config, telemetry.as_ref())
                        {
                            record_logger_error(&error, err);
                            break;
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = flush_metrics(&mut writer, &mut buffer, &config, telemetry.as_ref());
                    break;
                }
            }
        }
    })
}

fn flush_metrics(
    writer: &mut BufWriter<File>,
    buffer: &mut Vec<MetricRecord>,
    config: &MetricsLoggerConfig,
    telemetry: Option<&Arc<TelemetryRecorder<JsonlSink>>>,
) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    let start = Instant::now();
    let flushed = buffer.len();
    for record in buffer.drain(..) {
        serde_json::to_writer(&mut *writer, &record).map_err(|err| TorchError::Experiment {
            op: "metrics_logger.flush",
            msg: format!("failed to serialize metrics: {err}"),
        })?;
        writer
            .write_all(b"\n")
            .map_err(|err| TorchError::Experiment {
                op: "metrics_logger.flush",
                msg: format!("failed to write metrics: {err}"),
            })?;
    }
    writer.flush().map_err(|err| TorchError::Experiment {
        op: "metrics_logger.flush",
        msg: format!("failed to flush metrics: {err}"),
    })?;

    if let Some(recorder) = telemetry {
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        let event = TelemetryEvent::new("metrics_flush", duration_ms)
            .with_tag("count", flushed.to_string())
            .with_tag("batch_size", config.batch_size.to_string());
        if let Err(err) = recorder.record(event) {
            warn!("metrics telemetry failed: {err:?}");
        }
    }
    Ok(())
}

fn record_logger_error(error: &Arc<Mutex<Option<TorchError>>>, err: TorchError) {
    if let Ok(mut guard) = error.lock() {
        if guard.is_none() {
            *guard = Some(err.clone());
        }
    }
    warn!("metrics logger error: {err:?}");
}

fn build_csv_header(
    metric_names: &BTreeSet<String>,
    telemetry_names: &BTreeSet<String>,
) -> Vec<String> {
    let mut header = vec![
        "run_id".to_string(),
        "name".to_string(),
        "status".to_string(),
        "tags".to_string(),
        "created_at_unix".to_string(),
        "completed_at_unix".to_string(),
        "duration_secs".to_string(),
        "metrics_total_records".to_string(),
        "metrics_first_step".to_string(),
        "metrics_last_step".to_string(),
        "metrics_first_timestamp_unix".to_string(),
        "metrics_last_timestamp_unix".to_string(),
        "layout_validations".to_string(),
        "layout_failures".to_string(),
        "layout_overlap_failures".to_string(),
        "telemetry_total_events".to_string(),
    ];
    for name in metric_names {
        header.extend(metric_stat_headers(name));
    }
    for name in telemetry_names {
        header.extend(telemetry_stat_headers(name));
    }
    header
}

fn metric_stat_headers(name: &str) -> Vec<String> {
    vec![
        format!("metric.{name}.count"),
        format!("metric.{name}.min"),
        format!("metric.{name}.max"),
        format!("metric.{name}.mean"),
        format!("metric.{name}.p50"),
        format!("metric.{name}.p95"),
        format!("metric.{name}.last"),
    ]
}

fn telemetry_stat_headers(name: &str) -> Vec<String> {
    vec![
        format!("telemetry.{name}.count"),
        format!("telemetry.{name}.mean_ms"),
        format!("telemetry.{name}.p95_ms"),
        format!("telemetry.{name}.max_ms"),
    ]
}

fn build_csv_row(
    overview: &RunOverview,
    metric_names: &BTreeSet<String>,
    telemetry_names: &BTreeSet<String>,
) -> Vec<String> {
    let summary = overview.summary.as_ref();
    let metadata = &overview.metadata;
    let tags = metadata.tags.join("|");
    let (completed_at, duration_secs, metrics, telemetry, layout) = match summary {
        Some(summary) => (
            Some(summary.completed_at_unix),
            summary.duration_secs,
            Some(&summary.metrics),
            summary.telemetry.as_ref(),
            Some(&summary.layout),
        ),
        None => (None, None, None, None, None),
    };

    let mut row = vec![
        metadata.id.clone(),
        metadata.name.clone(),
        format!("{:?}", metadata.status),
        tags,
        metadata.created_at_unix.to_string(),
        completed_at
            .map(|value| value.to_string())
            .unwrap_or_default(),
        duration_secs
            .map(|value| value.to_string())
            .unwrap_or_default(),
        metrics
            .map(|value| value.total_records.to_string())
            .unwrap_or_default(),
        metrics
            .and_then(|value| value.first_step.map(|step| step.to_string()))
            .unwrap_or_default(),
        metrics
            .and_then(|value| value.last_step.map(|step| step.to_string()))
            .unwrap_or_default(),
        metrics
            .and_then(|value| value.first_timestamp_unix.map(|ts| ts.to_string()))
            .unwrap_or_default(),
        metrics
            .and_then(|value| value.last_timestamp_unix.map(|ts| ts.to_string()))
            .unwrap_or_default(),
        layout
            .map(|value| value.validations.to_string())
            .unwrap_or_default(),
        layout
            .map(|value| value.failures.to_string())
            .unwrap_or_default(),
        layout
            .map(|value| value.overlap_failures.to_string())
            .unwrap_or_default(),
        telemetry
            .map(|value| value.total_events.to_string())
            .unwrap_or_default(),
    ];

    for name in metric_names {
        if let Some(metrics) = metrics {
            if let Some(stats) = metrics.metrics.get(name) {
                row.push(stats.count.to_string());
                row.push(stats.min.to_string());
                row.push(stats.max.to_string());
                row.push(stats.mean.to_string());
                row.push(stats.p50.to_string());
                row.push(stats.p95.to_string());
                row.push(stats.last.to_string());
                continue;
            }
        }
        row.extend(std::iter::repeat(String::new()).take(7));
    }

    for name in telemetry_names {
        if let Some(telemetry) = telemetry {
            if let Some(stats) = telemetry.events.get(name) {
                row.push(stats.count.to_string());
                row.push(stats.mean_duration_ms.to_string());
                row.push(stats.p95_duration_ms.to_string());
                row.push(stats.max_duration_ms.to_string());
                continue;
            }
        }
        row.extend(std::iter::repeat(String::new()).take(4));
    }

    row
}

fn write_csv_row(writer: &mut BufWriter<File>, row: &[String]) -> Result<usize> {
    let mut bytes = 0usize;
    for (idx, field) in row.iter().enumerate() {
        if idx > 0 {
            writer
                .write_all(b",")
                .map_err(|err| TorchError::Experiment {
                    op: "experiment_store.export_run_summaries_csv",
                    msg: format!("failed to write csv delimiter: {err}"),
                })?;
            bytes += 1;
        }
        let escaped = csv_escape(field);
        writer
            .write_all(escaped.as_bytes())
            .map_err(|err| TorchError::Experiment {
                op: "experiment_store.export_run_summaries_csv",
                msg: format!("failed to write csv field: {err}"),
            })?;
        bytes += escaped.as_bytes().len();
    }
    writer
        .write_all(b"\n")
        .map_err(|err| TorchError::Experiment {
            op: "experiment_store.export_run_summaries_csv",
            msg: format!("failed to write csv newline: {err}"),
        })?;
    bytes += 1;
    Ok(bytes)
}

fn csv_escape(value: &str) -> String {
    let needs_quotes = value.contains(',') || value.contains('"') || value.contains('\n');
    if !needs_quotes {
        return value.to_string();
    }
    let escaped = value.replace('"', "\"\"");
    format!("\"{escaped}\"")
}
