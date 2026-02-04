use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{info, warn};
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::checkpoint::{save_state_dict, StateDict};
use crate::error::{Result, TorchError};
use crate::telemetry::{JsonlSink, TelemetryEvent, TelemetryRecorder};

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
        };
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
        };
        write_summary(&run_dir, &summary)?;
        Ok(summary)
    }

    fn run_dir(&self) -> PathBuf {
        self.store.root.join(&self.metadata.id)
    }
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
    serde_json::from_str(&buf).map_err(|err| TorchError::Experiment {
        op: "experiment_store.read_metadata",
        msg: format!("failed to parse metadata: {err}"),
    })
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

#[derive(Default)]
struct MetricAccumulator {
    values: Vec<f32>,
    sum: f64,
    min: f32,
    max: f32,
    last: f32,
    initialized: bool,
}

impl MetricAccumulator {
    fn push(&mut self, value: f32) {
        self.values.push(value);
        self.sum += value as f64;
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

    fn into_stats(mut self) -> MetricStats {
        let count = self.values.len() as u64;
        if count == 0 {
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
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = (self.sum / count as f64) as f32;
        let p50 = percentile(&self.values, 0.50);
        let p95 = percentile(&self.values, 0.95);
        MetricStats {
            count,
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
    values: Vec<f64>,
    sum: f64,
    max: f64,
    initialized: bool,
}

impl TelemetryAccumulator {
    fn push(&mut self, value: f64) {
        self.values.push(value);
        self.sum += value;
        if !self.initialized {
            self.max = value;
            self.initialized = true;
            return;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn into_stats(mut self) -> TelemetryStats {
        let count = self.values.len() as u64;
        if count == 0 {
            return TelemetryStats {
                count: 0,
                mean_duration_ms: 0.0,
                p95_duration_ms: 0.0,
                max_duration_ms: 0.0,
            };
        }
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_duration_ms = self.sum / count as f64;
        let p95_duration_ms = percentile_f64(&self.values, 0.95);
        TelemetryStats {
            count,
            mean_duration_ms,
            p95_duration_ms,
            max_duration_ms: self.max,
        }
    }
}

fn percentile(values: &[f32], percentile: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * clamped).round() as usize;
    values[idx]
}

fn percentile_f64(values: &[f64], percentile: f64) -> f64 {
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
