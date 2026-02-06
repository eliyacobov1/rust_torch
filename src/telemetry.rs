use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use log::warn;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub name: String,
    pub duration_ms: f64,
    pub timestamp_ms: u128,
    pub tags: BTreeMap<String, String>,
}

impl TelemetryEvent {
    pub fn new(name: impl Into<String>, duration_ms: f64) -> Self {
        Self {
            name: name.into(),
            duration_ms,
            timestamp_ms: current_timestamp_ms(),
            tags: BTreeMap::new(),
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStats {
    pub count: u64,
    pub mean_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub max_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub total_events: u64,
    pub events: BTreeMap<String, TelemetryStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TelemetryBudgetStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBudgetViolation {
    pub event_name: Option<String>,
    pub metric: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBudgetReport {
    pub status: TelemetryBudgetStatus,
    pub violations: Vec<TelemetryBudgetViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBudgetThreshold {
    pub max_mean_duration_ms: Option<f64>,
    pub max_p95_duration_ms: Option<f64>,
    pub max_max_duration_ms: Option<f64>,
    pub max_event_count: Option<u64>,
}

impl TelemetryBudgetThreshold {
    fn evaluate(
        &self,
        event_name: Option<&str>,
        stats: &TelemetryStats,
        violations: &mut Vec<TelemetryBudgetViolation>,
    ) {
        if let Some(limit) = self.max_mean_duration_ms {
            if stats.mean_duration_ms > limit {
                violations.push(TelemetryBudgetViolation {
                    event_name: event_name.map(str::to_string),
                    metric: "mean_duration_ms".to_string(),
                    expected: format!("<= {limit}"),
                    actual: format!("{}", stats.mean_duration_ms),
                });
            }
        }
        if let Some(limit) = self.max_p95_duration_ms {
            if stats.p95_duration_ms > limit {
                violations.push(TelemetryBudgetViolation {
                    event_name: event_name.map(str::to_string),
                    metric: "p95_duration_ms".to_string(),
                    expected: format!("<= {limit}"),
                    actual: format!("{}", stats.p95_duration_ms),
                });
            }
        }
        if let Some(limit) = self.max_max_duration_ms {
            if stats.max_duration_ms > limit {
                violations.push(TelemetryBudgetViolation {
                    event_name: event_name.map(str::to_string),
                    metric: "max_duration_ms".to_string(),
                    expected: format!("<= {limit}"),
                    actual: format!("{}", stats.max_duration_ms),
                });
            }
        }
        if let Some(limit) = self.max_event_count {
            if stats.count > limit {
                violations.push(TelemetryBudgetViolation {
                    event_name: event_name.map(str::to_string),
                    metric: "event_count".to_string(),
                    expected: format!("<= {limit}"),
                    actual: format!("{}", stats.count),
                });
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBudget {
    pub max_total_events: Option<u64>,
    pub global: TelemetryBudgetThreshold,
    pub per_event: BTreeMap<String, TelemetryBudgetThreshold>,
}

impl TelemetryBudget {
    pub fn evaluate(&self, summary: &TelemetrySummary) -> Result<TelemetryBudgetReport> {
        let mut violations = Vec::new();
        if let Some(limit) = self.max_total_events {
            if summary.total_events > limit {
                violations.push(TelemetryBudgetViolation {
                    event_name: None,
                    metric: "total_events".to_string(),
                    expected: format!("<= {limit}"),
                    actual: format!("{}", summary.total_events),
                });
            }
        }

        for (event_name, stats) in &summary.events {
            self.global
                .evaluate(Some(event_name.as_str()), stats, &mut violations);
            if let Some(threshold) = self.per_event.get(event_name) {
                threshold.evaluate(Some(event_name.as_str()), stats, &mut violations);
            }
        }

        let status = if violations.is_empty() {
            TelemetryBudgetStatus::Pass
        } else {
            TelemetryBudgetStatus::Fail
        };

        Ok(TelemetryBudgetReport { status, violations })
    }
}

pub trait TelemetrySink: Send + Sync + 'static {
    fn record(&self, event: TelemetryEvent) -> Result<()>;
}

#[derive(Debug)]
pub struct JsonlSink {
    writer: Arc<Mutex<BufWriter<File>>>,
}

impl JsonlSink {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: Arc::new(Mutex::new(BufWriter::new(file))),
        })
    }
}

impl TelemetrySink for JsonlSink {
    fn record(&self, event: TelemetryEvent) -> Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| anyhow!("telemetry writer lock poisoned"))?;
        serde_json::to_writer(&mut *writer, &event)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }
}

pub struct TelemetryRecorder<S: TelemetrySink> {
    sender: Sender<TelemetryEvent>,
    handle: Option<JoinHandle<()>>,
    _sink: Arc<S>,
}

impl<S: TelemetrySink> TelemetryRecorder<S> {
    pub fn new(sink: S) -> Self {
        let (sender, receiver) = mpsc::channel::<TelemetryEvent>();
        let sink = Arc::new(sink);
        let sink_clone = Arc::clone(&sink);
        let handle = thread::spawn(move || {
            for event in receiver {
                if let Err(err) = sink_clone.record(event) {
                    warn!("telemetry record failed: {err:?}");
                }
            }
        });
        Self {
            sender,
            handle: Some(handle),
            _sink: sink,
        }
    }

    pub fn record(&self, event: TelemetryEvent) -> Result<()> {
        self.sender
            .send(event)
            .map_err(|_| anyhow!("telemetry channel closed"))
    }

    pub fn timer(&self, name: impl Into<String>) -> TelemetryTimer<'_, S> {
        TelemetryTimer::new(self, name)
    }
}

impl<S: TelemetrySink> Drop for TelemetryRecorder<S> {
    fn drop(&mut self) {
        drop(self.sender.clone());
        if let Some(handle) = self.handle.take() {
            if let Err(err) = handle.join() {
                warn!("telemetry worker join failed: {err:?}");
            }
        }
    }
}

pub struct TelemetryTimer<'a, S: TelemetrySink> {
    recorder: &'a TelemetryRecorder<S>,
    name: String,
    start: Instant,
    tags: BTreeMap<String, String>,
}

impl<'a, S: TelemetrySink> TelemetryTimer<'a, S> {
    fn new(recorder: &'a TelemetryRecorder<S>, name: impl Into<String>) -> Self {
        Self {
            recorder,
            name: name.into(),
            start: Instant::now(),
            tags: BTreeMap::new(),
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

impl<'a, S: TelemetrySink> Drop for TelemetryTimer<'a, S> {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let mut event = TelemetryEvent::new(self.name.clone(), duration_ms);
        for (key, value) in std::mem::take(&mut self.tags) {
            event = event.with_tag(key, value);
        }
        if let Err(err) = self.recorder.record(event) {
            warn!("telemetry timer record failed: {err:?}");
        }
    }
}

pub fn jsonl_recorder_from_env(var: &str) -> Result<Option<TelemetryRecorder<JsonlSink>>> {
    let path = match std::env::var_os(var) {
        Some(value) => PathBuf::from(value),
        None => return Ok(None),
    };
    let sink = JsonlSink::new(path)?;
    Ok(Some(TelemetryRecorder::new(sink)))
}

fn current_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
