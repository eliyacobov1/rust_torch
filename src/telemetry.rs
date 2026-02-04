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
