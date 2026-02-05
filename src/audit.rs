use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use blake3::Hash;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditScope {
    Store,
    Run,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditStatus {
    Started,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub index: u64,
    pub timestamp_unix: u64,
    pub scope: AuditScope,
    pub run_id: Option<String>,
    pub stage: Option<String>,
    pub status: AuditStatus,
    pub message: String,
    pub findings: usize,
    pub prev_hash: Option<String>,
    pub hash: String,
}

impl AuditEvent {
    pub fn new(
        scope: AuditScope,
        run_id: Option<String>,
        stage: Option<String>,
        status: AuditStatus,
        message: impl Into<String>,
        findings: usize,
    ) -> Self {
        Self {
            index: 0,
            timestamp_unix: 0,
            scope,
            run_id,
            stage,
            status,
            message: message.into(),
            findings,
            prev_hash: None,
            hash: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MerkleAccumulator {
    leaves: Vec<Hash>,
}

impl MerkleAccumulator {
    pub fn new() -> Self {
        Self { leaves: Vec::new() }
    }

    pub fn append(&mut self, hash: Hash) {
        self.leaves.push(hash);
    }

    pub fn root(&self) -> Option<Hash> {
        if self.leaves.is_empty() {
            return None;
        }
        let mut level = self.leaves.clone();
        while level.len() > 1 {
            if level.len() % 2 == 1 {
                let last = *level.last().expect("non-empty");
                level.push(last);
            }
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                let combined = hash_pair(pair[0], pair[1]);
                next.push(combined);
            }
            level = next;
        }
        level.first().copied()
    }

    pub fn root_hex(&self) -> Option<String> {
        self.root().map(|hash| hash.to_hex().to_string())
    }
}

#[derive(Debug)]
pub struct AuditLog {
    path: PathBuf,
    writer: BufWriter<File>,
    last_hash: Option<Hash>,
    merkle: MerkleAccumulator,
    next_index: u64,
}

impl AuditLog {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| TorchError::Experiment {
                op: "audit_log.open",
                msg: format!("failed to create audit dir {}: {err}", parent.display()),
            })?;
        }

        let mut last_hash = None;
        let mut merkle = MerkleAccumulator::new();
        let mut next_index = 0;
        if path.exists() {
            let file = File::open(&path).map_err(|err| TorchError::Experiment {
                op: "audit_log.open",
                msg: format!("failed to open audit log {}: {err}", path.display()),
            })?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|err| TorchError::Experiment {
                    op: "audit_log.open",
                    msg: format!("failed to read audit log {}: {err}", path.display()),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let event: AuditEvent =
                    serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                        op: "audit_log.open",
                        msg: format!("failed to parse audit log {}: {err}", path.display()),
                    })?;
                let hash = Hash::from_hex(&event.hash).map_err(|err| TorchError::Experiment {
                    op: "audit_log.open",
                    msg: format!("failed to parse audit hash {}: {err}", event.hash),
                })?;
                merkle.append(hash);
                last_hash = Some(hash);
                next_index = next_index.max(event.index + 1);
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| TorchError::Experiment {
                op: "audit_log.open",
                msg: format!("failed to open audit log {}: {err}", path.display()),
            })?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            last_hash,
            merkle,
            next_index,
        })
    }

    pub fn record(&mut self, mut event: AuditEvent) -> Result<AuditEvent> {
        event.index = self.next_index;
        event.timestamp_unix = now_unix_seconds()?;
        event.prev_hash = self.last_hash.map(|hash| hash.to_hex().to_string());
        let hash = hash_event(&event)?;
        event.hash = hash.to_hex().to_string();
        self.last_hash = Some(hash);
        self.merkle.append(hash);
        self.next_index = self.next_index.saturating_add(1);
        let serialized = serde_json::to_string(&event).map_err(|err| TorchError::Experiment {
            op: "audit_log.record",
            msg: format!("failed to serialize audit event: {err}"),
        })?;
        self.writer
            .write_all(serialized.as_bytes())
            .and_then(|_| self.writer.write_all(b"\n"))
            .map_err(|err| TorchError::Experiment {
                op: "audit_log.record",
                msg: format!("failed to write audit event: {err}"),
            })?;
        self.writer.flush().map_err(|err| TorchError::Experiment {
            op: "audit_log.record",
            msg: format!("failed to flush audit log: {err}"),
        })?;
        Ok(event)
    }

    pub fn merkle_root_hex(&self) -> Option<String> {
        self.merkle.root_hex()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn now_unix_seconds() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|err| TorchError::Experiment {
            op: "audit_log.now_unix_seconds",
            msg: format!("system clock error: {err}"),
        })
}

#[derive(Serialize)]
struct AuditEventPayload<'a> {
    index: u64,
    timestamp_unix: u64,
    scope: &'a AuditScope,
    run_id: &'a Option<String>,
    stage: &'a Option<String>,
    status: &'a AuditStatus,
    message: &'a str,
    findings: usize,
    prev_hash: &'a Option<String>,
}

fn hash_event(event: &AuditEvent) -> Result<Hash> {
    let payload = AuditEventPayload {
        index: event.index,
        timestamp_unix: event.timestamp_unix,
        scope: &event.scope,
        run_id: &event.run_id,
        stage: &event.stage,
        status: &event.status,
        message: &event.message,
        findings: event.findings,
        prev_hash: &event.prev_hash,
    };
    let serialized = serde_json::to_vec(&payload).map_err(|err| TorchError::Experiment {
        op: "audit_log.hash_event",
        msg: format!("failed to serialize audit payload: {err}"),
    })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&serialized);
    Ok(hasher.finalize())
}

fn hash_pair(left: Hash, right: Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    hasher.finalize()
}
