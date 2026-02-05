use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use blake3::Hash;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceScheduleEntry {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub priority: u64,
    pub ordinal: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceSchedule {
    pub seed: u64,
    pub generated_at_unix: u64,
    pub entries: Vec<GovernanceScheduleEntry>,
}

#[derive(Debug, Clone)]
struct QueueItem {
    priority: u64,
    ordinal: usize,
    entry: GovernanceScheduleEntry,
}

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.ordinal == other.ordinal
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => other.ordinal.cmp(&self.ordinal),
            ordering => ordering,
        }
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct DeterministicScheduler {
    seed: u64,
    heap: BinaryHeap<QueueItem>,
}

impl DeterministicScheduler {
    pub fn new(seed: u64, run_dirs: Vec<PathBuf>) -> Result<Self> {
        let mut heap = BinaryHeap::with_capacity(run_dirs.len());
        for (ordinal, run_dir) in run_dirs.into_iter().enumerate() {
            let run_id = run_id_from_path(&run_dir)?;
            let priority = hash_priority(seed, &run_id);
            let entry = GovernanceScheduleEntry {
                run_id,
                run_dir,
                priority,
                ordinal,
            };
            heap.push(QueueItem {
                priority,
                ordinal,
                entry,
            });
        }
        Ok(Self { seed, heap })
    }

    pub fn drain(mut self) -> Vec<GovernanceScheduleEntry> {
        let mut entries = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.heap.pop() {
            entries.push(item.entry);
        }
        entries
    }

    pub fn into_schedule(self) -> Result<GovernanceSchedule> {
        Ok(GovernanceSchedule {
            seed: self.seed,
            generated_at_unix: now_unix_seconds()?,
            entries: self.drain(),
        })
    }
}

pub fn build_governance_schedule(seed: u64, run_dirs: Vec<PathBuf>) -> Result<Vec<GovernanceScheduleEntry>> {
    DeterministicScheduler::new(seed, run_dirs).map(DeterministicScheduler::drain)
}

fn run_id_from_path(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .ok_or_else(|| TorchError::Experiment {
            op: "governance.schedule",
            msg: format!("failed to derive run id from {}", path.display()),
        })
}

fn hash_priority(seed: u64, run_id: &str) -> u64 {
    let payload = format!("{seed}:{run_id}");
    let hash = blake3::hash(payload.as_bytes());
    priority_from_hash(hash)
}

fn priority_from_hash(hash: Hash) -> u64 {
    let bytes = hash.as_bytes();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[0..8]);
    u64::from_le_bytes(buf)
}

fn now_unix_seconds() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|err| TorchError::Experiment {
            op: "governance.schedule_time",
            msg: format!("system clock error: {err}"),
        })
}
