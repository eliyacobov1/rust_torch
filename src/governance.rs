use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
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
    stage_id: String,
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
    entries: BTreeMap<String, GovernanceScheduleEntry>,
}

impl DeterministicScheduler {
    pub fn new(seed: u64, run_dirs: Vec<PathBuf>) -> Result<Self> {
        let mut heap = BinaryHeap::with_capacity(run_dirs.len());
        let mut entries = BTreeMap::new();
        for (ordinal, run_dir) in run_dirs.into_iter().enumerate() {
            let run_id = run_id_from_path(&run_dir)?;
            let priority = hash_priority(seed, &run_id);
            let entry = GovernanceScheduleEntry {
                run_id: run_id.clone(),
                run_dir,
                priority,
                ordinal,
            };
            entries.insert(run_id.clone(), entry);
            heap.push(QueueItem {
                priority,
                ordinal,
                stage_id: run_id,
            });
        }
        Ok(Self {
            seed,
            heap,
            entries,
        })
    }

    pub fn drain(mut self) -> Vec<GovernanceScheduleEntry> {
        let mut entries = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.heap.pop() {
            if let Some(entry) = self.entries.remove(&item.stage_id) {
                entries.push(entry);
            }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceStage {
    pub id: String,
    pub run_id: String,
    pub stage: String,
    pub dependencies: BTreeSet<String>,
    pub ordinal: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceGraph {
    nodes: BTreeMap<String, GovernanceStage>,
}

impl GovernanceGraph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    pub fn add_stage(
        &mut self,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Result<String> {
        let run_id = run_id.into();
        let stage = stage.into();
        let stage_id = format!("{run_id}:{stage}");
        self.add_stage_with_id(stage_id.clone(), run_id, stage, dependencies)?;
        Ok(stage_id)
    }

    pub fn add_stage_with_id(
        &mut self,
        stage_id: String,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Result<()> {
        if self.nodes.contains_key(&stage_id) {
            return Err(TorchError::Experiment {
                op: "governance.graph.add_stage",
                msg: format!("duplicate stage id {stage_id}"),
            });
        }
        let ordinal = self.nodes.len();
        let node = GovernanceStage {
            id: stage_id.clone(),
            run_id: run_id.into(),
            stage: stage.into(),
            dependencies: dependencies.into_iter().collect(),
            ordinal,
        };
        self.nodes.insert(stage_id, node);
        Ok(())
    }

    pub fn stages(&self) -> impl Iterator<Item = &GovernanceStage> {
        self.nodes.values()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernancePlanEntry {
    pub stage_id: String,
    pub run_id: String,
    pub stage: String,
    pub priority: u64,
    pub ordinal: usize,
    pub wave: usize,
    pub lane: usize,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernancePlan {
    pub seed: u64,
    pub generated_at_unix: u64,
    pub entries: Vec<GovernancePlanEntry>,
    pub total_waves: usize,
    pub total_stages: usize,
}

impl GovernancePlan {
    pub fn total_lanes(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.lane)
            .max()
            .map(|lane| lane.saturating_add(1))
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct GovernancePlanner {
    seed: u64,
    graph: GovernanceGraph,
}

impl GovernancePlanner {
    pub fn new(seed: u64, graph: GovernanceGraph) -> Self {
        Self { seed, graph }
    }

    pub fn plan(&self) -> Result<GovernancePlan> {
        if self.graph.nodes.is_empty() {
            return Ok(GovernancePlan {
                seed: self.seed,
                generated_at_unix: now_unix_seconds()?,
                entries: Vec::new(),
                total_waves: 0,
                total_stages: 0,
            });
        }
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (stage_id, stage) in &self.graph.nodes {
            in_degree.insert(stage_id.clone(), stage.dependencies.len());
            for dep in &stage.dependencies {
                if !self.graph.nodes.contains_key(dep) {
                    return Err(TorchError::Experiment {
                        op: "governance.plan",
                        msg: format!("missing dependency {dep} for stage {stage_id}"),
                    });
                }
                dependents
                    .entry(dep.clone())
                    .or_default()
                    .insert(stage_id.clone());
            }
        }

        let mut heap = BinaryHeap::new();
        for (stage_id, degree) in &in_degree {
            if *degree == 0 {
                let node = self.graph.nodes.get(stage_id).expect("node exists");
                heap.push(QueueItem {
                    priority: hash_priority(self.seed, stage_id),
                    ordinal: node.ordinal,
                    stage_id: stage_id.clone(),
                });
            }
        }

        let mut entries = Vec::with_capacity(self.graph.nodes.len());
        let mut wave = 0usize;
        while !heap.is_empty() {
            let mut ready = Vec::new();
            while let Some(item) = heap.pop() {
                ready.push(item);
            }
            ready.sort_by(|a, b| match b.priority.cmp(&a.priority) {
                Ordering::Equal => a.ordinal.cmp(&b.ordinal),
                ordering => ordering,
            });
            for (lane, item) in ready.iter().enumerate() {
                let node = self
                    .graph
                    .nodes
                    .get(&item.stage_id)
                    .expect("node exists");
                entries.push(GovernancePlanEntry {
                    stage_id: node.id.clone(),
                    run_id: node.run_id.clone(),
                    stage: node.stage.clone(),
                    priority: item.priority,
                    ordinal: item.ordinal,
                    wave,
                    lane,
                    dependencies: node.dependencies.iter().cloned().collect(),
                });
                if let Some(children) = dependents.get(&item.stage_id) {
                    for child in children {
                        if let Some(degree) = in_degree.get_mut(child) {
                            *degree = degree.saturating_sub(1);
                            if *degree == 0 {
                                let child_node = self
                                    .graph
                                    .nodes
                                    .get(child)
                                    .expect("child exists");
                                heap.push(QueueItem {
                                    priority: hash_priority(self.seed, child),
                                    ordinal: child_node.ordinal,
                                    stage_id: child.clone(),
                                });
                            }
                        }
                    }
                }
            }
            wave = wave.saturating_add(1);
        }

        if entries.len() != self.graph.nodes.len() {
            return Err(TorchError::Experiment {
                op: "governance.plan",
                msg: "cycle detected in governance graph".to_string(),
            });
        }

        Ok(GovernancePlan {
            seed: self.seed,
            generated_at_unix: now_unix_seconds()?,
            total_waves: wave,
            total_stages: entries.len(),
            entries,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GovernanceReplayCursor {
    entries: Vec<GovernancePlanEntry>,
    position: usize,
}

impl GovernanceReplayCursor {
    pub fn new(plan: GovernancePlan) -> Self {
        Self {
            entries: plan.entries,
            position: 0,
        }
    }

    pub fn expect_stage(&mut self, stage_id: &str) -> Result<GovernancePlanEntry> {
        let Some(entry) = self.entries.get(self.position) else {
            return Err(TorchError::Experiment {
                op: "governance.replay",
                msg: format!("expected stage {stage_id} but plan is exhausted"),
            });
        };
        if entry.stage_id != stage_id {
            return Err(TorchError::Experiment {
                op: "governance.replay",
                msg: format!(
                    "deterministic plan mismatch: expected {stage_id}, got {}",
                    entry.stage_id
                ),
            });
        }
        self.position = self.position.saturating_add(1);
        Ok(entry.clone())
    }

    pub fn remaining(&self) -> usize {
        self.entries.len().saturating_sub(self.position)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GovernanceAction {
    Started,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceLedgerEvent {
    pub index: u64,
    pub timestamp_unix: u64,
    pub stage_id: String,
    pub run_id: String,
    pub stage: String,
    pub action: GovernanceAction,
    pub message: String,
    pub wave: usize,
    pub lane: usize,
    pub prev_hash: Option<String>,
    pub hash: String,
}

impl GovernanceLedgerEvent {
    pub fn new(
        stage_id: impl Into<String>,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        action: GovernanceAction,
        message: impl Into<String>,
        wave: usize,
        lane: usize,
    ) -> Self {
        Self {
            index: 0,
            timestamp_unix: 0,
            stage_id: stage_id.into(),
            run_id: run_id.into(),
            stage: stage.into(),
            action,
            message: message.into(),
            wave,
            lane,
            prev_hash: None,
            hash: String::new(),
        }
    }
}

#[derive(Debug)]
pub struct GovernanceLedger {
    path: PathBuf,
    writer: BufWriter<File>,
    last_hash: Option<Hash>,
    next_index: u64,
}

impl GovernanceLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| TorchError::Experiment {
                op: "governance_ledger.open",
                msg: format!("failed to create ledger dir {}: {err}", parent.display()),
            })?;
        }
        let mut last_hash = None;
        let mut next_index = 0;
        if path.exists() {
            let file = File::open(&path).map_err(|err| TorchError::Experiment {
                op: "governance_ledger.open",
                msg: format!("failed to open ledger {}: {err}", path.display()),
            })?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|err| TorchError::Experiment {
                    op: "governance_ledger.open",
                    msg: format!("failed to read ledger {}: {err}", path.display()),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let event: GovernanceLedgerEvent =
                    serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                        op: "governance_ledger.open",
                        msg: format!("failed to parse ledger {}: {err}", path.display()),
                    })?;
                let hash = Hash::from_hex(&event.hash).map_err(|err| TorchError::Experiment {
                    op: "governance_ledger.open",
                    msg: format!("failed to parse ledger hash {}: {err}", event.hash),
                })?;
                last_hash = Some(hash);
                next_index = next_index.max(event.index + 1);
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| TorchError::Experiment {
                op: "governance_ledger.open",
                msg: format!("failed to open ledger {}: {err}", path.display()),
            })?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            last_hash,
            next_index,
        })
    }

    pub fn record(&mut self, mut event: GovernanceLedgerEvent) -> Result<GovernanceLedgerEvent> {
        event.index = self.next_index;
        event.timestamp_unix = now_unix_seconds()?;
        event.prev_hash = self.last_hash.map(|hash| hash.to_hex().to_string());
        let hash = hash_ledger_event(&event)?;
        event.hash = hash.to_hex().to_string();
        self.last_hash = Some(hash);
        self.next_index = self.next_index.saturating_add(1);
        let serialized = serde_json::to_string(&event).map_err(|err| TorchError::Experiment {
            op: "governance_ledger.record",
            msg: format!("failed to serialize governance ledger event: {err}"),
        })?;
        self.writer
            .write_all(serialized.as_bytes())
            .and_then(|_| self.writer.write_all(b"\n"))
            .map_err(|err| TorchError::Experiment {
                op: "governance_ledger.record",
                msg: format!("failed to write governance ledger event: {err}"),
            })?;
        self.writer.flush().map_err(|err| TorchError::Experiment {
            op: "governance_ledger.record",
            msg: format!("failed to flush governance ledger: {err}"),
        })?;
        Ok(event)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub fn build_governance_schedule(seed: u64, run_dirs: Vec<PathBuf>) -> Result<Vec<GovernanceScheduleEntry>> {
    DeterministicScheduler::new(seed, run_dirs).map(DeterministicScheduler::drain)
}

pub fn deterministic_run_order(seed: u64, run_ids: &[String]) -> Vec<String> {
    let mut items = Vec::with_capacity(run_ids.len());
    for (ordinal, run_id) in run_ids.iter().enumerate() {
        let priority = hash_priority(seed, run_id);
        items.push((priority, ordinal, run_id.clone()));
    }
    items.sort_by(|a, b| match b.0.cmp(&a.0) {
        Ordering::Equal => a.1.cmp(&b.1),
        ordering => ordering,
    });
    items.into_iter().map(|(_, _, run_id)| run_id).collect()
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

fn hash_ledger_event(event: &GovernanceLedgerEvent) -> Result<Hash> {
    let payload = GovernanceLedgerPayload {
        index: event.index,
        timestamp_unix: event.timestamp_unix,
        stage_id: &event.stage_id,
        run_id: &event.run_id,
        stage: &event.stage,
        action: &event.action,
        message: &event.message,
        wave: event.wave,
        lane: event.lane,
        prev_hash: &event.prev_hash,
    };
    let payload = serde_json::to_vec(&payload).map_err(|err| TorchError::Experiment {
        op: "governance_ledger.hash_event",
        msg: format!("failed to serialize ledger payload: {err}"),
    })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&payload);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct GovernanceLedgerPayload<'a> {
    index: u64,
    timestamp_unix: u64,
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    action: &'a GovernanceAction,
    message: &'a str,
    wave: usize,
    lane: usize,
    prev_hash: &'a Option<String>,
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
