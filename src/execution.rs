use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use blake3::Hash;
use log::{info, warn};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};
use crate::governance::GovernanceGraph;
use crate::telemetry::{TelemetryEvent, TelemetryRecorder, TelemetrySink};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCostProfile {
    pub default_cost: u64,
    pub overrides: BTreeMap<String, u64>,
}

impl ExecutionCostProfile {
    pub fn cost_for(&self, stage_id: &str) -> u64 {
        self.overrides
            .get(stage_id)
            .copied()
            .unwrap_or(self.default_cost)
            .max(1)
    }
}

impl Default for ExecutionCostProfile {
    fn default() -> Self {
        Self {
            default_cost: 1,
            overrides: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStage {
    pub id: String,
    pub run_id: String,
    pub stage: String,
    pub dependencies: BTreeSet<String>,
    pub ordinal: usize,
    pub cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGraph {
    nodes: BTreeMap<String, ExecutionStage>,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    pub fn from_governance(
        governance: &GovernanceGraph,
        cost_profile: ExecutionCostProfile,
    ) -> Result<Self> {
        let mut graph = Self::new();
        for stage in governance.stages() {
            let cost = cost_profile.cost_for(&stage.id);
            graph.add_stage_with_id(
                stage.id.clone(),
                stage.run_id.clone(),
                stage.stage.clone(),
                stage.dependencies.iter().cloned().collect(),
                cost,
            )?;
        }
        Ok(graph)
    }

    pub fn add_stage(
        &mut self,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        dependencies: Vec<String>,
        cost: u64,
    ) -> Result<String> {
        let run_id = run_id.into();
        let stage = stage.into();
        let stage_id = format!("{run_id}:{stage}");
        self.add_stage_with_id(stage_id.clone(), run_id, stage, dependencies, cost)?;
        Ok(stage_id)
    }

    pub fn add_stage_with_id(
        &mut self,
        stage_id: String,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        dependencies: Vec<String>,
        cost: u64,
    ) -> Result<()> {
        if self.nodes.contains_key(&stage_id) {
            return Err(TorchError::Experiment {
                op: "execution.graph.add_stage",
                msg: format!("duplicate stage id {stage_id}"),
            });
        }
        let ordinal = self.nodes.len();
        let node = ExecutionStage {
            id: stage_id.clone(),
            run_id: run_id.into(),
            stage: stage.into(),
            dependencies: dependencies.into_iter().collect(),
            ordinal,
            cost: cost.max(1),
        };
        self.nodes.insert(stage_id, node);
        Ok(())
    }

    pub fn stages(&self) -> impl Iterator<Item = &ExecutionStage> {
        self.nodes.values()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionPlanEntry {
    pub stage_id: String,
    pub run_id: String,
    pub stage: String,
    pub dependencies: Vec<String>,
    pub priority: u64,
    pub ordinal: usize,
    pub lane: usize,
    pub start_tick: u64,
    pub end_tick: u64,
    pub cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub seed: u64,
    pub generated_at_unix: u64,
    pub entries: Vec<ExecutionPlanEntry>,
    pub total_lanes: usize,
    pub total_stages: usize,
    pub makespan_ticks: u64,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlanner {
    seed: u64,
    graph: ExecutionGraph,
    max_lanes: usize,
}

impl ExecutionPlanner {
    pub fn new(seed: u64, graph: ExecutionGraph) -> Self {
        Self {
            seed,
            graph,
            max_lanes: 1,
        }
    }

    pub fn with_max_lanes(mut self, lanes: usize) -> Self {
        self.max_lanes = lanes.max(1);
        self
    }

    pub fn plan(&self) -> Result<ExecutionPlan> {
        if self.graph.nodes.is_empty() {
            return Ok(ExecutionPlan {
                seed: self.seed,
                generated_at_unix: now_unix_seconds()?,
                entries: Vec::new(),
                total_lanes: self.max_lanes,
                total_stages: 0,
                makespan_ticks: 0,
            });
        }

        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (stage_id, stage) in &self.graph.nodes {
            in_degree.insert(stage_id.clone(), stage.dependencies.len());
            for dep in &stage.dependencies {
                if !self.graph.nodes.contains_key(dep) {
                    return Err(TorchError::Experiment {
                        op: "execution.plan",
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

        let mut lane_allocator = LaneAllocator::new(self.max_lanes);
        let mut completion: BTreeMap<String, u64> = BTreeMap::new();
        let mut entries = Vec::with_capacity(self.graph.nodes.len());
        let mut makespan = 0u64;

        while let Some(item) = heap.pop() {
            let node = self
                .graph
                .nodes
                .get(&item.stage_id)
                .expect("node exists");
            let earliest_start = node
                .dependencies
                .iter()
                .filter_map(|dep| completion.get(dep))
                .copied()
                .max()
                .unwrap_or(0);
            let (lane, lane_ready) = lane_allocator.next_lane();
            let start_tick = earliest_start.max(lane_ready);
            let end_tick = start_tick.saturating_add(node.cost);
            lane_allocator.update_lane(lane, end_tick);
            completion.insert(node.id.clone(), end_tick);
            makespan = makespan.max(end_tick);

            entries.push(ExecutionPlanEntry {
                stage_id: node.id.clone(),
                run_id: node.run_id.clone(),
                stage: node.stage.clone(),
                dependencies: node.dependencies.iter().cloned().collect(),
                priority: item.priority,
                ordinal: item.ordinal,
                lane,
                start_tick,
                end_tick,
                cost: node.cost,
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

        if entries.len() != self.graph.nodes.len() {
            return Err(TorchError::Experiment {
                op: "execution.plan",
                msg: "cycle detected in execution graph".to_string(),
            });
        }

        entries.sort_by(|a, b| match a.start_tick.cmp(&b.start_tick) {
            Ordering::Equal => a.lane.cmp(&b.lane),
            ordering => ordering,
        });

        Ok(ExecutionPlan {
            seed: self.seed,
            generated_at_unix: now_unix_seconds()?,
            total_lanes: self.max_lanes,
            total_stages: entries.len(),
            makespan_ticks: makespan,
            entries,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionAction {
    Started,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLedgerEvent {
    pub index: u64,
    pub timestamp_unix: u64,
    pub stage_id: String,
    pub run_id: String,
    pub stage: String,
    pub action: ExecutionAction,
    pub message: String,
    pub lane: usize,
    pub start_tick: u64,
    pub end_tick: u64,
    pub prev_hash: Option<String>,
    pub hash: String,
}

impl ExecutionLedgerEvent {
    pub fn new(
        stage_id: impl Into<String>,
        run_id: impl Into<String>,
        stage: impl Into<String>,
        action: ExecutionAction,
        message: impl Into<String>,
        lane: usize,
        start_tick: u64,
        end_tick: u64,
    ) -> Self {
        Self {
            index: 0,
            timestamp_unix: 0,
            stage_id: stage_id.into(),
            run_id: run_id.into(),
            stage: stage.into(),
            action,
            message: message.into(),
            lane,
            start_tick,
            end_tick,
            prev_hash: None,
            hash: String::new(),
        }
    }
}

#[derive(Debug)]
pub struct ExecutionLedger {
    path: std::path::PathBuf,
    writer: std::io::BufWriter<std::fs::File>,
    last_hash: Option<Hash>,
    next_index: u64,
}

impl ExecutionLedger {
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| TorchError::Experiment {
                op: "execution_ledger.open",
                msg: format!("failed to create ledger dir {}: {err}", parent.display()),
            })?;
        }
        let mut last_hash = None;
        let mut next_index = 0;
        if path.exists() {
            let file = std::fs::File::open(&path).map_err(|err| TorchError::Experiment {
                op: "execution_ledger.open",
                msg: format!("failed to open ledger {}: {err}", path.display()),
            })?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line.map_err(|err| TorchError::Experiment {
                    op: "execution_ledger.open",
                    msg: format!("failed to read ledger {}: {err}", path.display()),
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let event: ExecutionLedgerEvent =
                    serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                        op: "execution_ledger.open",
                        msg: format!("failed to parse ledger {}: {err}", path.display()),
                    })?;
                let hash = Hash::from_hex(&event.hash).map_err(|err| TorchError::Experiment {
                    op: "execution_ledger.open",
                    msg: format!("failed to parse ledger hash {}: {err}", event.hash),
                })?;
                last_hash = Some(hash);
                next_index = next_index.max(event.index + 1);
            }
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| TorchError::Experiment {
                op: "execution_ledger.open",
                msg: format!("failed to open ledger {}: {err}", path.display()),
            })?;
        Ok(Self {
            path,
            writer: std::io::BufWriter::new(file),
            last_hash,
            next_index,
        })
    }

    pub fn record(&mut self, mut event: ExecutionLedgerEvent) -> Result<ExecutionLedgerEvent> {
        event.index = self.next_index;
        event.timestamp_unix = now_unix_seconds()?;
        event.prev_hash = self.last_hash.map(|hash| hash.to_hex().to_string());
        let hash = hash_ledger_event(&event)?;
        event.hash = hash.to_hex().to_string();
        self.last_hash = Some(hash);
        self.next_index = self.next_index.saturating_add(1);
        let serialized = serde_json::to_string(&event).map_err(|err| TorchError::Experiment {
            op: "execution_ledger.record",
            msg: format!("failed to serialize execution ledger event: {err}"),
        })?;
        self.writer
            .write_all(serialized.as_bytes())
            .and_then(|_| self.writer.write_all(b"\n"))
            .map_err(|err| TorchError::Experiment {
                op: "execution_ledger.record",
                msg: format!("failed to write execution ledger event: {err}"),
            })?;
        self.writer.flush().map_err(|err| TorchError::Experiment {
            op: "execution_ledger.record",
            msg: format!("failed to flush execution ledger: {err}"),
        })?;
        Ok(event)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionStatus {
    Success,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStageReport {
    pub stage_id: String,
    pub run_id: String,
    pub stage: String,
    pub status: ExecutionStatus,
    pub message: String,
    pub lane: usize,
    pub start_tick: u64,
    pub end_tick: u64,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionRunStatus {
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRunReport {
    pub total_stages: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub status: ExecutionRunStatus,
    pub duration_ms: f64,
    pub stages: Vec<ExecutionStageReport>,
}

pub trait ExecutionTask: Send + Sync {
    fn execute(&self) -> Result<ExecutionStatus>;
}

#[derive(Default, Clone)]
pub struct ExecutionRegistry {
    tasks: BTreeMap<String, Arc<dyn ExecutionTask>>,
}

impl ExecutionRegistry {
    pub fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, stage_id: impl Into<String>, task: Arc<dyn ExecutionTask>) {
        self.tasks.insert(stage_id.into(), task);
    }

    pub fn get(&self, stage_id: &str) -> Option<Arc<dyn ExecutionTask>> {
        self.tasks.get(stage_id).cloned()
    }
}

#[derive(Debug)]
pub struct ExecutionEngine {
    ledger: Option<ExecutionLedger>,
}

impl ExecutionEngine {
    pub fn new(ledger: Option<ExecutionLedger>) -> Self {
        Self { ledger }
    }

    pub fn run<S: TelemetrySink>(
        &mut self,
        plan: &ExecutionPlan,
        registry: &ExecutionRegistry,
        telemetry: Option<&TelemetryRecorder<S>>,
    ) -> Result<ExecutionRunReport> {
        if plan.entries.is_empty() {
            return Ok(ExecutionRunReport {
                total_stages: 0,
                succeeded: 0,
                failed: 0,
                status: ExecutionRunStatus::Completed,
                duration_ms: 0.0,
                stages: Vec::new(),
            });
        }

        let start = Instant::now();
        let mut grouped: BTreeMap<u64, Vec<&ExecutionPlanEntry>> = BTreeMap::new();
        for entry in &plan.entries {
            grouped.entry(entry.start_tick).or_default().push(entry);
        }

        let mut completed: BTreeSet<String> = BTreeSet::new();
        let mut reports: Vec<ExecutionStageReport> = Vec::with_capacity(plan.entries.len());
        let mut failures = 0usize;

        info!(
            "execution engine start: stages={} lanes={} makespan_ticks={}",
            plan.total_stages, plan.total_lanes, plan.makespan_ticks
        );

        for (tick, entries) in grouped {
            let mut entries = entries;
            entries.sort_by(|a, b| a.lane.cmp(&b.lane));

            for entry in &entries {
                for dep in &entry.dependencies {
                    if !completed.contains(dep) {
                        return Err(TorchError::Experiment {
                            op: "execution.run",
                            msg: format!(
                                "stage {} scheduled before dependency {} completed",
                                entry.stage_id, dep
                            ),
                        });
                    }
                }
            }

            let results = thread::scope(|scope| {
                let mut handles = Vec::with_capacity(entries.len());
                for entry in &entries {
                    let task = registry.get(&entry.stage_id).ok_or_else(|| {
                        TorchError::Experiment {
                            op: "execution.run",
                            msg: format!("missing task for stage {}", entry.stage_id),
                        }
                    })?;
                    let entry = (*entry).clone();
                    let handle = scope.spawn(move || {
                        let stage_start = Instant::now();
                        let status = task.execute();
                        let duration_ms = stage_start.elapsed().as_secs_f64() * 1000.0;
                        (entry, status, duration_ms)
                    });
                    handles.push(handle);
                }
                let mut results = Vec::with_capacity(handles.len());
                for handle in handles {
                    let result = handle.join().map_err(|_| TorchError::Experiment {
                        op: "execution.run",
                        msg: "execution task panicked".to_string(),
                    })?;
                    results.push(result);
                }
                Ok::<_, TorchError>(results)
            })?;

            for (entry, status, duration_ms) in results {
                let status = match status {
                    Ok(ExecutionStatus::Success) => ExecutionStatus::Success,
                    Ok(ExecutionStatus::Failed) => ExecutionStatus::Failed,
                    Err(err) => {
                        warn!("execution stage {} failed: {err:?}", entry.stage_id);
                        ExecutionStatus::Failed
                    }
                };
                if status == ExecutionStatus::Failed {
                    failures = failures.saturating_add(1);
                }

                let report = ExecutionStageReport {
                    stage_id: entry.stage_id.clone(),
                    run_id: entry.run_id.clone(),
                    stage: entry.stage.clone(),
                    status: status.clone(),
                    message: if status == ExecutionStatus::Success {
                        "completed".to_string()
                    } else {
                        "failed".to_string()
                    },
                    lane: entry.lane,
                    start_tick: entry.start_tick,
                    end_tick: entry.end_tick,
                    duration_ms,
                };

                if let Some(ledger) = self.ledger.as_mut() {
                    let action = if status == ExecutionStatus::Success {
                        ExecutionAction::Completed
                    } else {
                        ExecutionAction::Failed
                    };
                    ledger.record(ExecutionLedgerEvent::new(
                        report.stage_id.clone(),
                        report.run_id.clone(),
                        report.stage.clone(),
                        ExecutionAction::Started,
                        format!("tick={tick}"),
                        report.lane,
                        report.start_tick,
                        report.end_tick,
                    ))?;
                    ledger.record(ExecutionLedgerEvent::new(
                        report.stage_id.clone(),
                        report.run_id.clone(),
                        report.stage.clone(),
                        action,
                        report.message.clone(),
                        report.lane,
                        report.start_tick,
                        report.end_tick,
                    ))?;
                }

                if let Some(recorder) = telemetry {
                    let mut event = TelemetryEvent::new("execution.stage", duration_ms)
                        .with_tag("stage_id", report.stage_id.clone())
                        .with_tag("run_id", report.run_id.clone())
                        .with_tag("lane", report.lane.to_string())
                        .with_tag("start_tick", report.start_tick.to_string())
                        .with_tag("end_tick", report.end_tick.to_string())
                        .with_tag(
                            "status",
                            match report.status {
                                ExecutionStatus::Success => "success",
                                ExecutionStatus::Failed => "failed",
                            },
                        );
                    event = event.with_tag("tick", tick.to_string());
                    recorder.record(event).map_err(|err| TorchError::Experiment {
                        op: "execution.telemetry",
                        msg: format!("failed to record execution telemetry: {err}"),
                    })?;
                }

                completed.insert(report.stage_id.clone());
                reports.push(report);
            }
        }

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        let succeeded = reports.len().saturating_sub(failures);
        let status = if failures > 0 {
            ExecutionRunStatus::Failed
        } else {
            ExecutionRunStatus::Completed
        };
        info!(
            "execution engine completed: status={:?} duration_ms={:.3} failures={}",
            status, duration_ms, failures
        );

        Ok(ExecutionRunReport {
            total_stages: reports.len(),
            succeeded,
            failed: failures,
            status,
            duration_ms,
            stages: reports,
        })
    }
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

#[derive(Debug, Clone, Copy)]
struct LaneState {
    time: u64,
    lane: usize,
}

struct LaneAllocator {
    tree: Vec<LaneState>,
    size: usize,
}

impl LaneAllocator {
    fn new(lanes: usize) -> Self {
        let size = lanes.next_power_of_two().max(1);
        let mut tree = vec![
            LaneState {
                time: u64::MAX,
                lane: usize::MAX,
            };
            2 * size
        ];
        for idx in 0..size {
            let node = if idx < lanes {
                LaneState { time: 0, lane: idx }
            } else {
                LaneState {
                    time: u64::MAX,
                    lane: usize::MAX,
                }
            };
            tree[size + idx] = node;
        }
        for idx in (1..size).rev() {
            tree[idx] = min_lane(tree[idx * 2], tree[idx * 2 + 1]);
        }
        Self { tree, size }
    }

    fn next_lane(&self) -> (usize, u64) {
        let state = self.tree[1];
        (state.lane, state.time)
    }

    fn update_lane(&mut self, lane: usize, time: u64) {
        let mut idx = self.size + lane;
        self.tree[idx] = LaneState { time, lane };
        while idx > 1 {
            idx /= 2;
            self.tree[idx] = min_lane(self.tree[idx * 2], self.tree[idx * 2 + 1]);
        }
    }
}

fn min_lane(left: LaneState, right: LaneState) -> LaneState {
    match left.time.cmp(&right.time) {
        Ordering::Less => left,
        Ordering::Greater => right,
        Ordering::Equal => {
            if left.lane <= right.lane {
                left
            } else {
                right
            }
        }
    }
}

fn hash_priority(seed: u64, stage_id: &str) -> u64 {
    let payload = format!("{seed}:{stage_id}");
    let hash = blake3::hash(payload.as_bytes());
    priority_from_hash(hash)
}

fn priority_from_hash(hash: Hash) -> u64 {
    let bytes = hash.as_bytes();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[0..8]);
    u64::from_le_bytes(buf)
}

fn hash_ledger_event(event: &ExecutionLedgerEvent) -> Result<Hash> {
    let payload = ExecutionLedgerPayload {
        index: event.index,
        timestamp_unix: event.timestamp_unix,
        stage_id: &event.stage_id,
        run_id: &event.run_id,
        stage: &event.stage,
        action: &event.action,
        message: &event.message,
        lane: event.lane,
        start_tick: event.start_tick,
        end_tick: event.end_tick,
        prev_hash: &event.prev_hash,
    };
    let payload = serde_json::to_vec(&payload).map_err(|err| TorchError::Experiment {
        op: "execution_ledger.hash_event",
        msg: format!("failed to serialize ledger payload: {err}"),
    })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&payload);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct ExecutionLedgerPayload<'a> {
    index: u64,
    timestamp_unix: u64,
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    action: &'a ExecutionAction,
    message: &'a str,
    lane: usize,
    start_tick: u64,
    end_tick: u64,
    prev_hash: &'a Option<String>,
}

fn now_unix_seconds() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|err| TorchError::Experiment {
            op: "execution.plan_time",
            msg: format!("system clock error: {err}"),
        })
}
