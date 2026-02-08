use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use blake3::Hash;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, TorchError};
use crate::execution::ExecutionPlan;
use crate::governance::GovernancePlan;

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

    pub fn leaves(&self) -> &[Hash] {
        &self.leaves
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

    pub fn leaves(&self) -> &[Hash] {
        self.merkle.leaves()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDigest {
    pub plan_hash: String,
    pub entries_root: Option<String>,
    pub entries: usize,
}

#[derive(Debug, Serialize)]
struct GovernancePlanAuditPayload<'a> {
    seed: u64,
    generated_at_unix: u64,
    total_waves: usize,
    total_stages: usize,
    plan_hash: String,
    entries_root: Option<String>,
    entries: Vec<GovernancePlanAuditEntry<'a>>,
}

#[derive(Debug, Serialize)]
struct GovernancePlanAuditEntry<'a> {
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    wave: usize,
    lane: usize,
}

pub fn record_governance_plan(audit_log: &mut AuditLog, plan: &GovernancePlan) -> Result<AuditEvent> {
    let digest = governance_plan_digest(plan)?;
    let entries = plan
        .entries
        .iter()
        .map(|entry| GovernancePlanAuditEntry {
            stage_id: entry.stage_id.as_str(),
            run_id: entry.run_id.as_str(),
            stage: entry.stage.as_str(),
            wave: entry.wave,
            lane: entry.lane,
        })
        .collect();
    let payload = GovernancePlanAuditPayload {
        seed: plan.seed,
        generated_at_unix: plan.generated_at_unix,
        total_waves: plan.total_waves,
        total_stages: plan.total_stages,
        plan_hash: digest.plan_hash,
        entries_root: digest.entries_root,
        entries,
    };
    let message = serde_json::to_string(&payload).map_err(|err| TorchError::Experiment {
        op: "audit_log.record_governance_plan",
        msg: format!("failed to serialize governance plan: {err}"),
    })?;
    audit_log.record(AuditEvent::new(
        AuditScope::Store,
        None,
        Some("governance.plan".to_string()),
        AuditStatus::Completed,
        message,
        0,
    ))
}

#[derive(Debug, Serialize)]
struct ExecutionPlanAuditPayload<'a> {
    seed: u64,
    generated_at_unix: u64,
    total_lanes: usize,
    total_stages: usize,
    makespan_ticks: u64,
    plan_hash: String,
    entries_root: Option<String>,
    entries: Vec<ExecutionPlanAuditEntry<'a>>,
}

#[derive(Debug, Serialize)]
struct ExecutionPlanAuditEntry<'a> {
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    lane: usize,
    start_tick: u64,
    end_tick: u64,
    cost: u64,
}

pub fn record_execution_plan(audit_log: &mut AuditLog, plan: &ExecutionPlan) -> Result<AuditEvent> {
    let digest = execution_plan_digest(plan)?;
    let entries = plan
        .entries
        .iter()
        .map(|entry| ExecutionPlanAuditEntry {
            stage_id: entry.stage_id.as_str(),
            run_id: entry.run_id.as_str(),
            stage: entry.stage.as_str(),
            lane: entry.lane,
            start_tick: entry.start_tick,
            end_tick: entry.end_tick,
            cost: entry.cost,
        })
        .collect();
    let payload = ExecutionPlanAuditPayload {
        seed: plan.seed,
        generated_at_unix: plan.generated_at_unix,
        total_lanes: plan.total_lanes,
        total_stages: plan.total_stages,
        makespan_ticks: plan.makespan_ticks,
        plan_hash: digest.plan_hash,
        entries_root: digest.entries_root,
        entries,
    };
    let message = serde_json::to_string(&payload).map_err(|err| TorchError::Experiment {
        op: "audit_log.record_execution_plan",
        msg: format!("failed to serialize execution plan: {err}"),
    })?;
    audit_log.record(AuditEvent::new(
        AuditScope::Store,
        None,
        Some("execution.plan".to_string()),
        AuditStatus::Completed,
        message,
        0,
    ))
}

#[derive(Debug, Serialize)]
struct GovernancePlanHashPayload<'a> {
    seed: u64,
    total_waves: usize,
    total_stages: usize,
    entries: Vec<GovernancePlanHashEntry<'a>>,
}

#[derive(Debug, Serialize)]
struct GovernancePlanHashEntry<'a> {
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    priority: u64,
    ordinal: usize,
    wave: usize,
    lane: usize,
    dependencies: &'a [String],
}

#[derive(Debug, Serialize)]
struct ExecutionPlanHashPayload<'a> {
    seed: u64,
    total_lanes: usize,
    total_stages: usize,
    makespan_ticks: u64,
    entries: Vec<ExecutionPlanHashEntry<'a>>,
}

#[derive(Debug, Serialize)]
struct ExecutionPlanHashEntry<'a> {
    stage_id: &'a str,
    run_id: &'a str,
    stage: &'a str,
    dependencies: &'a [String],
    priority: u64,
    ordinal: usize,
    lane: usize,
    start_tick: u64,
    end_tick: u64,
    cost: u64,
}

pub fn governance_plan_digest(plan: &GovernancePlan) -> Result<PlanDigest> {
    let mut merkle = MerkleAccumulator::new();
    let entries = plan
        .entries
        .iter()
        .map(|entry| {
            let entry_payload = GovernancePlanHashEntry {
                stage_id: entry.stage_id.as_str(),
                run_id: entry.run_id.as_str(),
                stage: entry.stage.as_str(),
                priority: entry.priority,
                ordinal: entry.ordinal,
                wave: entry.wave,
                lane: entry.lane,
                dependencies: entry.dependencies.as_slice(),
            };
            let entry_hash = hash_plan_payload(&entry_payload, "audit.governance_plan_entry")?;
            merkle.append(entry_hash);
            Ok(entry_payload)
        })
        .collect::<Result<Vec<_>>>()?;
    let payload = GovernancePlanHashPayload {
        seed: plan.seed,
        total_waves: plan.total_waves,
        total_stages: plan.total_stages,
        entries,
    };
    let hash = hash_plan_payload(&payload, "audit.governance_plan")?;
    Ok(PlanDigest {
        plan_hash: hash.to_hex().to_string(),
        entries_root: merkle.root_hex(),
        entries: plan.entries.len(),
    })
}

pub fn execution_plan_digest(plan: &ExecutionPlan) -> Result<PlanDigest> {
    let mut merkle = MerkleAccumulator::new();
    let entries = plan
        .entries
        .iter()
        .map(|entry| {
            let entry_payload = ExecutionPlanHashEntry {
                stage_id: entry.stage_id.as_str(),
                run_id: entry.run_id.as_str(),
                stage: entry.stage.as_str(),
                dependencies: entry.dependencies.as_slice(),
                priority: entry.priority,
                ordinal: entry.ordinal,
                lane: entry.lane,
                start_tick: entry.start_tick,
                end_tick: entry.end_tick,
                cost: entry.cost,
            };
            let entry_hash = hash_plan_payload(&entry_payload, "audit.execution_plan_entry")?;
            merkle.append(entry_hash);
            Ok(entry_payload)
        })
        .collect::<Result<Vec<_>>>()?;
    let payload = ExecutionPlanHashPayload {
        seed: plan.seed,
        total_lanes: plan.total_lanes,
        total_stages: plan.total_stages,
        makespan_ticks: plan.makespan_ticks,
        entries,
    };
    let hash = hash_plan_payload(&payload, "audit.execution_plan")?;
    Ok(PlanDigest {
        plan_hash: hash.to_hex().to_string(),
        entries_root: merkle.root_hex(),
        entries: plan.entries.len(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditVerificationStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditChainIssueKind {
    ParseError,
    IndexMismatch,
    PrevHashMismatch,
    HashMismatch,
    HashDecodeFailure,
    ExpectedRootMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditChainIssue {
    pub line: usize,
    pub index: Option<u64>,
    pub kind: AuditChainIssueKind,
    pub message: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditProofDirection {
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditProofStep {
    pub sibling_hash: String,
    pub direction: AuditProofDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditProof {
    pub leaf_index: usize,
    pub leaf_hash: String,
    pub root: String,
    pub path: Vec<AuditProofStep>,
    pub valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditVerificationReport {
    pub status: AuditVerificationStatus,
    pub total_events: usize,
    pub issues: Vec<AuditChainIssue>,
    pub merkle_root: Option<String>,
    pub expected_root: Option<String>,
    pub matches_expected_root: Option<bool>,
    pub proofs: Vec<AuditProof>,
    pub duration_ms: f64,
}

impl AuditVerificationReport {
    pub fn from_error(message: impl Into<String>) -> Self {
        Self {
            status: AuditVerificationStatus::Invalid,
            total_events: 0,
            issues: vec![AuditChainIssue {
                line: 0,
                index: None,
                kind: AuditChainIssueKind::ParseError,
                message: message.into(),
                expected: None,
                actual: None,
            }],
            merkle_root: None,
            expected_root: None,
            matches_expected_root: None,
            proofs: Vec::new(),
            duration_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuditVerificationConfig {
    pub verify_hashes: bool,
    pub verify_merkle: bool,
    pub include_proofs: bool,
    pub max_proofs: usize,
    pub expected_root: Option<String>,
}

impl Default for AuditVerificationConfig {
    fn default() -> Self {
        Self {
            verify_hashes: true,
            verify_merkle: true,
            include_proofs: false,
            max_proofs: 5,
            expected_root: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuditLedger {
    pub events: Vec<AuditEvent>,
}

impl AuditLedger {
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|err| TorchError::Experiment {
            op: "audit_ledger.from_path",
            msg: format!("failed to open audit log {}: {err}", path.display()),
        })?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|err| TorchError::Experiment {
                op: "audit_ledger.from_path",
                msg: format!("failed to read audit log {}: {err}", path.display()),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let event: AuditEvent =
                serde_json::from_str(&line).map_err(|err| TorchError::Experiment {
                    op: "audit_ledger.from_path",
                    msg: format!("failed to parse audit event: {err}"),
                })?;
            events.push(event);
        }
        Ok(Self { events })
    }
}

pub fn verify_audit_log(
    path: impl AsRef<Path>,
    config: &AuditVerificationConfig,
) -> Result<AuditVerificationReport> {
    let start = Instant::now();
    let path = path.as_ref();
    let file = File::open(path).map_err(|err| TorchError::Experiment {
        op: "audit_verify",
        msg: format!("failed to open audit log {}: {err}", path.display()),
    })?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    let mut issues = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| TorchError::Experiment {
            op: "audit_verify",
            msg: format!("failed to read audit log {}: {err}", path.display()),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<AuditEvent>(&line) {
            Ok(event) => events.push(event),
            Err(err) => issues.push(AuditChainIssue {
                line: line_idx + 1,
                index: None,
                kind: AuditChainIssueKind::ParseError,
                message: format!("failed to parse audit event: {err}"),
                expected: None,
                actual: None,
            }),
        }
    }

    let mut expected_index = 0_u64;
    let mut prev_hash: Option<String> = None;
    let mut leaf_hashes = Vec::with_capacity(events.len());

    if config.verify_hashes {
        let computed = events
            .par_iter()
            .map(|event| (event.index, event.hash.clone(), hash_event(event)))
            .collect::<Vec<(u64, String, Result<Hash>)>>();
        for (idx, stored_hash, hash_result) in computed {
            match hash_result {
                Ok(hash) => {
                    let expected = hash.to_hex().to_string();
                    if stored_hash != expected {
                        issues.push(AuditChainIssue {
                            line: (idx + 1) as usize,
                            index: Some(idx),
                            kind: AuditChainIssueKind::HashMismatch,
                            message: "audit hash mismatch".to_string(),
                            expected: Some(expected),
                            actual: Some(stored_hash),
                        });
                    }
                }
                Err(err) => issues.push(AuditChainIssue {
                    line: (idx + 1) as usize,
                    index: Some(idx),
                    kind: AuditChainIssueKind::HashMismatch,
                    message: format!("failed to recompute hash: {err}"),
                    expected: None,
                    actual: None,
                }),
            }
        }
    }

    for event in &events {
        if event.index != expected_index {
            issues.push(AuditChainIssue {
                line: (event.index + 1) as usize,
                index: Some(event.index),
                kind: AuditChainIssueKind::IndexMismatch,
                message: "audit index mismatch".to_string(),
                expected: Some(expected_index.to_string()),
                actual: Some(event.index.to_string()),
            });
            expected_index = event.index.saturating_add(1);
        } else {
            expected_index = expected_index.saturating_add(1);
        }

        if event.prev_hash != prev_hash {
            issues.push(AuditChainIssue {
                line: (event.index + 1) as usize,
                index: Some(event.index),
                kind: AuditChainIssueKind::PrevHashMismatch,
                message: "previous hash mismatch".to_string(),
                expected: prev_hash.clone(),
                actual: event.prev_hash.clone(),
            });
        }
        prev_hash = Some(event.hash.clone());

        match Hash::from_hex(&event.hash) {
            Ok(hash) => leaf_hashes.push(hash),
            Err(err) => issues.push(AuditChainIssue {
                line: (event.index + 1) as usize,
                index: Some(event.index),
                kind: AuditChainIssueKind::HashDecodeFailure,
                message: format!("failed to decode hash: {err}"),
                expected: None,
                actual: Some(event.hash.clone()),
            }),
        }
    }

    let should_build_tree =
        config.verify_merkle || config.include_proofs || config.expected_root.is_some();
    let merkle_tree = if should_build_tree {
        MerkleTree::from_leaves(leaf_hashes)
    } else {
        MerkleTree::from_leaves(Vec::new())
    };
    let merkle_root = if config.verify_merkle || config.expected_root.is_some() {
        merkle_tree.root_hex()
    } else {
        None
    };
    let mut matches_expected_root = None;
    if let Some(expected) = &config.expected_root {
        let matches = merkle_root.as_deref() == Some(expected.as_str());
        matches_expected_root = Some(matches);
        if !matches {
            issues.push(AuditChainIssue {
                line: 0,
                index: None,
                kind: AuditChainIssueKind::ExpectedRootMismatch,
                message: "merkle root mismatch".to_string(),
                expected: Some(expected.clone()),
                actual: merkle_root.clone(),
            });
        }
    }

    let proofs = if config.include_proofs {
        merkle_tree
            .sample_proofs(config.max_proofs)
            .into_iter()
            .map(|proof| {
                let mut audit_proof = AuditProof {
                    leaf_index: proof.leaf_index,
                    leaf_hash: proof.leaf_hash,
                    root: proof.root,
                    path: proof.path,
                    valid: false,
                };
                audit_proof.valid = verify_merkle_proof(&audit_proof);
                audit_proof
            })
            .collect()
    } else {
        Vec::new()
    };

    let status = if issues.is_empty() {
        AuditVerificationStatus::Valid
    } else {
        AuditVerificationStatus::Invalid
    };

    Ok(AuditVerificationReport {
        status,
        total_events: events.len(),
        issues,
        merkle_root,
        expected_root: config.expected_root.clone(),
        matches_expected_root,
        proofs,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

fn verify_merkle_proof(proof: &AuditProof) -> bool {
    let mut hash = match Hash::from_hex(&proof.leaf_hash) {
        Ok(value) => value,
        Err(_) => return false,
    };
    for step in &proof.path {
        let sibling = match Hash::from_hex(&step.sibling_hash) {
            Ok(value) => value,
            Err(_) => return false,
        };
        hash = match step.direction {
            AuditProofDirection::Left => hash_pair(sibling, hash),
            AuditProofDirection::Right => hash_pair(hash, sibling),
        };
    }
    hash.to_hex().to_string() == proof.root
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

fn hash_plan_payload<T: Serialize>(payload: &T, op: &'static str) -> Result<Hash> {
    let serialized = serde_json::to_vec(payload).map_err(|err| TorchError::Experiment {
        op,
        msg: format!("failed to serialize plan payload: {err}"),
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

#[derive(Debug, Clone)]
struct MerkleProofRaw {
    leaf_index: usize,
    leaf_hash: String,
    root: String,
    path: Vec<AuditProofStep>,
}

#[derive(Debug, Clone)]
struct MerkleTree {
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    fn from_leaves(leaves: Vec<Hash>) -> Self {
        if leaves.is_empty() {
            return Self { levels: Vec::new() };
        }
        let mut levels = Vec::new();
        let mut current = leaves;
        levels.push(current.clone());
        while current.len() > 1 {
            if current.len() % 2 == 1 {
                let last = *current.last().expect("non-empty");
                current.push(last);
            }
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks(2) {
                next.push(hash_pair(pair[0], pair[1]));
            }
            levels.push(next.clone());
            current = next;
        }
        Self { levels }
    }

    fn root_hex(&self) -> Option<String> {
        self.levels
            .last()
            .and_then(|level| level.first())
            .map(|hash| hash.to_hex().to_string())
    }

    fn proof(&self, index: usize) -> Option<MerkleProofRaw> {
        let root = self.root_hex()?;
        let mut idx = index;
        let leaf = self.levels.first()?.get(index)?.to_hex().to_string();
        let mut path = Vec::new();
        for level in &self.levels {
            if level.len() <= 1 {
                break;
            }
            let is_right = idx % 2 == 1;
            let sibling_idx = if is_right { idx - 1 } else { idx + 1 };
            let sibling = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                level[idx]
            };
            let direction = if is_right {
                AuditProofDirection::Left
            } else {
                AuditProofDirection::Right
            };
            path.push(AuditProofStep {
                sibling_hash: sibling.to_hex().to_string(),
                direction,
            });
            idx /= 2;
        }
        Some(MerkleProofRaw {
            leaf_index: index,
            leaf_hash: leaf,
            root,
            path,
        })
    }

    fn sample_proofs(&self, max_proofs: usize) -> Vec<MerkleProofRaw> {
        let Some(level) = self.levels.first() else {
            return Vec::new();
        };
        let total = level.len();
        if total == 0 || max_proofs == 0 {
            return Vec::new();
        }
        let sample_count = max_proofs.min(total);
        let step = (total as f64 / sample_count as f64).ceil() as usize;
        let mut proofs = Vec::new();
        let mut idx = 0;
        while idx < total && proofs.len() < sample_count {
            if let Some(proof) = self.proof(idx) {
                proofs.push(proof);
            }
            idx = idx.saturating_add(step.max(1));
        }
        proofs
    }
}
