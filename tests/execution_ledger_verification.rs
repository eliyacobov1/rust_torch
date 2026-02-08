use std::path::PathBuf;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::audit::MerkleAccumulator;
use rustorch::execution::{
    verify_execution_ledger, ExecutionAction, ExecutionGraph, ExecutionLedger,
    ExecutionLedgerEvent, ExecutionLedgerVerificationConfig, ExecutionLedgerVerificationStatus,
    ExecutionPlanner, ExecutionReplayCursor,
};

fn temp_ledger_path(name: &str) -> PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(6)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_execution_ledger_{name}_{suffix}.jsonl"))
}

fn write_sample_ledger(path: &PathBuf) {
    let mut ledger = ExecutionLedger::open(path).expect("execution ledger");
    let entries = vec![
        ExecutionLedgerEvent::new(
            "run-1:prepare",
            "run-1",
            "prepare",
            ExecutionAction::Started,
            "start",
            0,
            0,
            1,
        ),
        ExecutionLedgerEvent::new(
            "run-1:prepare",
            "run-1",
            "prepare",
            ExecutionAction::Completed,
            "done",
            0,
            0,
            1,
        ),
        ExecutionLedgerEvent::new(
            "run-1:train",
            "run-1",
            "train",
            ExecutionAction::Started,
            "start",
            1,
            1,
            3,
        ),
        ExecutionLedgerEvent::new(
            "run-1:train",
            "run-1",
            "train",
            ExecutionAction::Completed,
            "done",
            1,
            1,
            3,
        ),
    ];
    for event in entries {
        ledger.record(event).expect("record event");
    }
}

#[test]
fn execution_ledger_verification_accepts_valid_chain() {
    let path = temp_ledger_path("valid");
    write_sample_ledger(&path);

    let contents = std::fs::read_to_string(&path).expect("read");
    let mut accumulator = MerkleAccumulator::new();
    for line in contents.lines() {
        let event: ExecutionLedgerEvent = serde_json::from_str(line).expect("parse");
        let hash = blake3::Hash::from_hex(&event.hash).expect("hash");
        accumulator.append(hash);
    }

    let mut config = ExecutionLedgerVerificationConfig::default();
    config.expected_root = accumulator.root_hex();
    let report = verify_execution_ledger(&path, &config).expect("verify");
    assert_eq!(report.status, ExecutionLedgerVerificationStatus::Valid);
    assert!(report.matches_expected_root.unwrap_or(false));
}

#[test]
fn execution_ledger_verification_detects_tamper() {
    let path = temp_ledger_path("tamper");
    write_sample_ledger(&path);

    let contents = std::fs::read_to_string(&path).expect("read");
    let mut lines = Vec::new();
    for (idx, line) in contents.lines().enumerate() {
        if idx == 1 {
            let mut event: ExecutionLedgerEvent = serde_json::from_str(line).expect("parse");
            event.message = "tampered".to_string();
            lines.push(serde_json::to_string(&event).expect("serialize"));
        } else {
            lines.push(line.to_string());
        }
    }
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).expect("write");

    let report = verify_execution_ledger(&path, &ExecutionLedgerVerificationConfig::default())
        .expect("verify");
    assert_eq!(report.status, ExecutionLedgerVerificationStatus::Invalid);
    assert!(!report.issues.is_empty());
}

#[test]
fn execution_replay_cursor_validates_plan_ordering() {
    let mut graph = ExecutionGraph::new();
    let stage_a = graph
        .add_stage("run-1", "prepare", Vec::new(), 1)
        .expect("stage a");
    graph
        .add_stage("run-1", "train", vec![stage_a], 1)
        .expect("stage b");
    let plan = ExecutionPlanner::new(42, graph)
        .with_max_lanes(2)
        .plan()
        .expect("plan");

    let mut cursor = ExecutionReplayCursor::new(plan.clone());
    for entry in &plan.entries {
        let start = ExecutionLedgerEvent::new(
            entry.stage_id.clone(),
            entry.run_id.clone(),
            entry.stage.clone(),
            ExecutionAction::Started,
            "start",
            entry.lane,
            entry.start_tick,
            entry.end_tick,
        );
        cursor.expect_event(&start).expect("start ok");
        let finish = ExecutionLedgerEvent::new(
            entry.stage_id.clone(),
            entry.run_id.clone(),
            entry.stage.clone(),
            ExecutionAction::Completed,
            "done",
            entry.lane,
            entry.start_tick,
            entry.end_tick,
        );
        cursor.expect_event(&finish).expect("finish ok");
    }
    assert_eq!(cursor.remaining(), 0);
    assert!(!cursor.has_inflight());

    let mut cursor = ExecutionReplayCursor::new(plan.clone());
    let wrong_entry = plan.entries.get(1).expect("second entry");
    let wrong_event = ExecutionLedgerEvent::new(
        wrong_entry.stage_id.clone(),
        wrong_entry.run_id.clone(),
        wrong_entry.stage.clone(),
        ExecutionAction::Started,
        "start",
        wrong_entry.lane,
        wrong_entry.start_tick,
        wrong_entry.end_tick,
    );
    assert!(cursor.expect_event(&wrong_event).is_err());
}
