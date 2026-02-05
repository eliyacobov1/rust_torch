use std::sync::Arc;
use std::thread;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::audit::{AuditEvent, MerkleAccumulator};
use rustorch::experiment::{ExperimentStore, RunGovernanceConfig, RunGovernanceReport};

fn read_audit_events(path: &std::path::Path) -> Vec<AuditEvent> {
    let mut events = Vec::new();
    if let Ok(contents) = std::fs::read_to_string(path) {
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let event: AuditEvent = serde_json::from_str(line).expect("audit event parse");
            events.push(event);
        }
    }
    events
}

fn assert_audit_chain(events: &[AuditEvent]) {
    for (idx, event) in events.iter().enumerate() {
        assert_eq!(event.index, idx as u64);
        if idx == 0 {
            assert!(event.prev_hash.is_none());
        } else {
            assert_eq!(
                event.prev_hash.as_deref(),
                Some(events[idx - 1].hash.as_str())
            );
        }
    }
}

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_governance_test_{suffix}"))
}

fn create_run(store: &ExperimentStore, name: &str) {
    let mut run = store
        .create_run(name, serde_json::json!({}), Vec::new())
        .expect("run");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
}

fn schedule_run_ids(report: &RunGovernanceReport) -> Vec<String> {
    report
        .schedule_entries
        .iter()
        .map(|entry| entry.run_id.clone())
        .collect()
}

#[test]
fn governance_quarantines_invalid_summary() {
    let root = temp_root();
    let store = ExperimentStore::new(&root).expect("store");
    let mut run = store
        .create_run("quarantine", serde_json::json!({}), Vec::new())
        .expect("run");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
    let run_id = run.metadata().id.clone();
    let summary_path = store.root().join(&run_id).join("run_summary.json");
    let mut summary_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).expect("read summary"))
            .expect("parse summary");
    summary_value["duration_secs"] = serde_json::json!(-1.0);
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary_value).expect("serialize"),
    )
    .expect("write summary");

    let mut config = RunGovernanceConfig::default();
    config.quarantine = true;
    let report = store.validate_runs(&config).expect("report");

    assert_eq!(report.summary.total_runs, 1);
    assert_eq!(report.summary.invalid_runs, 1);
    assert_eq!(report.summary.quarantined_runs, 1);
    assert_eq!(report.remediation.total_tickets, 1);
    assert_eq!(report.remediation.quarantined, 1);
    let result = report.results.first().expect("result");
    assert!(result.quarantined);
    assert_eq!(report.schedule_entries.len(), 1);
    let quarantine_path = result.quarantine_path.as_ref().expect("path");
    assert!(quarantine_path.exists());
    assert!(!store.root().join(&run_id).exists());
}

#[test]
fn governance_schedule_is_deterministic() {
    let root = temp_root();
    let store = ExperimentStore::new(&root).expect("store");
    for idx in 0..6 {
        create_run(&store, &format!("schedule-{idx}"));
    }

    let mut config = RunGovernanceConfig::default();
    config.deterministic_seed = Some(42);
    let report_a = store.validate_runs(&config).expect("report");
    let report_b = store.validate_runs(&config).expect("report");

    assert_eq!(report_a.schedule_seed, 42);
    assert_eq!(report_b.schedule_seed, 42);
    assert_eq!(report_a.schedule_entries.len(), 6);
    assert_eq!(report_b.schedule_entries.len(), 6);
    assert_eq!(schedule_run_ids(&report_a), schedule_run_ids(&report_b));
}

#[test]
fn governance_parallel_validation_stress() {
    let root = temp_root();
    let store = Arc::new(ExperimentStore::new(&root).expect("store"));
    let mut handles = Vec::new();
    let runs_per_thread = 6;
    for idx in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for run_idx in 0..runs_per_thread {
                create_run(&store, &format!("stress-{idx}-{run_idx}"));
            }
        }));
    }
    for handle in handles {
        handle.join().expect("join");
    }

    let mut config = RunGovernanceConfig::default();
    config.max_workers = 4;
    config.audit_log = true;
    config.audit_log_path = Some(store.root().join("audit").join("stress.jsonl"));
    config.audit_include_proofs = true;
    config.audit_max_proofs = 3;
    let report = store.validate_runs(&config).expect("report");
    let expected = runs_per_thread * 4;

    assert_eq!(report.summary.total_runs, expected);
    assert_eq!(report.summary.valid_runs, expected);
    assert_eq!(report.summary.invalid_runs, 0);
    assert_eq!(report.remediation.total_tickets, 0);
    assert_eq!(report.schedule_entries.len(), expected);

    let audit_path = report.audit_log_path.expect("audit path");
    let events = read_audit_events(&audit_path);
    assert!(!events.is_empty());
    assert_audit_chain(&events);
    let mut merkle = MerkleAccumulator::new();
    for event in &events {
        let hash = blake3::Hash::from_hex(&event.hash).expect("hash");
        merkle.append(hash);
    }
    assert_eq!(report.audit_merkle_root, merkle.root_hex());
    let verification = report.audit_verification.expect("verification");
    assert_eq!(
        verification.status,
        rustorch::audit::AuditVerificationStatus::Valid
    );
    assert!(!verification.proofs.is_empty());
}

#[test]
fn governance_parallel_validation_race_safe() {
    let root = temp_root();
    let store = Arc::new(ExperimentStore::new(&root).expect("store"));
    for idx in 0..16 {
        create_run(&store, &format!("race-{idx}"));
    }

    let mut handles = Vec::new();
    for idx in 0..3 {
        let store = Arc::clone(&store);
        let audit_path = store.root().join("audit").join(format!("race-{idx}.jsonl"));
        handles.push(thread::spawn(move || {
            let mut config = RunGovernanceConfig::default();
            config.max_workers = 3;
            config.audit_log = true;
            config.audit_log_path = Some(audit_path);
            let report = store.validate_runs(&config).expect("report");
            assert_eq!(report.summary.invalid_runs, 0);
            assert_eq!(report.remediation.total_tickets, 0);
        }));
    }

    for handle in handles {
        handle.join().expect("join");
    }
}
