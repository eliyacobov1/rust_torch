use std::path::PathBuf;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::audit::{
    verify_audit_log, AuditEvent, AuditLog, AuditScope, AuditStatus, AuditVerificationConfig,
    AuditVerificationStatus,
};

fn temp_log_path(name: &str) -> PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(6)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_audit_{name}_{suffix}.jsonl"))
}

fn write_sample_events(path: &PathBuf) {
    let mut log = AuditLog::open(path).expect("audit log");
    let events = vec![
        AuditEvent::new(
            AuditScope::Store,
            None,
            None,
            AuditStatus::Started,
            "boot",
            0,
        ),
        AuditEvent::new(
            AuditScope::Run,
            Some("run-1".to_string()),
            Some("metadata".to_string()),
            AuditStatus::Completed,
            "metadata ok",
            0,
        ),
        AuditEvent::new(
            AuditScope::Run,
            Some("run-1".to_string()),
            Some("summary".to_string()),
            AuditStatus::Completed,
            "summary ok",
            0,
        ),
    ];
    for event in events {
        log.record(event).expect("record");
    }
}

#[test]
fn audit_verification_accepts_valid_chain() {
    let path = temp_log_path("valid");
    write_sample_events(&path);

    let mut config = AuditVerificationConfig::default();
    config.include_proofs = true;
    config.max_proofs = 2;
    let merkle_root = rustorch::audit::AuditLedger::from_path(&path)
        .expect("ledger")
        .events
        .iter()
        .map(|event| blake3::Hash::from_hex(&event.hash).expect("hash"))
        .collect::<Vec<_>>();
    let mut accumulator = rustorch::audit::MerkleAccumulator::new();
    for hash in merkle_root {
        accumulator.append(hash);
    }
    config.expected_root = accumulator.root_hex();

    let report = verify_audit_log(&path, &config).expect("verify");
    assert_eq!(report.status, AuditVerificationStatus::Valid);
    assert!(report.matches_expected_root.unwrap_or(false));
    assert!(!report.proofs.is_empty());
}

#[test]
fn audit_verification_detects_tamper() {
    let path = temp_log_path("tamper");
    write_sample_events(&path);

    let contents = std::fs::read_to_string(&path).expect("read");
    let mut lines = Vec::new();
    for (idx, line) in contents.lines().enumerate() {
        if idx == 1 {
            let mut event: AuditEvent = serde_json::from_str(line).expect("parse");
            event.message = "tampered".to_string();
            lines.push(serde_json::to_string(&event).expect("serialize"));
        } else {
            lines.push(line.to_string());
        }
    }
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).expect("write");

    let report = verify_audit_log(&path, &AuditVerificationConfig::default()).expect("verify");
    assert_eq!(report.status, AuditVerificationStatus::Invalid);
    assert!(!report.issues.is_empty());
}
