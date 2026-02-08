use serde_json::json;

use rustorch::audit::AuditLedger;
use rustorch::experiment::{ExperimentStore, RunGovernanceConfig};
use rustorch::Result;

fn create_store_root(label: &str) -> std::path::PathBuf {
    let mut root = std::env::temp_dir();
    let unique = format!(
        "rustorch_plan_anchor_{}_{}_{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default()
    );
    root.push(unique);
    root
}

fn write_completed_run(store: &ExperimentStore, name: &str) -> Result<()> {
    let mut run = store.create_run(name, json!({}), Vec::new())?;
    run.mark_completed()?;
    run.write_summary(None)?;
    Ok(())
}

#[test]
fn validation_report_includes_plan_digests_and_audit_entries() -> Result<()> {
    let root = create_store_root("report");
    let store = ExperimentStore::new(&root)?;
    write_completed_run(&store, "alpha")?;
    write_completed_run(&store, "beta")?;

    let mut config = RunGovernanceConfig::default();
    config.audit_log = true;
    let report = store.validate_runs(&config)?;

    assert!(!report.governance_plan_hash.is_empty());
    assert!(!report.execution_plan_hash.is_empty());
    assert!(report.governance_plan_root.is_some());
    assert!(report.execution_plan_root.is_some());

    let audit_log_path = report
        .audit_log_path
        .expect("audit log path expected");
    let ledger = AuditLedger::from_path(&audit_log_path)?;
    let has_execution_plan = ledger.events.iter().any(|event| {
        event
            .stage
            .as_deref()
            .map(|stage| stage == "execution.plan")
            .unwrap_or(false)
    });
    assert!(has_execution_plan, "execution.plan entry not found");

    std::fs::remove_dir_all(&root).map_err(|err| rustorch::TorchError::Experiment {
        op: "test.cleanup",
        msg: format!("failed to remove temp dir {}: {err}", root.display()),
    })?;
    Ok(())
}
