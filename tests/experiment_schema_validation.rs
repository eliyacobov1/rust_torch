use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::ExperimentStore;

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_schema_test_{suffix}"))
}

#[test]
fn read_summary_rejects_invalid_schema() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let mut run = store
        .create_run("schema_summary", serde_json::json!({}), Vec::new())
        .expect("run");
    run.mark_completed().expect("mark completed");
    run.write_summary(None).expect("summary");
    let run_id = run.metadata().id.clone();
    let summary_path = store.root().join(&run_id).join("run_summary.json");
    let mut summary_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).expect("read summary"))
            .expect("parse summary");
    summary_value["completed_at_unix"] = serde_json::json!(0);
    summary_value["duration_secs"] = serde_json::json!(-1.0);
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary_value).expect("serialize"),
    )
    .expect("write summary");

    assert!(store.read_summary(&run_id).is_err());
}

#[test]
fn open_run_rejects_invalid_metadata() {
    let store = ExperimentStore::new(temp_root()).expect("store");
    let run = store
        .create_run("schema_metadata", serde_json::json!({}), Vec::new())
        .expect("run");
    let run_id = run.metadata().id.clone();
    let metadata_path = store.root().join(&run_id).join("run.json");
    let mut metadata_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).expect("read metadata"))
            .expect("parse metadata");
    metadata_value["name"] = serde_json::json!("");
    std::fs::write(
        &metadata_path,
        serde_json::to_string_pretty(&metadata_value).expect("serialize"),
    )
    .expect("write metadata");

    assert!(store.open_run(&run_id).is_err());
}
