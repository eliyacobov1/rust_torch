use std::collections::BTreeMap;

use rand::{distributions::Alphanumeric, Rng};
use rustorch::experiment::{ExperimentStore, MetricsLoggerConfig};

fn temp_root() -> std::path::PathBuf {
    let suffix: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    std::env::temp_dir().join(format!("rustorch_metrics_{suffix}"))
}

#[test]
fn metrics_logger_writes_jsonl() {
    let root = temp_root();
    let store = ExperimentStore::new(&root).expect("store");
    let run = store
        .create_run("metrics", serde_json::json!({}), Vec::new())
        .expect("run");
    let logger = run
        .start_metrics_logger(
            MetricsLoggerConfig {
                batch_size: 2,
                flush_interval_ms: 50,
                max_queue: 16,
            },
            None,
        )
        .expect("logger");

    for step in 0..3usize {
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), step as f32);
        logger.log_metrics(step, metrics).expect("log");
    }
    logger.flush().expect("flush");

    let metrics_path = root.join(run.metadata().id.clone()).join("metrics.jsonl");
    let contents = std::fs::read_to_string(&metrics_path).expect("metrics file");
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 3);

    std::fs::remove_dir_all(root).expect("cleanup");
}
