use std::collections::BTreeMap;
use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rustorch::experiment::{ExperimentStore, MetricAggregation, RunComparisonConfig, RunFilter};
use rustorch::{Result, TorchError};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let runs = args
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(250);
    let metrics = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);
    let seed = args
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(42);

    let root = temp_root(seed);
    let store = ExperimentStore::new(root)?;
    let metric_names = (0..metrics)
        .map(|idx| format!("metric_{idx}"))
        .collect::<Vec<String>>();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut run_ids = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let mut run = store.create_run(
            &format!("bench_run_{run_idx}"),
            serde_json::json!({ "seed": seed, "metrics": metrics }),
            vec!["bench".to_string()],
        )?;
        for step in 0..5 {
            let mut row = BTreeMap::new();
            for name in &metric_names {
                let value = rng.gen_range(0.0..1.0) + step as f32 * 0.01;
                row.insert(name.clone(), value);
            }
            run.log_metrics(step, row)?;
        }
        run.mark_completed()?;
        run.write_summary(None)?;
        run_ids.push(run.metadata().id.clone());
    }

    let start_mem = current_rss_kb().unwrap_or(0);
    let start = Instant::now();
    let config = RunComparisonConfig {
        run_ids: run_ids.clone(),
        filter: RunFilter::default(),
        baseline_id: Some(run_ids[0].clone()),
        metric_aggregation: MetricAggregation::Mean,
        top_k: 5,
        build_graph: true,
        deterministic_seed: Some(seed),
        regression_gate: None,
    };
    let report = store.compare_runs(&config)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let end_mem = current_rss_kb().unwrap_or(start_mem);
    let delta_kb = end_mem.saturating_sub(start_mem);

    println!(
        "{{\"runs\":{runs},\"metrics\":{metrics},\"comparisons\":{},\"duration_ms\":{elapsed_ms:.2},\"rss_kb_delta\":{delta_kb}}}",
        report.comparisons.len()
    );
    Ok(())
}

fn current_rss_kb() -> Result<u64> {
    let content = std::fs::read_to_string("/proc/self/statm").map_err(|err| TorchError::Experiment {
        op: "bench_experiment_graph.current_rss_kb",
        msg: format!("failed to read statm: {err}"),
    })?;
    let mut parts = content.split_whitespace();
    let _total = parts.next();
    let rss = parts.next().ok_or(TorchError::Experiment {
        op: "bench_experiment_graph.current_rss_kb",
        msg: "statm missing rss field".to_string(),
    })?;
    let rss_pages = rss.parse::<u64>().map_err(|err| TorchError::Experiment {
        op: "bench_experiment_graph.current_rss_kb",
        msg: format!("invalid rss value: {err}"),
    })?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Ok(rss_pages.saturating_mul(page_size) / 1024)
}

fn temp_root(seed: u64) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustorch_graph_bench_{seed}"))
}
