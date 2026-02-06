use std::fs;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustorch::execution::{ExecutionGraph, ExecutionPlanner};

fn main() -> Result<()> {
    let nodes: usize = std::env::var("EXECUTION_PERF_NODES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("EXECUTION_PERF_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2025);
    let lanes: usize = std::env::var("EXECUTION_PERF_LANES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let start_mem = read_rss_bytes().unwrap_or(0);
    let start = Instant::now();

    let mut graph = ExecutionGraph::new();
    let mut previous = None;
    for idx in 0..nodes {
        let stage = format!("stage-{idx}");
        let deps = previous.iter().cloned().collect::<Vec<_>>();
        let stage_id = graph
            .add_stage("perf-run", stage, deps, 1)
            .map_err(|err| anyhow!("failed to add stage {idx}: {err}"))?;
        previous = Some(stage_id);
    }

    let plan = ExecutionPlanner::new(seed, graph)
        .with_max_lanes(lanes)
        .plan()
        .map_err(|err| anyhow!("failed to plan execution graph: {err}"))?;

    let elapsed = start.elapsed();
    let end_mem = read_rss_bytes().unwrap_or(start_mem);
    let delta = end_mem.saturating_sub(start_mem);

    println!(
        "wall_time_ms={:.3} memory_delta_bytes={} total_stages={} total_lanes={} makespan_ticks={}",
        elapsed.as_secs_f64() * 1000.0,
        delta,
        plan.total_stages,
        plan.total_lanes,
        plan.makespan_ticks
    );

    Ok(())
}

fn read_rss_bytes() -> Result<u64> {
    let statm = fs::read_to_string("/proc/self/statm")?;
    let mut parts = statm.split_whitespace();
    let _size: u64 = parts
        .next()
        .ok_or_else(|| anyhow!("statm missing size"))?
        .parse()?;
    let resident: u64 = parts
        .next()
        .ok_or_else(|| anyhow!("statm missing resident"))?
        .parse()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Ok(resident.saturating_mul(page_size))
}
