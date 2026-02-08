use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use rustorch::execution::{
    verify_execution_ledger, ExecutionAction, ExecutionLedger, ExecutionLedgerEvent,
    ExecutionLedgerVerificationConfig, ExecutionLedgerVerificationStatus,
};

fn main() -> Result<()> {
    let stages: usize = std::env::var("EXECUTION_LEDGER_PERF_STAGES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2000);
    let lanes: usize = std::env::var("EXECUTION_LEDGER_PERF_LANES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let ledger_path = std::env::temp_dir().join(format!(
        "execution_ledger_perf_{}_{}.jsonl",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));

    let mut ledger = ExecutionLedger::open(&ledger_path)
        .map_err(|err| anyhow!("failed to open execution ledger: {err}"))?;
    for idx in 0..stages {
        let stage_id = format!("perf-run:stage-{idx}");
        let lane = idx % lanes.max(1);
        let start_tick = idx as u64;
        let end_tick = start_tick.saturating_add(1);
        ledger
            .record(ExecutionLedgerEvent::new(
                stage_id.clone(),
                "perf-run",
                "stage",
                ExecutionAction::Started,
                "start",
                lane,
                start_tick,
                end_tick,
            ))
            .map_err(|err| anyhow!("failed to record start event: {err}"))?;
        ledger
            .record(ExecutionLedgerEvent::new(
                stage_id,
                "perf-run",
                "stage",
                ExecutionAction::Completed,
                "done",
                lane,
                start_tick,
                end_tick,
            ))
            .map_err(|err| anyhow!("failed to record completion event: {err}"))?;
    }

    let start_mem = read_rss_bytes().unwrap_or(0);
    let start = Instant::now();
    let report =
        verify_execution_ledger(&ledger_path, &ExecutionLedgerVerificationConfig::default())
            .map_err(|err| anyhow!("failed to verify execution ledger: {err}"))?;
    let elapsed = start.elapsed();
    let end_mem = read_rss_bytes().unwrap_or(start_mem);
    let delta = end_mem.saturating_sub(start_mem);

    if report.status != ExecutionLedgerVerificationStatus::Valid {
        return Err(anyhow!(
            "execution ledger verification failed with {} issues",
            report.issues.len()
        ));
    }

    println!(
        "wall_time_ms={:.3} memory_delta_bytes={} total_events={} total_stages={}",
        elapsed.as_secs_f64() * 1000.0,
        delta,
        report.total_events,
        stages
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
