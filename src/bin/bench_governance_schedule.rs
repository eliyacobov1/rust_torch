use std::path::PathBuf;
use std::time::Instant;

use rustorch::governance::DeterministicScheduler;
use rustorch::{Result, TorchError};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let count = args.get(1).and_then(|arg| arg.parse::<usize>().ok()).unwrap_or(10_000);
    let seed = args.get(2).and_then(|arg| arg.parse::<u64>().ok()).unwrap_or(42);

    let run_dirs = (0..count)
        .map(|idx| PathBuf::from(format!("/tmp/governance-run-{idx}")))
        .collect::<Vec<PathBuf>>();

    let rss_before = read_rss_kb().unwrap_or(0);
    let start = Instant::now();
    let scheduler = DeterministicScheduler::new(seed, run_dirs)?;
    let schedule = scheduler.drain();
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rss_after = read_rss_kb().unwrap_or(rss_before);
    let rss_delta = rss_after.saturating_sub(rss_before);

    println!("governance_schedule_bench");
    println!("runs={count} seed={seed}");
    println!("schedule_entries={}", schedule.len());
    println!("wall_time_ms={duration_ms:.3}");
    println!("rss_kb_before={rss_before}");
    println!("rss_kb_after={rss_after}");
    println!("rss_kb_delta={rss_delta}");

    Ok(())
}

fn read_rss_kb() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status").map_err(|err| {
        TorchError::Experiment {
            op: "bench_governance_schedule.read_rss",
            msg: format!("failed to read /proc/self/status: {err}"),
        }
    })?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let value = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| TorchError::Experiment {
                    op: "bench_governance_schedule.read_rss",
                    msg: "missing VmRSS value".to_string(),
                })?;
            let rss_kb = value.parse::<u64>().map_err(|err| TorchError::Experiment {
                op: "bench_governance_schedule.read_rss",
                msg: format!("invalid VmRSS value {value}: {err}"),
            })?;
            return Ok(rss_kb);
        }
    }
    Err(TorchError::Experiment {
        op: "bench_governance_schedule.read_rss",
        msg: "VmRSS entry missing".to_string(),
    })
}
