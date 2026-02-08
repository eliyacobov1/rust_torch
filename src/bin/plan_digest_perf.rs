use std::env;
use std::time::Instant;

use rustorch::audit::{execution_plan_digest, governance_plan_digest};
use rustorch::execution::{ExecutionCostProfile, ExecutionGraph, ExecutionPlanner};
use rustorch::governance::{GovernanceGraph, GovernancePlanner};
use rustorch::Result;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T> {
    let value = args
        .iter()
        .position(|arg| arg == flag)
        .and_then(|idx| args.get(idx + 1))
        .map(|value| value.parse::<T>())
        .transpose()
        .map_err(|_| rustorch::TorchError::InvalidArgument {
            op: "plan_digest_perf.parse_arg",
            msg: format!("invalid value for {flag}"),
        })?;
    Ok(value.unwrap_or(default))
}

fn build_governance_plan(stages: usize, seed: u64) -> Result<rustorch::GovernancePlan> {
    let mut graph = GovernanceGraph::new();
    let mut previous: Option<String> = None;
    for idx in 0..stages {
        let run_id = format!("run-{idx}");
        let stage_id = format!("{run_id}:validate");
        let deps = previous.iter().cloned().collect::<Vec<_>>();
        graph.add_stage_with_id(stage_id.clone(), run_id, "validate", deps)?;
        previous = Some(stage_id);
    }
    GovernancePlanner::new(seed, graph).plan()
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let stages: usize = parse_arg(&args, "--stages", 10_000)?;
    let lanes: usize = parse_arg(&args, "--lanes", 8)?;
    let seed: u64 = parse_arg(&args, "--seed", 42)?;

    let start = Instant::now();
    let governance_plan = build_governance_plan(stages, seed)?;
    let graph =
        ExecutionGraph::from_governance_plan(&governance_plan, ExecutionCostProfile::default())?;
    let execution_plan = ExecutionPlanner::new(seed, graph)
        .with_max_lanes(lanes)
        .plan()?;
    let governance_digest = governance_plan_digest(&governance_plan)?;
    let execution_digest = execution_plan_digest(&execution_plan)?;
    let duration = start.elapsed().as_secs_f64();

    println!("Plan Digest Benchmark");
    println!("stages: {stages}");
    println!("lanes: {lanes}");
    println!("governance_hash: {}", governance_digest.plan_hash);
    println!("execution_hash: {}", execution_digest.plan_hash);
    println!("duration_s: {duration:.6}");
    Ok(())
}
