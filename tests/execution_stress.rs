use std::collections::BTreeMap;
use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
use std::thread;

use blake3::Hasher;
use rustorch::execution::{
    ExecutionEngine, ExecutionGraph, ExecutionPlanner, ExecutionRegistry, ExecutionStatus,
    ExecutionTask,
};

#[derive(Debug)]
struct CounterTask {
    counter: Arc<AtomicUsize>,
}

impl ExecutionTask for CounterTask {
    fn execute(&self) -> rustorch::Result<ExecutionStatus> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(ExecutionStatus::Success)
    }
}

fn build_graph(nodes: usize) -> ExecutionGraph {
    let mut graph = ExecutionGraph::new();
    let mut previous: Vec<String> = Vec::new();
    for idx in 0..nodes {
        let stage = format!("stage-{idx}");
        let deps = if previous.is_empty() {
            Vec::new()
        } else {
            previous.iter().cloned().collect()
        };
        let stage_id = graph
            .add_stage("stress-run", stage, deps, 1)
            .expect("add stage");
        previous = vec![stage_id];
    }
    graph
}

fn plan_digest(plan: &rustorch::execution::ExecutionPlan) -> String {
    let mut hasher = Hasher::new();
    let payload = serde_json::to_vec(plan).expect("serialize plan");
    hasher.update(&payload);
    hasher.finalize().to_hex().to_string()
}

#[test]
fn execution_plan_is_stable_under_concurrency() {
    let graph = build_graph(64);
    let mut handles = Vec::new();
    for _ in 0..4 {
        let graph = graph.clone();
        handles.push(thread::spawn(move || {
            let plan = ExecutionPlanner::new(2025, graph)
                .with_max_lanes(4)
                .plan()
                .expect("plan");
            plan_digest(&plan)
        }));
    }

    let mut digests = BTreeMap::new();
    for handle in handles {
        let digest = handle.join().expect("join");
        *digests.entry(digest).or_insert(0usize) += 1;
    }

    assert_eq!(digests.len(), 1, "plan digests should match");
}

#[test]
fn execution_engine_handles_stress_load() {
    let graph = build_graph(32);
    let plan = ExecutionPlanner::new(777, graph)
        .with_max_lanes(4)
        .plan()
        .expect("plan");

    let counter = Arc::new(AtomicUsize::new(0));
    let mut registry = ExecutionRegistry::new();
    for entry in &plan.entries {
        registry.register(
            entry.stage_id.clone(),
            Arc::new(CounterTask {
                counter: Arc::clone(&counter),
            }),
        );
    }

    let mut engine = ExecutionEngine::new(None);
    let report = engine
        .run::<rustorch::telemetry::JsonlSink>(&plan, &registry, None)
        .expect("run");

    assert_eq!(report.total_stages, plan.total_stages);
    assert_eq!(counter.load(Ordering::SeqCst), plan.total_stages);
    assert_eq!(report.failed, 0);
}
