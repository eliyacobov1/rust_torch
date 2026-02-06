use std::sync::{Arc, Barrier};
use std::thread;

use rustorch::governance::{GovernanceGraph, GovernancePlanner};

#[test]
fn governance_planner_is_deterministic_under_parallel_calls() {
    let mut graph = GovernanceGraph::new();
    let root = graph
        .add_stage("run-parallel", "root", Vec::new())
        .expect("add stage");
    let mut prev = root;
    for idx in 0..64 {
        let stage = format!("node-{idx}");
        let next = graph
            .add_stage("run-parallel", stage, vec![prev.clone()])
            .expect("add stage");
        prev = next;
    }

    let graph = Arc::new(graph);
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let graph = Arc::clone(&graph);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            let plan = GovernancePlanner::new(77, (*graph).clone())
                .plan()
                .expect("plan");
            plan.entries
                .iter()
                .map(|entry| entry.stage_id.clone())
                .collect::<Vec<_>>()
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().expect("join"));
    }

    for window in results.windows(2) {
        assert_eq!(window[0], window[1]);
    }
}
