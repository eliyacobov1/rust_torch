use std::collections::BTreeMap;

use rustorch::execution::{ExecutionCostProfile, ExecutionGraph, ExecutionPlanner};

#[test]
fn execution_plan_is_deterministic_and_respects_dependencies() {
    let mut graph = ExecutionGraph::new();
    let stage_a = graph
        .add_stage("run-1", "a", Vec::new(), 3)
        .expect("add stage a");
    let stage_b = graph
        .add_stage("run-1", "b", vec![stage_a.clone()], 2)
        .expect("add stage b");
    let stage_c = graph
        .add_stage("run-1", "c", vec![stage_a.clone()], 1)
        .expect("add stage c");
    graph
        .add_stage("run-1", "d", vec![stage_b.clone(), stage_c.clone()], 1)
        .expect("add stage d");

    let plan_a = ExecutionPlanner::new(42, graph.clone())
        .with_max_lanes(2)
        .plan()
        .expect("plan a");
    let plan_b = ExecutionPlanner::new(42, graph)
        .with_max_lanes(2)
        .plan()
        .expect("plan b");

    assert_eq!(plan_a.total_lanes, 2);
    assert_eq!(plan_a.entries.len(), 4);
    assert_eq!(plan_a.entries, plan_b.entries);

    let mut completion = BTreeMap::new();
    for entry in &plan_a.entries {
        for dep in &entry.dependencies {
            let dep_end = completion.get(dep).copied().unwrap_or(0);
            assert!(
                entry.start_tick >= dep_end,
                "stage {} starts before dependency {} completes",
                entry.stage_id,
                dep
            );
        }
        completion.insert(entry.stage_id.clone(), entry.end_tick);
    }
}

#[test]
fn execution_cost_profile_applies_overrides() {
    let mut overrides = BTreeMap::new();
    overrides.insert("run-x:hot".to_string(), 7);
    let profile = ExecutionCostProfile {
        default_cost: 2,
        overrides,
    };

    let mut governance = rustorch::governance::GovernanceGraph::new();
    governance
        .add_stage_with_id("run-x:hot".to_string(), "run-x", "hot", Vec::new())
        .expect("add governance stage");
    let graph = ExecutionGraph::from_governance(&governance, profile.clone()).expect("convert");

    let plan = ExecutionPlanner::new(9, graph)
        .with_max_lanes(1)
        .plan()
        .expect("plan");

    assert_eq!(plan.entries[0].cost, 7);
    assert_eq!(profile.cost_for("run-x:hot"), 7);
}
