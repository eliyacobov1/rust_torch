use rustorch::governance::{GovernanceGraph, GovernancePlanner, GovernanceReplayCursor};

#[test]
fn deterministic_plan_is_stable() {
    let mut graph = GovernanceGraph::new();
    let a = graph
        .add_stage("run-a", "ingest", Vec::new())
        .expect("add stage");
    let b = graph
        .add_stage("run-a", "train", vec![a.clone()])
        .expect("add stage");
    let _c = graph
        .add_stage("run-a", "validate", vec![b.clone()])
        .expect("add stage");

    let planner = GovernancePlanner::new(42, graph.clone());
    let plan_a = planner.plan().expect("plan");
    let plan_b = GovernancePlanner::new(42, graph).plan().expect("plan");

    assert_eq!(plan_a.entries.len(), plan_b.entries.len());
    let ids_a: Vec<String> = plan_a.entries.iter().map(|entry| entry.stage_id.clone()).collect();
    let ids_b: Vec<String> = plan_b.entries.iter().map(|entry| entry.stage_id.clone()).collect();
    assert_eq!(ids_a, ids_b);

    let mut cursor = GovernanceReplayCursor::new(plan_a);
    assert!(cursor.expect_stage("run-a:ingest").is_ok());
    assert!(cursor.expect_stage("run-a:train").is_ok());
    assert!(cursor.expect_stage("run-a:validate").is_ok());
    assert_eq!(cursor.remaining(), 0);
}

#[test]
fn governance_plan_handles_large_graph() {
    let mut graph = GovernanceGraph::new();
    let mut previous = None;
    for idx in 0..250 {
        let stage = format!("stage-{idx}");
        let deps = previous.iter().cloned().collect::<Vec<_>>();
        let stage_id = graph
            .add_stage("run-stress", stage, deps)
            .expect("add stage");
        previous = Some(stage_id);
    }

    let plan = GovernancePlanner::new(2024, graph).plan().expect("plan");
    assert_eq!(plan.total_stages, 250);
    assert_eq!(plan.entries.len(), 250);
    assert!(plan.total_waves >= 1);
}
