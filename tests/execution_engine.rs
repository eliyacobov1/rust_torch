use std::sync::Arc;

use rustorch::execution::{
    ExecutionEngine, ExecutionGraph, ExecutionPlanner, ExecutionRegistry, ExecutionStatus,
    ExecutionTask,
};

#[derive(Debug)]
struct TestTask {
    fail: bool,
}

impl ExecutionTask for TestTask {
    fn execute(&self) -> rustorch::Result<ExecutionStatus> {
        if self.fail {
            Ok(ExecutionStatus::Failed)
        } else {
            Ok(ExecutionStatus::Success)
        }
    }
}

#[test]
fn execution_engine_runs_all_stages() {
    let mut graph = ExecutionGraph::new();
    let stage_a = graph
        .add_stage("run-1", "a", Vec::new(), 1)
        .expect("stage a");
    graph
        .add_stage("run-1", "b", vec![stage_a.clone()], 1)
        .expect("stage b");

    let plan = ExecutionPlanner::new(7, graph)
        .with_max_lanes(2)
        .plan()
        .expect("plan");

    let mut registry = ExecutionRegistry::new();
    registry.register(stage_a.clone(), Arc::new(TestTask { fail: false }));
    registry.register("run-1:b", Arc::new(TestTask { fail: false }));

    let mut engine = ExecutionEngine::new(None);
    let report = engine
        .run::<rustorch::telemetry::JsonlSink>(&plan, &registry, None)
        .expect("run");

    assert_eq!(report.total_stages, 2);
    assert_eq!(report.failed, 0);
}

#[test]
fn execution_engine_reports_failures() {
    let mut graph = ExecutionGraph::new();
    let stage_a = graph
        .add_stage("run-2", "a", Vec::new(), 1)
        .expect("stage a");
    graph
        .add_stage("run-2", "b", vec![stage_a.clone()], 1)
        .expect("stage b");

    let plan = ExecutionPlanner::new(11, graph)
        .with_max_lanes(1)
        .plan()
        .expect("plan");

    let mut registry = ExecutionRegistry::new();
    registry.register(stage_a.clone(), Arc::new(TestTask { fail: false }));
    registry.register("run-2:b", Arc::new(TestTask { fail: true }));

    let mut engine = ExecutionEngine::new(None);
    let report = engine
        .run::<rustorch::telemetry::JsonlSink>(&plan, &registry, None)
        .expect("run");

    assert_eq!(report.total_stages, 2);
    assert_eq!(report.failed, 1);
}
