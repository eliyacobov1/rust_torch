use criterion::{criterion_group, criterion_main, Criterion};
use rustorch::governance::{GovernanceGraph, GovernancePlanner};

fn bench_governance_plan(c: &mut Criterion) {
    c.bench_function("governance_plan_large", |b| {
        b.iter(|| {
            let mut graph = GovernanceGraph::new();
            let mut previous = None;
            for idx in 0..500 {
                let stage = format!("stage-{idx}");
                let deps = previous.iter().cloned().collect::<Vec<_>>();
                let stage_id = graph
                    .add_stage("bench-run", stage, deps)
                    .expect("add stage");
                previous = Some(stage_id);
            }
            let plan = GovernancePlanner::new(1337, graph).plan().expect("plan");
            criterion::black_box(plan.total_stages);
        })
    });
}

criterion_group!(benches, bench_governance_plan);
criterion_main!(benches);
