# Principal-Grade Repository Guide (v3)

## Mission
Deliver a **governance-first, deterministic ML systems platform** with verifiable integrity proofs, replayable orchestration, telemetry-backed regression defense, and senior-principal interview readiness.

## Core Architecture Pillars
1. **Deterministic Governance Execution Engine**
   - Replayable, dependency-aware orchestration (DAG + explicit wave/lane ordering).
   - Ledger-backed execution events for traceability.
2. **Experiment Intelligence Graph Engine**
   - Delta indexing, provenance-aware comparisons, regression gates.
   - Auditable comparison graphs with deterministic ordering.
3. **Telemetry + Performance Defense**
   - Budget enforcement (mean/p95/max/volume).
   - Benchmarks emitting wall time + memory delta.
4. **Training Orchestration Suite**
   - Multi-stage workflows with explicit governance plans.
   - Signed/verified artifacts and deterministic replay manifests.

## Non-Negotiable Engineering Bar
- Deterministic concurrency: ordering and scheduling must be explicit and replayable.
- Structured logging for major workflows.
- Explicit error handling using `anyhow`/`thiserror` patterns.
- Rust 2021, `snake_case` for functions/modules, `CamelCase` for types.
- Performance evidence: benchmarks and scripts must report wall time + memory delta.

## Repository Map (Primary Touchpoints)
- `src/experiment.rs` — experiment persistence, governance, graph analytics.
- `src/governance.rs` — deterministic scheduling, replay utilities, governance ledger.
- `src/audit.rs` — audit ledger + Merkle verification.
- `src/telemetry.rs` — telemetry sinks, budgets, timers.
- `src/training.rs` — training orchestration + governance stages.
- `tests/` — integration, stress, and race-condition tests.

## Release Gates
1. Deterministic replay across governance + analytics.
2. Regression gate enforcement with auditable findings.
3. Telemetry export with enforced budgets.
4. Benchmarks emitting wall time + memory delta for analytics and audit verification.

## Commit & PR Expectations
- Commit messages: short, imperative, specific (e.g., "Add governance replay ledger").
- PRs must include: summary, key changes, and test commands.
- Note API changes explicitly and link related issues if applicable.
