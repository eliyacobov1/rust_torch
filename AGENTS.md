# Principal-Grade Repository Guide (v2)

## Mission
Ship a **governance-first, deterministic ML systems platform** with verifiable integrity proofs, replayable orchestration, and telemetry-backed regression defense that stands up in principal and senior systems interviews.

## Core Architecture Pillars
1. **Deterministic Governance Orchestration**
   - Replayable schedules, ordered ledgers, quarantine automation.
2. **Experiment Intelligence Graph Engine**
   - Delta indexing, provenance-aware comparisons, regression gates.
3. **Telemetry + Performance Defense**
   - Wall-time + memory budgets with export-ready telemetry.
4. **Training Orchestration Suite**
   - Multi-stage workflows, signed artifacts, deterministic replay.

## Non-Negotiable Engineering Bar
- Deterministic concurrency: ordering and scheduling must be explicit and replayable.
- Structured logging for all major workflows.
- Explicit error handling using `anyhow`/`thiserror` patterns.
- Rust 2021, `snake_case` for functions/modules, `CamelCase` for types.
- Performance evidence: benchmarks must report wall time + memory delta.

## Repository Map (Primary Touchpoints)
- `src/experiment.rs` — experiment persistence, governance, graph analytics.
- `src/governance.rs` — deterministic scheduling, replay utilities.
- `src/audit.rs` — audit ledger + Merkle verification.
- `src/telemetry.rs` — telemetry sinks and timers.
- `src/training.rs` — training orchestration.
- `tests/` — integration, stress, and race-condition tests.

## Release Gates
1. Deterministic replay across governance + analytics.
2. Regression gate enforcement with auditable findings.
3. Telemetry export with regression thresholds.
4. Benchmarks emitting wall time + memory delta for analytics and audit verification.

## Commit & PR Expectations
- Commit messages: short, imperative, specific (e.g., "Add regression gate engine").
- PRs must include: summary, key changes, and test commands.
- Note API changes explicitly and link related issues if applicable.
