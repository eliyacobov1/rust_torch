# Principal-Grade Repository Guide

## Mission
Transform this repository into a **governance-first, deterministic ML systems platform** that can withstand principal-level interviews: end-to-end integrity proofs, replayable orchestration, and production-grade telemetry with regression defense.

## Architectural Pillars (Hardest-Path Emphasis)
1. **Deterministic Governance Orchestration** — replayable validation, deterministic scheduling, audit ledger integrity proofs.
2. **Experiment Intelligence Graph Engine** — delta indexing, provenance-grade comparisons, regression gating.
3. **Telemetry + Performance Defense** — export-ready telemetry with hard budgets, wall-time + memory accounting.
4. **Training Orchestration Suite** — multi-stage workflows, signed artifacts, deterministic replay.

## Project Structure
- `src/` Rust core (tensor, autograd, ops, storage, CLI in `main.rs`).
- `src/experiment.rs` experiment persistence, telemetry summaries, run comparisons, governance.
- `src/audit.rs` audit ledger, verification, Merkle proofs.
- `src/governance.rs` deterministic scheduling + governance replay utilities.
- `src/checkpoint.rs` checkpoint serialization and `state_dict` save/load.
- `tests/` integration + Python tests.
- `python/` Python bindings helpers.
- `cpp_ext/` optional C++ extensions.
- `examples/`, `benches/`, `docs/` for usage, benchmarks, documentation.

## Build, Test, and Dev Commands
- `cargo build --release` — build the Rust library and CLI.
- `cargo build --release --no-default-features --bin rustorch_cli` — Rust-only CLI.
- `cargo run --release --no-default-features --bin rustorch_cli` — run CLI locally.
- `cargo run --release --bin bench_governance_schedule -- 10000 42` — governance scheduler benchmark (wall time + memory delta).
- `maturin develop --release` — build/install Python bindings.
- `cargo test` — Rust unit + integration tests.
- `pytest tests/test_tensor.py` — Python binding test.
- `python scripts/bench_run_compare.py` — analytics benchmark.
- `python scripts/bench_run_validate.py` — governance benchmark.
- `python scripts/bench_audit_verify.py` — audit verification benchmark.

## Engineering Bar (Non-Negotiable)
- Favor **explicit error handling** (`anyhow`/`thiserror` patterns in use).
- Add **structured logging** for all major workflows.
- Prefer **deterministic concurrency** and avoid hidden side effects.
- Use **Rust 2021**, `snake_case` for functions/modules, `CamelCase` for types/traits.

## Testing Requirements
- Governance changes must include **quarantine coverage** and **race-condition checks**.
- Analytics changes must include **stress and concurrency** coverage.
- Benchmarks must output **wall time + memory delta** for performance claims.

## Commit & PR Expectations
- Commit messages are short, imperative, and specific (e.g., "Add audit verifier").
- PRs must include: concise summary, key changes, and test commands.
- Link related issues if they exist; note API changes explicitly.

## Principal Milestones (Active)
1. Deterministic governance replay pipeline + scheduling ledger.
2. Graph-based experiment comparison with regression gating.
3. Performance telemetry export with regression defense.
4. Deterministic replay for validation + analytics.
5. End-to-end training orchestration with signed artifacts.
