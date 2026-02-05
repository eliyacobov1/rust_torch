# Principal-Grade Repository Guide

## Mission
Accelerate this repository to a **senior/principal interview-ready system** by delivering governance-grade integrity, deterministic analytics, and production orchestration. Every change should push a subsystem forward, not just tweak a component.

## Architectural Pillars (Hardest-Path Emphasis)
1. **Governance Audit Engine** — Verification pipeline, remediation workflows, audit ledger integrity proofs.
2. **Experiment Intelligence & Analytics** — Graph-based comparisons, delta indexing, provenance-grade reporting.
3. **Service/API Orchestration** — Multi-stage training workflows with reproducible configs and telemetry export.
4. **Algorithmic Training Suite** — Model, optimizer, dataset, and trainer pipelines with observability.

## Project Structure
- `src/` Rust core (tensor, autograd, ops, storage, CLI in `main.rs`).
- `src/experiment.rs` experiment persistence, telemetry summaries, run comparisons, governance.
- `src/audit.rs` audit ledger, verification, Merkle proofs.
- `src/checkpoint.rs` checkpoint serialization and `state_dict` save/load.
- `tests/` integration + Python tests.
- `python/` Python bindings helpers.
- `cpp_ext/` optional C++ extensions.
- `examples/`, `benches/`, `docs/` for usage, benchmarks, documentation.

## Build, Test, and Dev Commands
- `cargo build --release` — build the Rust library and CLI.
- `cargo build --release --no-default-features --bin rustorch_cli` — Rust-only CLI.
- `cargo run --release --no-default-features --bin rustorch_cli` — run CLI locally.
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
1. Governance audit verification CLI + remediation tickets + proof sampling.
2. Graph-based experiment comparison with regression gating.
3. Performance telemetry export with regression defense.
4. Deterministic replay for validation + analytics.
5. End-to-end training orchestration with signed artifacts.
