# Repository Guidelines (Senior-Grade Focus)

## Architectural Pillars (Hardest-Path Emphasis)
1. **Run Governance & Validation** — Schema versioning, quarantine workflows, audit trails, and compliance-grade reports.
2. **Experiment Intelligence & Analytics** — Graph-based comparisons, delta indexing, validation, and reporting.
3. **Service/API Orchestration** — High-level training workflows with reproducible config surfaces and telemetry export.
4. **Algorithmic Training Suite** — Model, optimizer, dataset, and trainer pipelines with observability.

## Project Structure & Module Organization
- `src/` holds the Rust core (tensor, autograd, ops, storage, CLI in `main.rs`).
- `src/experiment.rs` implements experiment persistence, telemetry summaries, run comparisons, and run governance.
- `src/checkpoint.rs` implements versioned checkpoint serialization and `state_dict` save/load.
- `tests/` contains Rust integration tests plus Python tests.
- `python/` hosts Python-side packaging/bindings helpers.
- `cpp_ext/` contains optional C++ extension sources.
- `examples/`, `benches/`, and `docs/` provide usage, benchmarks, and documentation.

## Build, Test, and Development Commands
- `cargo build --release`: build the Rust library and CLI.
- `cargo build --release --no-default-features --bin rustorch_cli`: build Rust-only CLI (no Python bindings).
- `cargo run --release --no-default-features --bin rustorch_cli`: run the CLI locally.
- `maturin develop --release`: build/install Python bindings into the active venv.
- `cargo test`: run Rust tests in `tests/` and unit tests.
- `pytest tests/test_tensor.py`: run the Python binding test.
- `python python/rust_backend/backend_check.py`: smoke test for the `torch.compile` backend registration.
- `python cpp_ext/build.py` then `python examples/eager_privateuse1_demo.py`: build + run the PrivateUse1 backend demo.
- `cargo bench`: run benchmarks in `benches/`.
- `python scripts/bench_run_compare.py`: run analytics benchmark (wall time + memory delta).
- `python scripts/bench_run_validate.py`: benchmark governance validation (wall time + memory delta).

## Coding Style & Naming Conventions
- Rust 2021 edition; use `snake_case` for modules/functions and `CamelCase` for types/traits.
- Default rustfmt style (4-space indentation). Run `cargo fmt` before committing if available.
- Prefer explicit error handling with `anyhow`/`thiserror` patterns already used in the codebase.

## Testing Guidelines
- Rust tests live in `tests/` (e.g., `forward_ops.rs`, `backward_ops.rs`, `integration.rs`).
- Python coverage is minimal and focused on bindings; keep tests in `tests/test_tensor.py`.
- Name tests descriptively and keep them grouped by operation/feature area.
- Stress + concurrency tests are required for analytics and experiment persistence changes.
- Governance changes must include quarantine coverage and race-condition checks.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and specific (e.g., “Implement MSE loss”).
- PRs should include: a concise summary, key changes, and how tests were run.
- Link related issues if they exist; include before/after notes for API changes.

## Senior-Grade Roadmap (Next Milestones)
1. **Governance Enforcement & Remediation**
   - Auto-repair for recoverable run metadata issues.
   - Quarantine reports with remediation guidance and audit trails.
   - Scheduled validation jobs and drift dashboards.
2. **Performance Telemetry Export**
   - Push layout validation + autograd telemetry into experiment store.
   - Export into CSV/JSON with regression thresholds and alerting hooks.
3. **Parallel Execution & Determinism**
   - Work-stealing scheduler and deterministic replay mode.
   - Trace export for run-level profiling.
4. **End-to-End Training Orchestration**
   - CLI workflows for multi-stage training and evaluation pipelines.
   - Reproducible config bundles with artifact signing.

## Intermediate Development Task (Codex Cloud)
- Add an intermediate milestone: set up a PyTorch NN that uses `rustorch` as the backend and performs MNIST classification with minimal loss.
- This task is intended to run in the Codex cloud environment so MNIST can be fetched there (no local dataset downloads).
- When implementing the MNIST training script, gate dataset downloads behind a cloud-only signal (e.g., require an environment variable like `CODEX_CLOUD=1` or a `CLOUD_MNIST_OK=1` flag). If the flag is missing, the script should exit with a clear message instead of downloading.
- Status:
  - Implemented in `examples/mnist_rustorch_demo.py` with download gating via
    `CLOUD_MNIST_OK=1`, `CODEX_CLOUD=1`, or `MNIST_ALLOW_DOWNLOAD=1`.
  - Training loop uses a CNN compiled with `torch.compile(backend="rust_backend")`
    and includes conv2d + cross-entropy loss.
