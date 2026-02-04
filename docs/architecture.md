# Architecture Overview

## Objectives
- Provide a clear mental model for the Rust core, experiment store, and Python bindings.
- Document design patterns, concurrency boundaries, and error taxonomy usage.
- Offer guidance for extending algorithms and services without violating invariants.

## System Boundaries
### Rust Core (`src/`)
- **Tensor + Autograd**: Defines `Tensor`, `TensorInner`, and the autograd graph primitives.
- **Ops**: Stateless operator functions in `src/ops/` that validate layout, execute math, and wire autograd.
- **Storage**: `Storage` owns the dense f32 buffer; layout is defined by shape/strides.
- **Training Orchestration**: `training.rs` builds training loops that depend on the experiment store, optimizer, and model APIs.

### Experiment Persistence (`src/experiment.rs`)
- **Run metadata** persisted as JSON (`run.json`), metrics as JSONL (`metrics.jsonl`), telemetry as JSONL, and summaries as JSON (`run_summary.json`).
- **Artifacts** captured in `artifacts.json` with an explicit `ArtifactKind` enum.
- **Checkpoint format** (`RTCH`) is versioned and validated on load.

### Python Bindings (`python/`)
- Optional bindings with `python-bindings` feature to integrate with PyTorch’s `torch.compile` backend registration.
- The Rust core remains the single source of truth for layout invariants and error taxonomy.

## Design Patterns
### Layered Architecture
- **Domain layer**: tensor, autograd, ops, models, optimizers.
- **Application layer**: training orchestration, experiment store, CLI.
- **Infrastructure layer**: checkpoint serialization, JSONL logging, telemetry sinks.

This separation makes it explicit where invariants are enforced (domain) versus where side effects are managed (application + infrastructure).

### Strategy + Builder
- **Optimizer strategy**: `Optimizer` trait with `Sgd` implementation. Extension points are explicit via trait methods.
- **Dataset builder**: `TensorDataset::batch_iter` and synthetic data factories provide configurable builders without coupling to training loops.

### RAII + Scoped Timers
- Telemetry timers are constructed with RAII and emit metrics when dropped, guaranteeing event emission even when early returns occur.

## Concurrency Model
- **Autograd scheduler** supports configurable parallel backward traversal; workers are scoped per batch.
- **Metrics logging** uses a bounded synchronous channel (backpressure) and a single worker thread for IO.
- **Telemetry recording** uses a dedicated thread that serializes events into JSONL files.

Thread-safe boundaries are enforced through `Arc`, `Mutex`, and `SyncSender` with explicit error propagation.

## Error Taxonomy & Boundaries
- **`TorchError`** defines structured errors for layout validation, checkpoint compatibility, training flow, and experiment persistence.
- **Fail-fast validation**: every op validates layout before computation, returning explicit errors rather than panicking.
- **I/O isolation**: experiment store and telemetry sinks encapsulate filesystem interactions and surface consistent errors.

## Observability & Metrics
- Metrics are recorded per step and rolled up into summaries (min/max/mean/p50/p95/last).
- Telemetry events capture duration and tags, with rollups computed during run summaries.
- Layout validation counters are persisted into run summaries to provide coverage and failure context.

## Extensibility Guidelines
- New ops should:
  1. Validate layout (`validate_layout` or `validate_strided_layout`).
  2. Use explicit error messages with the operation name.
  3. Provide gradient wiring via `GradFn` if required.
- New training workflows should:
  1. Create a run, start telemetry/metrics logging, and reset layout counters.
  2. Record artifacts and checkpoint state dicts via the experiment store.
  3. Persist rollup summaries in all exit paths (success or failure).

## Testing & Benchmarks
- Integration tests live under `tests/` and align with operator correctness.
- Benchmarks focus on autograd scheduling, layout validation, metrics logging, and run summary rollups.

