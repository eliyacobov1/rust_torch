# Torch Integration Plan (M0–M4)

This plan tracks staged integration milestones for the Rust backend and the
associated test/demo coverage.

## M0: Backend registration
- Register the `rust_backend` with `torch.compile`.
- Smoke test backend registration via `python python/rust_backend/backend_check.py`.

## M1: FX graph execution (basic)
- Accept FX graphs from `torch.compile` and execute a minimal subset of ops.
- Expand `examples/compile_backend_demo.py` to cover simple matmul + activation.

## M2: Training loop compatibility
- Support common ops needed for a small training loop (linear, conv, relu, loss).
- Add an end-to-end MNIST demo that trains via `torch.compile` and evaluates accuracy.
- Demo should gate dataset downloads behind a cloud-only flag and store data
  persistently in the environment.
- Status:
  - Implemented in `examples/mnist_rustorch_demo.py` with download gating via
    `CLOUD_MNIST_OK=1`, `CODEX_CLOUD=1`, or `MNIST_ALLOW_DOWNLOAD=1`.
  - Training loop uses a CNN (conv2d + linear + relu + cross-entropy loss) compiled
    with `torch.compile(backend="rust_backend")`.

## M3: Autograd coverage
- Ensure backward ops used by the MNIST demo are routed through the backend.
- Add/extend Rust tests in `tests/` to cover new backward ops.

## M4: Performance & parity
- Compare results with eager PyTorch for correctness on representative models.
- Document performance notes and remaining operator gaps.

## Current status recap (recent work)
- MNIST demo uses a CNN compiled through `torch.compile(backend="rust_backend")`
  with gated dataset downloads (M2).
- FX runner support and backward op coverage have been expanded to cover core
  ops needed by the demo (M1/M3).
- Tensor layout semantics are now explicit: tensors are validated against
  shape/stride/storage invariants on construction, and core ops require
  contiguous layouts with structured `LayoutError`/`TorchError` reporting when
  unsupported strided inputs are encountered.
- FX lowering enforces contiguous + valid storage inputs and logs layout
  diagnostics before falling back to eager execution.
- Parity harness tests now compare eager vs compiled forward/grad results for
  representative MLP and CNN models (M4).
- `log_softmax` + `nll_loss` ops are available to support classification losses
  outside of the MNIST demo path.
- Forward/backward kernels for `batch_norm`, `dropout`, and `max_pool2d` are now
  available, expanding CNN training coverage beyond the MNIST baseline.

## Next recommended task (post-M4)
**Harden shape/stride semantics + error taxonomy for production use.**

**Why this next?**
- Operator coverage is broader now, but real-world models will hit edge cases
  in layouts, broadcasting, and dynamic shapes.
- Formalizing tensor layout semantics and error reporting reduces debugging
  time and prevents silent correctness issues.

**Scope (significant, production-oriented)**
- Introduce explicit shape/stride validation in core tensor ops and FX lowering.
- Add contiguous/strided tensor semantics with clear fallbacks or errors for
  unsupported layouts.
- Define a user-facing error taxonomy (unsupported op, layout mismatch, dtype
  incompatibility) and propagate it through `rustorch.run_fx`.
- Expand Rust and Python parity tests to cover strided tensors, broadcasting,
  and dynamic shapes in compiled graphs.

### Decision: focus task for the next development cycle
**Implement robust shape/stride semantics with a clear error taxonomy.**

**Acceptance criteria**
- Tensor ops validate shapes/strides and return structured errors for unsupported
  layouts, dtypes, or dynamic shapes.
- FX lowering emits actionable errors and documents supported layout/dtype
  combinations for each op.
- Parity harness and Rust tests include strided tensor, broadcasting, and
  dynamic-shape scenarios (forward + backward).
- Documentation updated to describe layout assumptions and failure modes.

**Why this is the next task**
- It de-risks correctness as operator coverage expands, ensuring the backend
  fails fast with useful diagnostics instead of silent miscomputations.
- It lays groundwork for serialization, optimizer state, and future performance
  work by clarifying tensor semantics.

## Production readiness roadmap (proposed)
1. **Operator + autograd coverage for common training stacks**
   - Implement backward ops needed for CNNs and MLPs beyond MNIST (conv2d, maxpool2d,
     batchnorm, dropout, and any missing cross-entropy loss paths), plus corresponding
     FX lowering.
   - Expand Rust and Python tests to cover forward/backward parity.
2. **Shape/stride semantics and error handling**
   - Introduce robust shape inference, contiguous/strided tensor semantics, and
     explicit error reporting for unsupported layouts or dynamic shapes.
3. **Serialization & checkpointing**
   - Add save/load utilities for tensors and model state (Rust + Python API),
     with versioned metadata for long-term compatibility.
4. **Optimizer and training utilities**
   - Provide optimizer state handling (SGD/Adam) and parameter grouping APIs to
     support real training loops with `torch.compile`.
5. **Performance profiling + benchmarking**
   - Expand `benches/` to track kernel throughput and end-to-end model latency,
     plus profiling hooks for operator-level timing.
6. **Release hardening**
   - CI coverage for `cargo test`, Python binding tests, and compile backend demos;
     document supported platforms and wheel build steps.

## Additional production-grade milestones (proposed)
1. **API surface stabilization + error taxonomy**
   - Formalize error types for unsupported ops, layout mismatches, and dtype
     incompatibilities; document them in Rust + Python.
   - Add user-facing diagnostics in `rustorch.run_fx` for actionable errors.
2. **Distributed-friendly state and serialization**
   - `state_dict`-style save/load with versioned metadata and dtype/layout checks.
   - Interop with PyTorch checkpoint formats where feasible.
3. **Optimizer + scheduler parity**
   - Implement SGD + Adam with weight decay, gradient clipping, and LR scheduling.
   - Provide parameter group APIs to align with PyTorch ergonomics.
4. **Performance telemetry**
   - Add operator-level timing hooks, configurable tracing, and benchmark baselines.
   - Integrate into `benches/` and document target throughput/latency goals.
5. **Packaging + release automation**
   - Multi-platform wheel builds, ABI compatibility checks, and release checklists.
   - Explicit support matrix for OS/Python/CUDA (even if CUDA is "not yet").
