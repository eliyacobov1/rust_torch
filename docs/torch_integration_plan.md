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
- Parity harness tests now compare eager vs compiled forward/grad results for
  representative MLP and CNN models (M4).

## Next recommended task (post-M4)
**Expand operator + autograd coverage for production training loops.**

**Why this next?**
- Parity harness coverage is in place, but the operator surface is still too
  narrow for modern training stacks beyond the MNIST demo.
- Closing the gap on high-value ops (batchnorm, pooling, log-softmax) unlocks
  more realistic models and provides higher confidence in correctness.

**Scope (significant, production-oriented)**
- Implement forward/backward kernels for `max_pool2d`, `batch_norm`, `dropout`,
  `log_softmax`, and `nll_loss` (or `cross_entropy` fused paths).
- Extend FX lowering to route these ops through the Rust backend with clear
  error messages when unsupported.
- Add Rust tests in `tests/` for forward/backward parity of new kernels.
- Extend the parity harness to include a small CNN with batchnorm + pooling and
  a classifier head that exercises the new losses.

### Decision: focus task for the next development cycle
**Build the "CNN training stack" operator set with full parity coverage.**

**Acceptance criteria**
- Forward/backward kernels for `max_pool2d`, `batch_norm`, `log_softmax`,
  `nll_loss` (or fused `cross_entropy`) implemented in Rust.
- FX lowering routes these ops to the Rust backend and emits clear errors for
  unsupported tensor layouts or shapes.
- Parity harness extended with a CNN + batchnorm + pooling model and validated
  against eager PyTorch for forward and backward checks.
- Rust tests added for each new op (forward + backward) with numeric tolerances.

**Why this is the next task**
- It extends the current MNIST demo into a representative production-style
  training loop without changing infrastructure already delivered (FX runner,
  parity harness, CI).
- It de-risks correctness on the most common CNN building blocks before taking
  on larger API or performance investments.

## Production readiness roadmap (proposed)
1. **Operator + autograd coverage for common training stacks**
   - Implement backward ops needed for CNNs and MLPs beyond MNIST (conv2d, maxpool2d,
     batchnorm, log_softmax, and cross-entropy loss), plus corresponding FX lowering.
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
