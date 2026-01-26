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
