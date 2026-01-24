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

## M3: Autograd coverage
- Ensure backward ops used by the MNIST demo are routed through the backend.
- Add/extend Rust tests in `tests/` to cover new backward ops.

## M4: Performance & parity
- Compare results with eager PyTorch for correctness on representative models.
- Document performance notes and remaining operator gaps.
