# Rustorch — Minimal Torch-Style Autograd Backend in Rust

**Project thesis:** Rustorch is a systems-focused, CPU-only tensor + autograd backend that mirrors key Torch semantics (broadcasting, batched matmul, and common NN ops) while keeping the core runtime small and auditable. It favors explicit data movement and a compact autograd engine over heavyweight runtime machinery.

## System Architecture

**Data flow (high level):**

1. **Tensor creation** → tensors are backed by a contiguous `Vec<f32>` with shape metadata.
2. **Op execution** → operations (e.g., `add`, `matmul`, `relu`, `dropout`, `max_pool2d`, `batch_norm`) are implemented in Rust and return new tensors.
3. **Autograd graph** → if a tensor requires gradients, each op attaches a `GradFn` node that captures its parents.
4. **Backward pass** → a recursive traversal walks the graph, materializes grads, and accumulates gradients into per-tensor buffers.
5. **Python bindings (optional)** → PyO3 exposes the same ops and a small `run_fx` hook for Torch FX integration.

**Safety & concurrency model:**
- Tensors are reference-counted (`Arc`) and gradients are protected by `Mutex` to keep mutation safe across shared references.
- Autograd nodes are trait objects (`GradFn`) stored behind `Arc<dyn ... + Send + Sync>`, enabling safe sharing even when multiple tensors reference the same gradient function.

**Torch stack integration:**
- There is no direct `tch-rs`/LibTorch dependency in the Rust core. The backend is fully native Rust with optional Python bindings to interface with Torch tooling.

## Crate Dependency Analysis

| Crate | Role in this project | Notes |
| --- | --- | --- |
| `anyhow` | Error propagation convenience | Used alongside `thiserror` to describe failures cleanly. |
| `thiserror` | Structured error types | Powers `TorchError` variants. |
| `rand` | Randomness for ops | Used for `dropout` RNG. |
| `rayon` | Parallelism (present) | Dependency is present but not currently wired into kernels. |
| `pyo3` (optional) | Python bindings | Enabled by default via the `python-bindings` feature. |
| `numpy` (optional) | NumPy interop | Used to exchange tensor data with Python. |
| `ndarray` (optional) | Array shaping for Python | Used to create NumPy-backed arrays. |

## Key Implementation Details

- **Manual layer definition (no TorchScript loader):** the CLI demonstrates an explicit two-layer MLP built from `Tensor` + `ops`, showing the intended low-level usage path.
- **Broadcasting + batched matmul:** shape logic is implemented in the tensor core and used by `ops::matmul` to support batched matrix multiplication.
- **Autograd graph:** each op attaches a dedicated `GradFn` that accumulates into per-tensor gradient buffers; `backward` performs a recursive traversal with cycle protection.
- **CPU-only execution:** there is no device abstraction or GPU dispatch path in the current code; all kernels are straightforward CPU loops.
- **Python exposure:** `PyTensor` mirrors the Rust ops and can call into a Torch FX runner via `run_fx` to integrate with Torch tooling.
- **Layout/stride semantics:** tensors are expected to be contiguous; strided layouts are validated against storage span requirements and rejected with explicit `LayoutError` diagnostics in Rust and Python bindings.

## Performance Philosophy

Rustorch is designed as a **systems-level prototype**: zero-cost abstractions, predictable memory layouts, and explicit ownership give C++-class performance without a GC. The runtime aims to keep tensor storage and autograd structures transparent, so it’s easy to reason about performance and memory behavior. Parallelism is intended via `rayon` in future kernels, but the current implementation prioritizes clarity and correctness.

### Benchmarks (placeholder)

| Benchmark | Description | Status |
| --- | --- | --- |
| `bench_elementwise` | Element-wise ops throughput | Defined in `benches/bench_elementwise.rs` (results TBD). |

## Build & Environment Setup

### Rust-only build (no Python bindings)

```bash
cargo build --release --no-default-features --bin rustorch_cli
cargo run --release --no-default-features --bin rustorch_cli
```

### Python bindings (default feature)

```bash
cargo build --release
maturin develop --release
```

To run the Python bindings test:

```bash
pytest tests/test_tensor.py
```

### LibTorch environment (if integrating `tch-rs` or custom LibTorch FFI)

Rustorch does **not** currently depend on `tch-rs`, but if you extend the project with LibTorch-based crates or FFI, use the standard LibTorch environment variables:

```bash
# Point to an extracted LibTorch distribution
export LIBTORCH=/path/to/libtorch
# Ensure the runtime linker can find libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
# (macOS) export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"
```

Then build with Cargo as usual:

```bash
cargo build --release
```

## Project Map

```
.
├── Cargo.toml
├── README.md
├── src
│   ├── autograd.rs        # GradFn graph + backward traversal
│   ├── error.rs           # TorchError + Result types
│   ├── lib.rs             # Crate exports
│   ├── main.rs            # CLI demo (MLP forward/backward)
│   ├── ops
│   │   ├── kernels.rs     # CPU kernels (matmul/add/mul)
│   │   └── mod.rs         # Tensor ops + loss functions
│   ├── py.rs              # PyO3 bindings + Torch FX hook
│   ├── storage.rs         # Tensor storage wrapper
│   └── tensor.rs          # Tensor + shape/broadcast logic
├── python
│   └── rust_backend        # Torch backend registration helpers
├── tests                  # Rust + Python tests
├── examples               # Torch compile / MNIST demos
├── benches                # Criterion benchmarks
├── cpp_ext                # Optional C++ extension
└── docs                   # Integration plans + notes
```

## Torch Backend Demos

```bash
python examples/compile_backend_demo.py
```

MNIST demo (cloud-gated dataset download):

```bash
export CLOUD_MNIST_OK=1
python examples/mnist_rustorch_demo.py
```

Override dataset location:

```bash
RUSTORCH_MNIST_ROOT=/path/to/data python examples/mnist_rustorch_demo.py
```
