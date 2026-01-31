# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Rust core (tensor, autograd, ops, storage, CLI in `main.rs`).
- `src/checkpoint.rs` implements versioned checkpoint serialization and `state_dict` save/load.
- `tests/` contains Rust integration tests plus a Python test (`test_tensor.py`).
- `python/` hosts Python-side packaging/bindings helpers.
- `cpp_ext/` contains optional C++ extension sources.
- `examples/`, `benches/`, and `docs/` provide usage, benchmarks, and documentation.
- `docs/torch_integration_plan.md` tracks the staged PyTorch integration plan (M0–M4) and testing milestones.

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
- `scripts/build_rust_python.sh` and `scripts/build_cpp_ext.sh`: helper build scripts (see script content before use).

## Coding Style & Naming Conventions
- Rust 2021 edition; use `snake_case` for modules/functions and `CamelCase` for types/traits.
- Default rustfmt style (4-space indentation). Run `cargo fmt` before committing if available.
- Prefer explicit error handling with `anyhow`/`thiserror` patterns already used in the codebase.

## Testing Guidelines
- Rust tests live in `tests/` (e.g., `forward_ops.rs`, `backward_ops.rs`, `integration.rs`).
- Python coverage is minimal and focused on bindings; keep tests in `tests/test_tensor.py`.
- Name tests descriptively and keep them grouped by operation/feature area.
- Keep new backend tests aligned with milestones in `docs/torch_integration_plan.md`.

## Intermediate Development Task (Codex Cloud)
- Add an intermediate milestone: set up a PyTorch NN that uses `rustorch` as the backend and performs MNIST classification with minimal loss.
- This task is intended to run in the Codex cloud environment so MNIST can be fetched there (no local dataset downloads).
- When implementing the MNIST training script, gate dataset downloads behind a cloud-only signal (e.g., require an environment variable like `CODEX_CLOUD=1` or a `CLOUD_MNIST_OK=1` flag). If the flag is missing, the script should exit with a clear message instead of downloading.
- Status:
  - Implemented in `examples/mnist_rustorch_demo.py` with download gating via
    `CLOUD_MNIST_OK=1`, `CODEX_CLOUD=1`, or `MNIST_ALLOW_DOWNLOAD=1`.
  - Training loop uses a CNN compiled with `torch.compile(backend="rust_backend")`
    and includes conv2d + cross-entropy loss.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and specific (e.g., “Implement MSE loss”).
- PRs should include: a concise summary, key changes, and how tests were run.
- Link related issues if they exist; include before/after notes for API changes.

## Production-Grade Roadmap (Next 2–3 Milestones)
1. **Autograd graph scheduler + gradient accumulation**
   - Dependency-tracked backward traversal for shared subgraphs.
   - Fail-fast validation for loss scalar/grad buffers.
   - Lightweight observability (logging/stats) for traversal health.
2. **Tensor layout/stride semantics + error taxonomy**
   - Formalize contiguous/strided tensor invariants.
   - Add shape/stride validation in core ops + FX lowering.
   - Expose structured errors via Rust + Python.
3. **Serialization & checkpointing**
   - Implement `state_dict`-style save/load (Rust + Python).
   - Versioned metadata with dtype/layout compatibility checks.
4. **Release hardening & packaging automation**
   - CI matrix for Rust + Python bindings + FX demos.
   - Automated wheel builds and benchmark regression checks.

## Configuration Tips
- Python bindings are enabled by default via the `python-bindings` feature. Disable with `--no-default-features` for Rust-only builds.
- Ensure a compatible Python (>=3.9) environment when running `maturin develop`.
- `python/rust_backend/backend.py` registers the `rust_backend` for `torch.compile` and expects `rustorch.run_fx`.
- Use `rustorch.save_state_dict` / `rustorch.load_state_dict` from Python for checkpoint round-trips (format header `RTCH`, versioned metadata + raw f32 payload).
- `cpp_ext/` is a PrivateUse1 example; its kernels are expected to be wired to Rust via FFI.
- GitHub CLI in this environment is gitsome; use `gh create-issue owner/repo -t ... -d ...` for issues.
