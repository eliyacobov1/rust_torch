
# rustorch (dual mode)

Rust tensor + autograd engine with optional Python bindings.

## Build with Python (default)

```bash
cargo build --release
maturin develop --release
```

To build a distributable Python wheel:

```bash
maturin build --release
```

To run the Python bindings test:

```bash
pytest tests/test_tensor.py
```

## Build Rust-only (no Python)

```bash
cargo build --release --no-default-features --bin rustorch_cli
cargo run --release --no-default-features --bin rustorch_cli
```

## PyTorch compile backend demos

Register the backend and run a small compile demo:

```bash
python examples/compile_backend_demo.py
```

Run the MNIST training demo (downloads are gated for cloud environments):

```bash
export CLOUD_MNIST_OK=1
python examples/mnist_rustorch_demo.py
```

By default the dataset is stored in `~/.cache/rustorch/mnist`. Override the
location with `RUSTORCH_MNIST_ROOT=/path/to/data` if needed.

For quick smoke tests, you can limit the dataset size:

```bash
MNIST_TRAIN_LIMIT=2048 MNIST_TEST_LIMIT=1024 python examples/mnist_rustorch_demo.py
```
