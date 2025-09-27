
# rustorch (dual mode)

Rust tensor + autograd engine with optional Python bindings.

## Build with Python (default)

```bash
cargo build --release
maturin develop --release
```

## Build Rust-only (no Python)

```bash
cargo build --release --no-default-features --bin rustorch_cli
cargo run --release --no-default-features --bin rustorch_cli
```
