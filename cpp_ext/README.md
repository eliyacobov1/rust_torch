# PrivateUse1 Device Backend (C++ shim)

This directory shows how to register a tiny custom device backend using PyTorch's **PrivateUse1**.
We implement `aten::add.Tensor` for the custom device and set a fallback to CPU for unsupported ops.

Next steps (TODO):
- Wire the kernel to Rust via FFI (e.g., build Rust staticlib + `extern "C"` functions).
- Implement more kernels (`mul`, `mm`, `relu`, `sum`, `empty_strided`, etc.).
- Add a device guard and (optional) generator.

Build:
```bash
python build.py
python ../examples/eager_privateuse1_demo.py
```