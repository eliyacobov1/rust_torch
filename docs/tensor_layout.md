# Tensor layout semantics

Rustorch tensors track both **shape** and **strides** to describe layout. The runtime distinguishes two layout classes:

- **Contiguous**: strides match the default row-major layout for the shape.
- **Strided**: strides are valid for the shape but differ from the contiguous layout.

## Layout invariants

All tensors must satisfy the following invariants:

- `shape.len()` equals `strides.len()`.
- For non-empty tensors, every dimension with `dim > 1` has a `stride > 0`.
- The backing storage length must match the required span implied by the shape/strides.
- Zero-sized tensors must use empty storage.

## Contiguity requirements

Most Rustorch ops (and the FX runner inputs) currently require **contiguous** layouts. Operations that accept strided layouts validate them explicitly and will emit `TorchError::InvalidLayout` or `TorchError::NonContiguous` when the invariants are violated.

If you need to work with strided tensors, validate the layout with `Tensor::validate_strided_layout` before passing them into ops that support non-contiguous layouts.

For contiguous tensors constructed from raw data, prefer `Tensor::try_from_vec_f32` to validate the storage length matches the shape. The Python bindings use this path and will emit a `LayoutError` if the layout invariants are violated.
