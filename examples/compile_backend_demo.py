import torch
import python.rust_backend.backend as _  # register backend
import rustorch

@torch.compile(backend="rust_backend")
def toy(x, y):
    # For now this runs as-is (no-op compile). Extend run_fx in Rust to intercept ops.
    return (x @ y).relu()

if __name__ == "__main__":
    x = torch.randn(16, 32)
    y = torch.randn(32, 8)
    out = toy(x, y)
    print("OK, out:", tuple(out.shape))