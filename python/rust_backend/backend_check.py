import torch
import rust_backend  # noqa: F401  (ensures backend is registered)

@torch.compile(backend="rust_backend")
def f(x, y):
    return (x @ y).relu()

if __name__ == "__main__":
    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    out = f(x, y)
    print("compile backend roundtrip OK, out shape:", out.shape)