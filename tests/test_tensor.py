import numpy as np
import pytest
import rustorch

def test_add_mul_matmul():
    x = rustorch.PyTensor(np.ones((4, 3), dtype=np.float32), requires_grad=True)
    y = rustorch.PyTensor((np.ones((4, 3), dtype=np.float32) * 2), requires_grad=True)
    z = x.add(y).mul(y)  # (1+2)*2 = 6
    arr = z.numpy()
    assert arr.shape == (4, 3)
    assert np.allclose(arr, 6.0)

    a = rustorch.PyTensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    b = rustorch.PyTensor(np.random.randn(5, 6).astype(np.float32), requires_grad=True)
    c = a.matmul(b)
    assert c.numpy().shape == (4, 6)


def test_numpy_requires_contiguous_layout():
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    non_contig = base.T
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(rustorch.LayoutError):
        rustorch.PyTensor(non_contig, requires_grad=False)
