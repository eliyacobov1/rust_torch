import copy
from typing import Iterable

import pytest


torch = pytest.importorskip("torch")

import rust_backend.backend as _  # noqa: F401


def _clone_module(module: torch.nn.Module) -> torch.nn.Module:
    clone = copy.deepcopy(module)
    clone.load_state_dict(module.state_dict())
    return clone


def _collect_param_grads(module: torch.nn.Module) -> list[torch.Tensor]:
    grads = []
    for param in module.parameters():
        assert param.grad is not None
        grads.append(param.grad.detach().clone())
    return grads


def _assert_grad_parity(eager_grads: Iterable[torch.Tensor], compiled_grads: Iterable[torch.Tensor]) -> None:
    for eager_grad, compiled_grad in zip(eager_grads, compiled_grads, strict=True):
        torch.testing.assert_close(eager_grad, compiled_grad, rtol=1e-4, atol=1e-5)


def test_parity_harness_mlp_forward_and_backward() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    loss_fn = torch.nn.MSELoss()

    inputs = torch.randn(6, 8)
    targets = torch.randn(6, 4)

    eager_model = _clone_module(model)
    eager_model.train()
    eager_model.zero_grad(set_to_none=True)
    eager_out = eager_model(inputs)
    eager_loss = loss_fn(eager_out, targets)
    eager_loss.backward()

    compiled_model = _clone_module(model)
    compiled_model.train()
    compiled_model.zero_grad(set_to_none=True)
    compiled = torch.compile(compiled_model, backend="rust_backend")
    compiled_out = compiled(inputs)
    compiled_loss = loss_fn(compiled_out, targets)
    compiled_loss.backward()

    torch.testing.assert_close(eager_out, compiled_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eager_loss, compiled_loss, rtol=1e-4, atol=1e-5)
    _assert_grad_parity(_collect_param_grads(eager_model), _collect_param_grads(compiled_model))


def test_parity_harness_cnn_forward_and_backward() -> None:
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 26 * 26, 10),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))

    eager_model = _clone_module(model)
    eager_model.train()
    eager_model.zero_grad(set_to_none=True)
    eager_out = eager_model(inputs)
    eager_loss = loss_fn(eager_out, targets)
    eager_loss.backward()

    compiled_model = _clone_module(model)
    compiled_model.train()
    compiled_model.zero_grad(set_to_none=True)
    compiled = torch.compile(compiled_model, backend="rust_backend")
    compiled_out = compiled(inputs)
    compiled_loss = loss_fn(compiled_out, targets)
    compiled_loss.backward()

    torch.testing.assert_close(eager_out, compiled_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eager_loss, compiled_loss, rtol=1e-4, atol=1e-5)
    _assert_grad_parity(_collect_param_grads(eager_model), _collect_param_grads(compiled_model))
