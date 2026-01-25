import logging

import pytest


torch = pytest.importorskip("torch")

import rust_backend.backend as _  # noqa: F401


class MnistTiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 26 * 26, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MnistTrainer(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        return self.criterion(output, target)


def test_compile_backend_handles_mnist_backward_ops(caplog: pytest.LogCaptureFixture) -> None:
    model = MnistTiny()
    trainer = MnistTrainer(model)
    compiled = torch.compile(trainer, backend="rust_backend")

    batch = 4
    data = torch.randn(batch, 1, 28, 28)
    target = torch.randint(0, 10, (batch,))

    caplog.set_level(logging.WARNING, logger="rust_backend.fx_runner")
    loss = compiled(data, target)
    loss.backward()

    assert loss.item() == pytest.approx(loss.item())
    assert model.net[0].weight.grad is not None
    assert model.net[-1].weight.grad is not None
    assert "fallback to eager" not in caplog.text
