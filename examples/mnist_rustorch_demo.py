import gzip
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import utils as tv_utils
from torchvision.datasets.mnist import read_image_file, read_label_file

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import rust_backend.backend as _  # noqa: F401 - register backend

DEFAULT_DATA_ROOT = Path(os.environ.get("RUSTORCH_MNIST_ROOT", "~/.cache/rustorch/mnist")).expanduser()
DOWNLOAD_FLAG = (
    os.environ.get("CLOUD_MNIST_OK") == "1"
    or os.environ.get("CODEX_CLOUD") == "1"
    or os.environ.get("MNIST_ALLOW_DOWNLOAD") == "1"
)
MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _mnist_exists(data_root: Path) -> bool:
    training_file = data_root / "MNIST" / "processed" / "training.pt"
    test_file = data_root / "MNIST" / "processed" / "test.pt"
    return training_file.exists() and test_file.exists()


def _build_dataloaders(data_root: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    if DOWNLOAD_FLAG and not _mnist_exists(data_root):
        _download_and_prepare_mnist(data_root)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        data_root,
        train=True,
        download=DOWNLOAD_FLAG,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        data_root,
        train=False,
        download=DOWNLOAD_FLAG,
        transform=transform,
    )
    train_limit_env = os.environ.get("MNIST_TRAIN_LIMIT")
    test_limit_env = os.environ.get("MNIST_TEST_LIMIT")
    if train_limit_env is None and test_limit_env is None and not DOWNLOAD_FLAG:
        train_limit = 10_000
        test_limit = 2_000
    else:
        train_limit = int(train_limit_env or "0")
        test_limit = int(test_limit_env or "0")
    if train_limit > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_limit))
    if test_limit > 0:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_limit))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _download_and_prepare_mnist(data_root: Path) -> None:
    raw_dir = data_root / "MNIST" / "raw"
    processed_dir = data_root / "MNIST" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for filename, md5 in MNIST_FILES.items():
        gz_path = raw_dir / filename
        extracted_path = raw_dir / filename.replace(".gz", "")
        if not extracted_path.exists():
            tv_utils.download_url(MNIST_MIRROR + filename, raw_dir, filename=filename, md5=md5)
            with gzip.open(gz_path, "rb") as gz_file:
                with open(extracted_path, "wb") as out_file:
                    out_file.write(gz_file.read())

    training_file = processed_dir / "training.pt"
    test_file = processed_dir / "test.pt"
    if not training_file.exists():
        train_data = read_image_file(raw_dir / "train-images-idx3-ubyte")
        train_labels = read_label_file(raw_dir / "train-labels-idx1-ubyte")
        torch.save((train_data, train_labels), training_file)
    if not test_file.exists():
        test_data = read_image_file(raw_dir / "t10k-images-idx3-ubyte")
        test_labels = read_label_file(raw_dir / "t10k-labels-idx1-ubyte")
        torch.save((test_data, test_labels), test_file)


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 26 * 26, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MnistTrainer(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        return self.criterion(output, target)


def train_epoch(
    trainer: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    trainer.train()
    total_loss = 0.0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = trainer(data, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(data_loader.dataset)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(data_loader.dataset)


def main() -> None:
    if not DOWNLOAD_FLAG and not _mnist_exists(DEFAULT_DATA_ROOT):
        print(
            "MNIST dataset not found. Set CLOUD_MNIST_OK=1 (or CODEX_CLOUD=1) "
            "to allow download in the cloud environment, or set MNIST_ALLOW_DOWNLOAD=1 "
            "to download locally.",
            flush=True,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("MNIST_BATCH_SIZE", "128"))
    epochs = int(os.environ.get("MNIST_EPOCHS", "3"))
    lr = float(os.environ.get("MNIST_LR", "1e-3"))

    train_loader, test_loader = _build_dataloaders(DEFAULT_DATA_ROOT, batch_size)

    model = MnistCNN().to(device)
    trainer = MnistTrainer(model).to(device)
    compiled_trainer = torch.compile(trainer, backend="rust_backend")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(compiled_trainer, optimizer, train_loader, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}", flush=True)

    accuracy = evaluate(model, test_loader, device)
    print(f"Final accuracy: {accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
