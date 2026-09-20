"""PyTorch model zoo — used when torch is installed, for larger-scale runs
(especially the image datasets, CIFAR-10 / Fashion-MNIST, where a real CNN
matters). Not required for the default synthetic/adult tabular pipeline,
which runs entirely on `fedpriv.models.mlp_numpy` with zero extra
dependencies.

Exposes the same "flat parameter vector" interface as the NumPy model
(`get_params_vector` / `set_params_vector`) so `fedpriv.training.torch_trainer`
can plug into the exact same FedAvg / DP / network-simulation code as the
NumPy path — the federated + privacy layers never need to know which
backend they're aggregating parameters for.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TorchMLP(nn.Module):
        """For tabular data (synthetic / adult) when a torch backend is preferred."""

        def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class TorchCNN(nn.Module):
        """Small CNN for CIFAR-10 (3x32x32) / Fashion-MNIST (1x28x28)."""

        def __init__(self, in_channels: int, n_classes: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            spatial = 32 // 4 if in_channels == 3 else 28 // 4
            self.fc1 = nn.Linear(32 * spatial * spatial, 128)
            self.fc2 = nn.Linear(128, n_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    def get_params_vector(model: "nn.Module"):
        import numpy as np

        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])

    def set_params_vector(model: "nn.Module", vec) -> None:
        cursor = 0
        for p in model.parameters():
            numel = p.numel()
            chunk = vec[cursor : cursor + numel].reshape(p.shape)
            with torch.no_grad():
                p.copy_(torch.as_tensor(chunk, dtype=p.dtype))
            cursor += numel
