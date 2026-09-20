"""A small MLP implemented directly in NumPy (manual forward/backward).

Why not just require PyTorch? Because the whole point of this reference
implementation is that `pip install -r requirements.txt` + `python app.py`
works everywhere, including offline/CI/sandboxed environments, and the
FL + DP mechanics (FedAvg, clipping, noise, accounting, non-IID
partitioning, network simulation) are exactly the same regardless of what
computes the gradients. Swap in `fedpriv.models.torch_models` for GPU-
scale image experiments without touching any federated/privacy code —
both expose the same "flat parameter vector in, flat parameter vector +
gradient out" interface, which is what makes them interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class MLPShapes:
    in_dim: int
    hidden_dim: int
    n_classes: int

    def shapes(self):
        return [
            ("W1", (self.in_dim, self.hidden_dim)),
            ("b1", (self.hidden_dim,)),
            ("W2", (self.hidden_dim, self.n_classes)),
            ("b2", (self.n_classes,)),
        ]

    def n_params(self) -> int:
        return sum(int(np.prod(s)) for _, s in self.shapes())


class NumpyMLP:
    """Single-hidden-layer MLP: Linear -> ReLU -> Linear -> Softmax."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 64, seed: int = 42):
        self.shapes = MLPShapes(in_dim, hidden_dim, n_classes)
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim)).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, n_classes)).astype(np.float32),
            "b2": np.zeros(n_classes, dtype=np.float32),
        }

    # ---- flat parameter (de)serialization: this is the wire format FedAvg/DP operate on ----

    def get_params_vector(self) -> np.ndarray:
        return np.concatenate([self.params[name].ravel() for name, _ in self.shapes.shapes()])

    def set_params_vector(self, vec: np.ndarray) -> None:
        cursor = 0
        for name, shape in self.shapes.shapes():
            size = int(np.prod(shape))
            self.params[name] = vec[cursor : cursor + size].reshape(shape).astype(np.float32)
            cursor += size

    def n_params(self) -> int:
        return self.shapes.n_params()

    # ---- forward / backward ----

    def forward(self, X: np.ndarray):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        probs = _softmax(z2)
        cache = (X, z1, a1, probs)
        return probs, cache

    def loss_and_grad_vector(self, X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4
                              ) -> Tuple[float, np.ndarray]:
        n = X.shape[0]
        probs, (X_, z1, a1, _) = self.forward(X)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y] = 1.0
        loss = -np.sum(y_onehot * np.log(np.clip(probs, 1e-9, 1.0))) / n
        loss += l2_reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))

        dz2 = (probs - y_onehot) / n
        dW2 = a1.T @ dz2 + 2 * l2_reg * self.params["W2"]
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = X_.T @ dz1 + 2 * l2_reg * self.params["W1"]
        db1 = dz1.sum(axis=0)

        grad_vector = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return float(loss), grad_vector.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs
