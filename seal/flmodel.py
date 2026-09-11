"""A minimal federated (multinomial) softmax classifier, trained with FedAvg.

Kept dependency-light (numpy only) so that every step of training,
unlearning and per-sample loss evaluation used by the audit protocol is
fully transparent and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class Params:
    W: np.ndarray
    b: np.ndarray

    def copy(self) -> "Params":
        return Params(self.W.copy(), self.b.copy())


@dataclass
class FederatedSoftmax:
    """Multinomial logistic regression trained via FedAvg.

    ``forward`` / ``per_sample_loss`` are also the primitives the audit
    certificate calls on the client's own data, so they live on the model
    rather than in the training loop.
    """

    n_features: int
    n_classes: int
    l2: float = 1e-3
    params: Params = field(init=False)

    def __post_init__(self) -> None:
        self.params = Params(
            W=np.zeros((self.n_features, self.n_classes)),
            b=np.zeros(self.n_classes),
        )

    def init(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.params = Params(
            W=rng.normal(scale=0.01, size=(self.n_features, self.n_classes)),
            b=np.zeros(self.n_classes),
        )

    def forward(self, X: np.ndarray, params: Params | None = None) -> np.ndarray:
        p = params or self.params
        return _softmax(X @ p.W + p.b)

    def per_sample_loss(self, X: np.ndarray, y: np.ndarray, params: Params | None = None) -> np.ndarray:
        probs = self.forward(X, params)
        eps = 1e-12
        return -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))

    def accuracy(self, X: np.ndarray, y: np.ndarray, params: Params | None = None) -> float:
        probs = self.forward(X, params)
        return float((probs.argmax(axis=1) == y).mean())

    def _grad(self, X: np.ndarray, y: np.ndarray, params: Params) -> Params:
        n = X.shape[0]
        probs = self.forward(X, params)
        Y = _one_hot(y, self.n_classes)
        gW = X.T @ (probs - Y) / n + self.l2 * params.W
        gb = (probs - Y).mean(axis=0)
        return Params(gW, gb)

    def local_sgd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Params,
        lr: float,
        epochs: int,
        batch_size: int,
        seed: int,
        ascent: bool = False,
    ) -> Params:
        """Run `epochs` passes of mini-batch SGD starting from `params`.

        `ascent=True` performs gradient *ascent* (used to construct the
        dishonest "spoofing" mechanism, which deliberately raises the loss
        on the forget set without touching anything else).
        """
        rng = np.random.default_rng(seed)
        p = params.copy()
        n = X.shape[0]
        if n == 0:
            return p
        sign = 1.0 if ascent else -1.0
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                g = self._grad(X[idx], y[idx], p)
                p.W = p.W + sign * lr * g.W
                p.b = p.b + sign * lr * g.b
        return p

    def fedavg_train(
        self,
        clients_X: list[np.ndarray],
        clients_y: list[np.ndarray],
        rounds: int,
        local_epochs: int,
        lr: float,
        batch_size: int,
        seed: int,
        init_params: Params | None = None,
    ) -> Params:
        """Standard FedAvg: local SGD at each client, weighted average of params."""
        params = (init_params.copy() if init_params is not None else self.params.copy())
        sizes = np.array([max(len(X), 1) for X in clients_X], dtype=float)
        weights = sizes / sizes.sum()
        for r in range(rounds):
            local_params = []
            for k, (X, y) in enumerate(zip(clients_X, clients_y)):
                lp = self.local_sgd(X, y, params, lr, local_epochs, batch_size, seed=seed * 10_000 + r * 100 + k)
                local_params.append(lp)
            W = sum(w * lp.W for w, lp in zip(weights, local_params))
            b = sum(w * lp.b for w, lp in zip(weights, local_params))
            params = Params(W, b)
        return params
