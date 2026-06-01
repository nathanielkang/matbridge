"""DenseLoss baseline stub — inverse-density sample weights on y."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KernelDensity


class DenseLossBaseline:
    name = "denseloss"

    def __init__(self, *, hidden_dim: int = 32, epochs: int = 50, seed: int = 42, bandwidth: float = 0.5):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.bandwidth = bandwidth
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
        kde.fit(y.reshape(-1, 1))
        log_d = kde.score_samples(y.reshape(-1, 1))
        w = np.exp(-log_d)
        w = w / (w.mean() + 1e-8)
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            max_iter=max(self.epochs, 2),
            random_state=self.seed,
        )
        self.model.fit(X, y, sample_weight=w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
