"""FDS baseline stub — feature distribution smoothing weights."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class FDSBaseline:
    name = "fds"

    def __init__(self, *, hidden_dim: int = 32, epochs: int = 50, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.model: MLPRegressor | None = None

    def _fds_weights(self, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
        edges = np.linspace(y.min(), y.max() + 1e-8, n_bins + 1)
        idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins).astype(float) + 1.0
        inv = 1.0 / counts[idx]
        return inv / inv.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        w = self._fds_weights(y)
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            max_iter=max(self.epochs, 2),
            random_state=self.seed,
        )
        self.model.fit(X, y, sample_weight=w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
