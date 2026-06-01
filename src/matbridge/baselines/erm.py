"""ERM baseline — vanilla regressor on real rows only."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class ERMBaseline:
    name = "erm"

    def __init__(self, *, hidden_dim: int = 32, epochs: int = 50, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.seed = seed
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, max(self.hidden_dim // 2, 1)),
            max_iter=max(self.epochs, 2),
            random_state=self.seed,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
