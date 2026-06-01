"""TabDDPM-aug baseline stub — unconditional diffusion augmentation placeholder."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class TabDDPMAugBaseline:
    name = "tabddpm_aug"

    def __init__(self, *, eta: float = 0.5, seed: int = 42, n_synthetic: int = 64):
        self.eta = eta
        self.seed = seed
        self.n_synthetic = n_synthetic
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_syn = max(int(self.n_synthetic * self.eta), 1)
        idx = rng.integers(0, len(y), size=n_syn)
        noise = 0.05 * rng.standard_normal((n_syn, X.shape[1]))
        X_syn = X[idx] + noise
        y_syn = y[idx] + 0.05 * rng.standard_normal(n_syn)
        X_aug = np.vstack([X, X_syn])
        y_aug = np.concatenate([y, y_syn])
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=50, random_state=self.seed)
        self.model.fit(X_aug, y_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
