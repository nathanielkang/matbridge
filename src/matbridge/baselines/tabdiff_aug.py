"""TabDiff-aug baseline stub — score-based tabular diffusion placeholder."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class TabDiffAugBaseline:
    name = "tabdiff_aug"

    def __init__(self, *, eta: float = 0.5, seed: int = 42, n_synthetic: int = 64):
        self.eta = eta
        self.seed = seed
        self.n_synthetic = n_synthetic
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed + 1)
        n_syn = max(int(self.n_synthetic * self.eta), 1)
        mu = X.mean(axis=0)
        cov = np.cov(X.T) + 1e-4 * np.eye(X.shape[1])
        X_syn = rng.multivariate_normal(mu, cov, size=n_syn)
        y_syn = y.mean() + 0.1 * rng.standard_normal(n_syn)
        X_aug = np.vstack([X, X_syn])
        y_aug = np.concatenate([y, y_syn])
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=50, random_state=self.seed)
        self.model.fit(X_aug, y_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
