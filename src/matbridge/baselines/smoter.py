"""SMOTER baseline stub — synthetic minority over-sampling for regression."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor


class SMOTERBaseline:
    name = "smoter"

    def __init__(self, *, eta: float = 0.5, seed: int = 42, tail_quantile: float = 0.9):
        self.eta = eta
        self.seed = seed
        self.tail_quantile = tail_quantile
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        thresh = np.quantile(y, self.tail_quantile)
        tail_mask = y >= thresh
        X_tail = X[tail_mask]
        y_tail = y[tail_mask]
        if len(X_tail) < 2:
            X_aug, y_aug = X, y
        else:
            nn = NearestNeighbors(n_neighbors=min(2, len(X_tail))).fit(X_tail)
            n_syn = max(int(len(X_tail) * self.eta), 1)
            syn_X, syn_y = [], []
            for _ in range(n_syn):
                i = rng.integers(0, len(X_tail))
                _, idx = nn.kneighbors(X_tail[i : i + 1])
                j = idx[0, 1] if idx.shape[1] > 1 else idx[0, 0]
                lam = rng.random()
                syn_X.append(X_tail[i] + lam * (X_tail[j] - X_tail[i]))
                syn_y.append(y_tail[i] + lam * (y_tail[j] - y_tail[i]))
            X_aug = np.vstack([X, np.vstack(syn_X)])
            y_aug = np.concatenate([y, np.array(syn_y)])
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=50, random_state=self.seed)
        self.model.fit(X_aug, y_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
