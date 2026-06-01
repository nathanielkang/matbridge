"""Encoder g with PCA on standardized (x, y) — frozen after fit (Assumption B1')."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class EncoderConfig:
    embed_dim: int = 16
    spectral_norm: bool = True


@dataclass
class EncoderResult:
    z_train: np.ndarray
    z_test: np.ndarray
    lipschitz_g_hat: float
    frozen: bool
    encoder: TabularEncoder | None = None


class TabularEncoder:
    """PCA embedding on [X, y] with optional Lipschitz logging."""

    def __init__(self, config: EncoderConfig | None = None, seed: int = 42):
        self.config = config or EncoderConfig()
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._frozen = False

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 2) -> None:
        del epochs  # PCA is closed-form; epochs kept for API parity
        xy = np.hstack([X, y.reshape(-1, 1).astype(np.float64)])
        self._scaler = StandardScaler().fit(xy)
        xy_s = self._scaler.transform(xy)
        n_comp = min(self.config.embed_dim, xy_s.shape[1], max(xy_s.shape[0] - 1, 1))
        self._pca = PCA(n_components=n_comp, random_state=self.seed)
        self._pca.fit(xy_s)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self._scaler is None or self._pca is None:
            raise RuntimeError("Encoder not fit")
        xy = np.hstack([X, y.reshape(-1, 1).astype(np.float64)])
        xy_s = self._scaler.transform(xy)
        return self._pca.transform(xy_s).astype(np.float64)

    def empirical_lipschitz(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        z = self.transform(X_val, y_val)
        xy = np.hstack([X_val, y_val.reshape(-1, 1)])
        dist_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dist_z = np.linalg.norm(np.diff(z, axis=0), axis=1)
        mask = dist_xy > 1e-8
        if not mask.any():
            return 0.0
        return float(np.median(dist_z[mask] / dist_xy[mask]))


def fit_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 2,
    seed: int = 42,
) -> EncoderResult:
    enc = TabularEncoder(seed=seed)
    enc.fit(X_train, y_train, epochs=epochs)
    enc.freeze()
    return EncoderResult(
        z_train=enc.transform(X_train, y_train),
        z_test=enc.transform(X_val, y_val),
        lipschitz_g_hat=enc.empirical_lipschitz(X_val, y_val),
        frozen=enc.frozen,
        encoder=enc,
    )
