"""CopulaCalib vine-copula joint sampler stub (matched-eta fence).

Full vine implementation not in CopulaCalib reference repo; this stub mirrors
the Gaussian joint API and documents the canonical path for future parity:

  6_2026_Spring_Conference/4_ICDM_2026/B_CopulaCalib/2_Code/copulacalib.py
  (Gaussian only — vine is DUEL contract comparator, to be implemented via
   pyvinecopulib or bivariate cascade following ICDM manuscript spec)

For smoke: falls back to pairwise Gaussian bivariate copula chain (stub).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from matbridge.baselines.copula_gaussian import (
    CopulaCalibGaussianJoint,
    apply_empirical_quantile,
    calibrate_moments,
    compute_normal_scores,
    compute_pseudo_observations,
    create_bins,
    estimate_correlation_matrix,
)

COPULACALIB_REF = Path(__file__).resolve().parents[5] / (
    "6_2026_Spring_Conference/4_ICDM_2026/B_CopulaCalib/2_Code/copulacalib.py"
)


def _vine_chain_sample(U_bin: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Stub: sequential bivariate Gaussian copula conditioning (C-vine lite)."""
    rng = np.random.default_rng(seed)
    d = U_bin.shape[1]
    if d == 1:
        return rng.uniform(size=(n_samples, 1))
    Z = compute_normal_scores(U_bin)
    Sigma = estimate_correlation_matrix(Z)
    out = np.zeros((n_samples, d))
    out[:, 0] = rng.uniform(size=n_samples)
    for j in range(1, d):
        rho = float(np.clip(Sigma[0, j], -0.99, 0.99))
        z0 = stats.norm.ppf(np.clip(out[:, 0], 1e-6, 1 - 1e-6))
        zj = rho * z0 + np.sqrt(max(1 - rho ** 2, 1e-6)) * rng.standard_normal(n_samples)
        out[:, j] = stats.norm.cdf(zj)
    return out


class CopulaCalibVineJoint:
    name = "copula_vine_joint"
    reference_path = str(COPULACALIB_REF)

    def __init__(
        self,
        *,
        n_bins: int = 10,
        synthetic_factor: float = 2.0,
        eta: float = 0.5,
        seed: int = 42,
    ):
        self.n_bins = n_bins
        self.synthetic_factor = synthetic_factor
        self.eta = eta
        self.seed = seed
        self.model = None
        self.last_X_aug: np.ndarray | None = None
        self.last_y_aug: np.ndarray | None = None
        self._gaussian_fallback = CopulaCalibGaussianJoint(
            n_bins=n_bins, synthetic_factor=synthetic_factor, eta=eta, seed=seed
        )

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        min_samples = max(10, n_samples // (self.n_bins * 2))
        bin_edges, bin_indices = create_bins(y, self.n_bins)
        syn_X, syn_y = [], []
        for k in range(self.n_bins):
            mask = bin_indices == k
            n_k = int(mask.sum())
            if n_k < 3:
                continue
            deficit = max(0, min_samples - n_k)
            if deficit <= 0:
                continue
            n_syn = int(deficit * self.synthetic_factor * self.eta)
            if n_syn <= 0:
                continue
            X_bin = X[mask]
            U_bin = compute_pseudo_observations(X_bin)
            U_syn = _vine_chain_sample(U_bin, n_syn, self.seed + k)
            X_syn = apply_empirical_quantile(U_syn, X_bin)
            X_syn = calibrate_moments(X_syn, X_bin)
            y_syn = np.random.uniform(bin_edges[k], bin_edges[k + 1], size=n_syn)
            syn_X.append(X_syn)
            syn_y.append(y_syn)
        if syn_X:
            X_aug = np.vstack([X, *syn_X]).astype(np.float64)
            y_aug = np.concatenate([y, *syn_y]).astype(np.float64)
        else:
            X_aug, y_aug = X.astype(np.float64), y.astype(np.float64)
        self.last_X_aug = X_aug
        self.last_y_aug = y_aug
        return X_aug, y_aug

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> None:
        from sklearn.neural_network import MLPRegressor

        X_aug, y_aug = self.fit_transform(X, y)
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=max(epochs, 10),
            random_state=self.seed,
            learning_rate_init=1e-3,
        )
        self.model.fit(X_aug, y_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()
