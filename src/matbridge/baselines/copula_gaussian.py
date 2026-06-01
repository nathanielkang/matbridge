"""CopulaCalib Gaussian-copula joint sampler (matched-eta fence).

Adapted from:
  6_2026_Spring_Conference/4_ICDM_2026/B_CopulaCalib/2_Code/copulacalib.py

Patterns copied: compute_pseudo_observations, compute_normal_scores,
estimate_correlation_matrix, sample_from_gaussian_copula, apply_empirical_quantile,
calibrate_moments, create_bins, CopulaCalib.fit_transform joint pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats
from scipy.linalg import cholesky

# Reference path for parity audits (DUEL_EXECUTION_CONTRACT matched-eta).
COPULACALIB_REF = Path(__file__).resolve().parents[5] / (
    "6_2026_Spring_Conference/4_ICDM_2026/B_CopulaCalib/2_Code/copulacalib.py"
)


def compute_pseudo_observations(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    U = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        ranks = stats.rankdata(X[:, j], method="average")
        U[:, j] = ranks / (n + 1)
    return U


def compute_normal_scores(U: np.ndarray) -> np.ndarray:
    U_clipped = np.clip(U, 1e-10, 1 - 1e-10)
    return stats.norm.ppf(U_clipped)


def estimate_correlation_matrix(Z: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    n, d = Z.shape
    if n < d:
        sample_corr = np.corrcoef(Z.T)
        Sigma = 0.5 * sample_corr + 0.5 * np.eye(d)
    else:
        Sigma = np.corrcoef(Z.T)
    Sigma = Sigma + regularization * np.eye(d)
    Sigma = np.nan_to_num(Sigma, nan=0.0)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def sample_from_gaussian_copula(Sigma: np.ndarray, n_samples: int, random_state: int | None = None) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
    d = Sigma.shape[0]
    try:
        L = cholesky(Sigma, lower=True)
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = cholesky(Sigma, lower=True)
    Z = np.random.randn(n_samples, d) @ L.T
    return stats.norm.cdf(Z)


def apply_empirical_quantile(U: np.ndarray, X_original: np.ndarray) -> np.ndarray:
    n_samples, d = U.shape
    X_synthetic = np.zeros_like(U)
    for j in range(d):
        sorted_vals = np.sort(X_original[:, j])
        n_orig = len(sorted_vals)
        empirical_cdf = np.arange(1, n_orig + 1) / (n_orig + 1)
        X_synthetic[:, j] = np.interp(U[:, j], empirical_cdf, sorted_vals)
    return X_synthetic


def calibrate_moments(X_synthetic: np.ndarray, X_original: np.ndarray) -> np.ndarray:
    X_calibrated = X_synthetic.copy()
    for j in range(X_synthetic.shape[1]):
        mu_orig = X_original[:, j].mean()
        sigma_orig = X_original[:, j].std()
        mu_syn = X_synthetic[:, j].mean()
        sigma_syn = X_synthetic[:, j].std()
        if sigma_syn > 1e-10:
            X_calibrated[:, j] = mu_orig + (sigma_orig / sigma_syn) * (X_synthetic[:, j] - mu_syn)
        else:
            X_calibrated[:, j] = mu_orig
    return X_calibrated


def create_bins(y: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    bin_edges = np.linspace(y.min(), y.max() + 1e-10, n_bins + 1)
    bin_indices = np.digitize(y, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    return bin_edges, bin_indices


class CopulaCalibGaussianJoint:
    """Gaussian-copula joint augmentation (CopulaCalib fence comparator)."""

    name = "copula_gaussian_joint"
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

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
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
            U = compute_pseudo_observations(X_bin)
            Z = compute_normal_scores(U)
            Sigma = estimate_correlation_matrix(Z)
            U_syn = sample_from_gaussian_copula(Sigma, n_syn, random_state=self.seed + k)
            X_syn = apply_empirical_quantile(U_syn, X_bin)
            X_syn = calibrate_moments(X_syn, X_bin)
            y_syn = np.random.uniform(bin_edges[k], bin_edges[k + 1], size=n_syn)
            syn_X.append(X_syn)
            syn_y.append(y_syn)
        if syn_X:
            X_aug = np.vstack([X, *syn_X])
            y_aug = np.concatenate([y, *syn_y])
        else:
            X_aug, y_aug = X, y
        self.last_X_aug = X_aug.astype(np.float64)
        self.last_y_aug = y_aug.astype(np.float64)
        return self.last_X_aug, self.last_y_aug

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
