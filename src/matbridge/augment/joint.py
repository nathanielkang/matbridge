"""Bin-conditional MaTBridge augmentation (matched-eta parity with CopulaCalib)."""

from __future__ import annotations

import numpy as np

from matbridge.baselines.copula_gaussian import create_bins


def _resample_bin_rows(
    X_bin: np.ndarray,
    y_bin: np.ndarray,
    n_syn: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample intact (x, y) pairs — preserves tail coupling (vs copula uniform-y in bin)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_bin), size=n_syn, replace=True)
    return X_bin[idx].astype(np.float64), y_bin[idx].astype(np.float64).ravel()


def _jitter_bin_rows(
    X_bin: np.ndarray,
    y_bin: np.ndarray,
    n_syn: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Light X jitter on resampled pairs for lower bins only."""
    rng = np.random.default_rng(seed)
    Xs, ys = _resample_bin_rows(X_bin, y_bin, n_syn, seed)
    x_scale = np.std(X_bin, axis=0) + 1e-8
    Xs = Xs + rng.normal(0.0, 0.02, size=Xs.shape) * x_scale
    Xs = calibrate_moments(Xs, X_bin)
    return Xs.astype(np.float64), ys


def matbridge_joint_augment(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    synthetic_factor: float = 2.0,
    eta: float = 0.5,
    seed: int = 42,
    tail_bins_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror CopulaCalib bin deficit protocol; SMSB rows replace copula draws."""
    n_samples = X.shape[0]
    min_samples = max(10, n_samples // (n_bins * 2))
    _, bin_indices = create_bins(y, n_bins)
    syn_X, syn_y = [], []

    for k in range(n_bins):
        mask = bin_indices == k
        n_k = int(mask.sum())
        if n_k < 3:
            continue
        # Optional tail-only synthesis (linkage DGPs); real benchmarks use all deficit bins.
        if tail_bins_only and k < n_bins // 2:
            continue
        deficit = max(0, min_samples - n_k)
        n_syn = int(deficit * synthetic_factor * eta)
        if n_syn <= 0:
            continue

        X_bin = X[mask]
        y_bin = y[mask]

        # Preserve (x,y) pairs in every bin — CopulaCalib draws y uniform-in-bin.
        Xs, ys = _resample_bin_rows(X_bin, y_bin, n_syn, seed=seed + k + 1)

        syn_X.append(Xs)
        syn_y.append(ys)

    if syn_X:
        return (
            np.vstack([X, *syn_X]).astype(np.float64),
            np.concatenate([y, *syn_y]).astype(np.float64),
        )
    return X.astype(np.float64), y.astype(np.float64)
