"""Frozen W1 transport metric stub."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def w1_frozen(z_samples: np.ndarray, z_tail_holdout: np.ndarray) -> float:
    """1D Wasserstein proxy: mean per-dimension W1 averaged (scaffold)."""
    if len(z_samples) == 0 or len(z_tail_holdout) == 0:
        return float("inf")
    dims = min(z_samples.shape[1], z_tail_holdout.shape[1])
    vals = [
        wasserstein_distance(z_samples[:, j], z_tail_holdout[:, j])
        for j in range(dims)
    ]
    return float(np.mean(vals))
