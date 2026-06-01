"""Stratum-bridge row synthesis in (x,y) space."""

from __future__ import annotations

import numpy as np

from matbridge.bridge.smsb import fit_stratum_labels


def synthesize_stratum_bridge_rows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_syn: int,
    seed: int,
    tail_bias: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-to-tail transport: extrapolate toward tail stratum in (x,y) space."""
    del tail_bias
    rng = np.random.default_rng(seed)
    strata = fit_stratum_labels(y)
    head_idx = np.where(strata.head)[0]
    tail_idx = np.where(strata.tail)[0]
    if len(tail_idx) < 2:
        tail_idx = np.argsort(y)[-max(len(y) // 4, 4) :]
    if len(head_idx) < 2:
        head_idx = np.argsort(y)[: max(len(y) // 4, 4)]

    y_tail_q = float(np.quantile(y, 0.80))
    y_std = float(np.std(y) + 1e-8)
    y_lo = float(np.quantile(y, 0.02))
    y_hi = float(np.quantile(y, 0.995))
    x_lo = np.quantile(X, 0.02, axis=0)
    x_hi = np.quantile(X, 0.98, axis=0)
    syn_x, syn_y = [], []

    n_jitter = max(n_syn // 2, 1)
    for _ in range(n_jitter):
        it = int(rng.choice(tail_idx))
        x_s = X[it] + rng.normal(0.0, 0.015, size=X.shape[1])
        y_s = float(y[it] + rng.normal(0.0, 0.02 * y_std))
        x_s = np.clip(x_s, x_lo, x_hi)
        y_s = float(np.clip(y_s, y_lo, y_hi))
        syn_x.append(x_s)
        syn_y.append(y_s)

    n_extra = max(n_syn - n_jitter, 0)
    for _ in range(n_extra):
        ih = int(rng.choice(head_idx))
        it = int(rng.choice(tail_idx))
        alpha = float(rng.uniform(1.02, 1.12))
        x_s = X[it] + (alpha - 1.0) * (X[it] - X[ih])
        y_s = float(y[it] + (alpha - 1.0) * (y[it] - y[ih]))
        x_s = np.clip(x_s, x_lo, x_hi)
        y_s = float(np.clip(y_s, y_lo, y_hi))
        syn_x.append(x_s)
        syn_y.append(y_s)

    X_out = np.asarray(syn_x, dtype=np.float64)
    y_out = np.asarray(syn_y, dtype=np.float64).ravel()
    keep = y_out >= y_tail_q
    if keep.sum() < max(n_syn // 4, 1):
        keep = np.ones(len(y_out), dtype=bool)
    return X_out[keep], y_out[keep]
