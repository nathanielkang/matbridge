"""Stratum-marginal Schrödinger bridge (SMSB) — IPF stub (DSBM-style).

Implements:
  - Stratum labeling (head/tail quantile split)
  - Score-based IPF with explicit plateau stop (patience on rising MMD)
  - Encoder-freeze guard: z must come from a frozen encoder (enforced by callers)
  - Pi_vert projection handled in bridge/projection.py (hard ICR=0 by construction)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Minimum stratum size to run IPF; below this, return degenerate bridge.
_MIN_STRATUM = 2
# Relative MMD improvement threshold for plateau detection (< tol = no improvement).
_PLATEAU_TOL = 1e-4


@dataclass
class StratumLabels:
    head: np.ndarray
    tail: np.ndarray


@dataclass
class SMSBResult:
    bridge_pairs: np.ndarray
    ipf_iters: int
    mmd_history: list[float]
    delta_ot: float
    stopped_early: bool
    plateau_reason: str = ""


def fit_stratum_labels(y: np.ndarray, head_q: float = 0.25, tail_q: float = 0.75) -> StratumLabels:
    """Head = middle quantiles; tail = extremes (PROPOSAL §2).

    head: [Q25, Q75] — dense region used to seed IPF source marginal.
    tail: (<Q25) ∪ (>Q75) — sparse region used as IPF target marginal.
    """
    lo, hi = np.quantile(y, [head_q, tail_q])
    head = (y >= lo) & (y <= hi)
    tail = (y < lo) | (y > hi)
    return StratumLabels(head=head, tail=tail)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None, max_pts: int = 256) -> float:
    """RBF-MMD^2 between two embedding sets (subsampled for large n)."""
    if len(x) < _MIN_STRATUM or len(y) < _MIN_STRATUM:
        return 0.0
    if len(x) > max_pts:
        idx = np.linspace(0, len(x) - 1, max_pts, dtype=int)
        x = x[idx]
    if len(y) > max_pts:
        idx = np.linspace(0, len(y) - 1, max_pts, dtype=int)
        y = y[idx]
    if gamma is None:
        # Median heuristic bandwidth (small subsample only)
        all_pts = np.vstack([x, y])
        if len(all_pts) > max_pts:
            idx = np.linspace(0, len(all_pts) - 1, max_pts, dtype=int)
            all_pts = all_pts[idx]
        dists = np.linalg.norm(all_pts[:, None] - all_pts[None], axis=-1)
        median_d = float(np.median(dists[dists > 0])) if (dists > 0).any() else 1.0
        gamma = 1.0 / (2.0 * median_d ** 2 + 1e-8)

    xx = np.sum(x ** 2, axis=1, keepdims=True)
    yy = np.sum(y ** 2, axis=1, keepdims=True)
    d_xx = np.clip(xx + xx.T - 2 * x @ x.T, 0.0, None)
    d_yy = np.clip(yy + yy.T - 2 * y @ y.T, 0.0, None)
    d_xy = np.clip(xx + yy.T - 2 * x @ y.T, 0.0, None)
    k_xx = np.exp(-gamma * d_xx).mean()
    k_yy = np.exp(-gamma * d_yy).mean()
    k_xy = np.exp(-gamma * d_xy).mean()
    return float(max(k_xx + k_yy - 2 * k_xy, 0.0))


def run_ipf_stub(
    z: np.ndarray,
    y: np.ndarray,
    *,
    max_iters: int = 5,
    n_bridge: int = 64,
    seed: int = 42,
    plateau_patience: int = 2,
) -> SMSBResult:
    """Score-based IPF stub with explicit plateau stop.

    Algorithm (DSBM-style, Assumption B2 in PROPOSAL):
      1. Partition z into head/tail strata from y.
      2. For each IPF step t ∈ [1..max_iters]:
         a. Interpolate mean(head) → mean(tail) with Brownian noise ∝ t.
         b. Compute MMD(bridge, tail) — lower is better.
         c. Stop early if MMD rises for `plateau_patience` consecutive iters
            OR improvement < _PLATEAU_TOL (plateau detected).
      3. Return bridge embedding pairs for downstream Pi_vert projection.

    Encoder-freeze contract: caller MUST freeze encoder before calling this.
    The 'z' embedding is treated as read-only within IPF.
    """
    rng = np.random.default_rng(seed)
    strata = fit_stratum_labels(y)
    z_head = z[strata.head]
    z_tail = z[strata.tail]

    if len(z_head) < _MIN_STRATUM or len(z_tail) < _MIN_STRATUM:
        pairs = rng.standard_normal((n_bridge, z.shape[1]))
        return SMSBResult(
            bridge_pairs=pairs.astype(np.float64),
            ipf_iters=0,
            mmd_history=[0.0],
            delta_ot=0.0,
            stopped_early=True,
            plateau_reason="insufficient_stratum_size",
        )

    mu_h = z_head.mean(axis=0)
    mu_t = z_tail.mean(axis=0)
    delta_ot = float(np.linalg.norm(mu_h - mu_t))

    # Adaptive noise scale: proportional to mean stratum spread
    sigma_h = float(np.mean(np.std(z_head, axis=0)))
    sigma_t = float(np.mean(np.std(z_tail, axis=0)))
    noise_scale = 0.1 * max(sigma_h, sigma_t, 0.01)

    mmd_history: list[float] = []
    rise_count = 0
    iters_done = 0
    bridge = rng.standard_normal((n_bridge, z.shape[1]))  # initial bridge
    plateau_reason = ""

    for it in range(max_iters):
        t = (it + 1) / max(max_iters, 1)
        # Interpolate mean head → mean tail; noise decays as bridge converges
        bridge = (
            (1 - t) * mu_h
            + t * mu_t
            + noise_scale * (1 - t + 0.05) * rng.standard_normal((n_bridge, z.shape[1]))
        )
        mmd = _mmd_rbf(bridge, z_tail)
        mmd_history.append(mmd)
        iters_done = it + 1

        # Plateau: MMD rose or improvement < tolerance
        if len(mmd_history) >= 2:
            delta_mmd = mmd_history[-2] - mmd_history[-1]  # positive = improving
            if delta_mmd < 0:
                rise_count += 1
            elif abs(delta_mmd) < _PLATEAU_TOL * (mmd_history[-2] + 1e-8):
                rise_count += 1  # negligible improvement counts as plateau
            else:
                rise_count = 0

            if rise_count >= plateau_patience:
                plateau_reason = f"plateau_at_iter_{iters_done}"
                break

    return SMSBResult(
        bridge_pairs=bridge.astype(np.float64),
        ipf_iters=iters_done,
        mmd_history=mmd_history,
        delta_ot=delta_ot,
        stopped_early=rise_count >= plateau_patience,
        plateau_reason=plateau_reason,
    )
