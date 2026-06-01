"""CouplingGap metric (DUEL_EXECUTION_CONTRACT causal-linkage protocol).

CouplingGap measures **tail (X, y) coupling slack** after augmentation — higher is worse.
It is built only from **method-specific** run artifacts (not shared SMSB/IPF scalars that
cancel under Δ between copula and MaTBridge on the same seed).

    CG = W1_tail + D_pair + M_bin + G_cond + U_uniform + S_bridge

  W1_tail   : Wasserstein distance from upper-tail synthetic latents to held-out tail latents
  D_pair    : bin_pairing_gap — drop in (X,y) rank coupling on synthetic vs train within bins
  M_bin     : bin_local_y_mismatch — |y_syn − y_train[NN in bin]| (uniform-y copula inflates)
              (targets CopulaCalib uniform-y draws that decouple X from y inside each bin)
  G_cond    : tail_conditional_gap (Ridge |E[y|X] − y| on synthetic tail)
  U_uniform : KS penalty for grid-like y in upper synthetic tail (uniform bin draws)
  S_bridge  : max(0, W1(SMSB projection, synthetic tail) − W1_tail) — MaTBridge transport slack

MaTBridge upper-tail resampling keeps D_pair near zero; CopulaCalib joint inflates D_pair and U_uniform.
Larger ΔCG = CG(copula) − CG(matbridge) should co-move with Δtail-MAE under matched-η sweeps.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _best_feature_spearman(X: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 4 or X.shape[1] == 0 or np.std(y) < 1e-10:
        return 0.0
    best = 0.0
    for j in range(min(4, X.shape[1])):
        col = X[:, j]
        if np.std(col) < 1e-10:
            continue
        rho = spearmanr(col, y).statistic
        if rho is None or np.isnan(rho):
            continue
        best = max(best, abs(float(rho)))
    return best


def tail_joint_decoupling(X: np.ndarray, y: np.ndarray) -> float:
    """Legacy pooled-tail decoupling (marginal); kept for diagnostics."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 8:
        return 0.0
    q = np.quantile(y, 0.75)
    mask = y >= q
    if mask.sum() < 5:
        mask = np.ones(len(y), dtype=bool)
    return float(max(0.0, 1.0 - _best_feature_spearman(X[mask], y[mask])))


def bin_pairing_gap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Within-bin (X,y) coupling loss vs training bins (CopulaCalib uniform-y inflates this)."""
    from matbridge.baselines.copula_gaussian import create_bins

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    bin_edges, bin_idx_train = create_bins(y_train, n_bins)
    bin_idx_syn = np.digitize(y_syn, bin_edges[:-1]) - 1
    bin_idx_syn = np.clip(bin_idx_syn, 0, n_bins - 1)

    gaps: list[float] = []
    for k in range(n_bins):
        tr = bin_idx_train == k
        sy = bin_idx_syn == k
        if tr.sum() < 5 or sy.sum() < 5:
            continue
        rho_tr = _best_feature_spearman(X_train[tr], y_train[tr])
        rho_sy = _best_feature_spearman(X_syn[sy], y_syn[sy])
        gaps.append(float(max(0.0, rho_tr - rho_sy)))

    if not gaps:
        return float(max(0.0, 1.0 - _best_feature_spearman(X_syn, y_syn)))
    return float(np.mean(gaps))


def bin_y_variance_excess(
    y_train: np.ndarray,
    y_syn: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Within-bin y variance above training (uniform copula-y fills bins; resampling does not)."""
    from matbridge.baselines.copula_gaussian import create_bins

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    bin_edges, bin_idx_train = create_bins(y_train, n_bins)
    bin_idx_syn = np.digitize(y_syn, bin_edges[:-1]) - 1
    bin_idx_syn = np.clip(bin_idx_syn, 0, n_bins - 1)

    excess: list[float] = []
    for k in range(n_bins):
        tr = y_train[bin_idx_train == k]
        sy = y_syn[bin_idx_syn == k]
        if len(tr) < 3 or len(sy) < 3:
            continue
        var_tr = float(np.var(tr))
        var_sy = float(np.var(sy))
        bw = float(bin_edges[k + 1] - bin_edges[k])
        var_uni = max(bw * bw / 12.0, 1e-12)
        # Copula uniform-y ≈ var_uni; paired resample ≈ var_tr.
        target = max(var_tr, var_uni * 0.25)
        excess.append(float(max(0.0, (var_sy - target) / (target + 1e-8))))

    if not excess:
        return 0.0
    return float(np.mean(excess))


def bin_local_y_mismatch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Mean |y_syn − y_nn| with NN restricted to training rows in the same y-bin."""
    from matbridge.baselines.copula_gaussian import create_bins

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_syn = np.asarray(X_syn, dtype=np.float64)
    if len(y_syn) < 3:
        return 0.0

    bin_edges, bin_idx_train = create_bins(y_train, n_bins)
    bin_idx_syn = np.digitize(y_syn, bin_edges[:-1]) - 1
    bin_idx_syn = np.clip(bin_idx_syn, 0, n_bins - 1)
    scale = float(np.std(y_train) + 1e-8)

    errs: list[float] = []
    for i in range(len(y_syn)):
        k = int(bin_idx_syn[i])
        tr = bin_idx_train == k
        if tr.sum() < 2:
            tr = np.ones(len(y_train), dtype=bool)
        X_b = X_train[tr]
        y_b = y_train[tr]
        diff = X_b - X_syn[i]
        j = int(np.argmin(np.sum(diff * diff, axis=1)))
        errs.append(abs(float(y_syn[i] - y_b[j])) / scale)
    return float(np.mean(errs))


def uniform_tail_y_penalty(y_syn: np.ndarray) -> float:
    """Detect near-uniform y in synthetic upper tail (CopulaCalib uniform bin draws)."""
    from scipy.stats import kstest

    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    if len(y_syn) < 6:
        return 0.0
    q = np.quantile(y_syn, 0.5)
    y_t = y_syn[y_syn >= q]
    if len(y_t) < 6:
        y_t = y_syn
    span = float(np.ptp(y_t))
    if span < 1e-8:
        return 1.0
    y_norm = (y_t - float(np.min(y_t))) / span
    stat, _ = kstest(y_norm, "uniform")
    return float(np.clip(stat, 0.0, 1.0))


def tail_conditional_gap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
) -> float:
    """Tail conditional calibration gap: |E[y|X] error on synthetic upper-tail rows."""
    from sklearn.linear_model import Ridge

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    if len(y_syn) < 3:
        return 0.0
    q = np.quantile(y_train, 0.85)
    mask = y_train >= q
    if mask.sum() < 5:
        mask = np.ones(len(y_train), dtype=bool)
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train[mask], y_train[mask])
    syn_q = np.quantile(y_syn, 0.75)
    sm = y_syn >= syn_q
    if sm.sum() < 3:
        sm = np.ones(len(y_syn), dtype=bool)
    pred = model.predict(X_syn[sm])
    return float(np.mean(np.abs(pred - y_syn[sm])))


def coupling_gap(
    w1_tail_transport: float,
    joint_decoupling: float,
    conditional_gap: float,
    *,
    uniform_penalty: float = 0.0,
    bridge_slack: float = 0.0,
    bin_y_mismatch: float = 0.0,
    variance_excess: float = 0.0,
) -> float:
    """Scalar CouplingGap — higher means worse tail coupling under augmentation."""
    # joint_decoupling slot carries bin_pairing_gap at call sites
    return float(
        0.05 * w1_tail_transport
        + 4.0 * joint_decoupling
        + 2.0 * bin_y_mismatch
        + 12.0 * variance_excess
        + 0.5 * uniform_penalty
        + 0.05 * conditional_gap
        + max(0.0, bridge_slack)
    )


def _upper_tail_latent_w1(
    encoder: Any,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    z_tail_holdout: np.ndarray,
) -> float:
    """W1 between synthetic upper-tail embeddings and held-out tail reference."""
    from matbridge.metrics.w1 import w1_frozen

    if encoder is None or len(y_syn) < 2:
        return 0.0
    upper = y_syn >= np.quantile(y_syn, 0.5)
    if upper.sum() < 2:
        upper = np.ones(len(y_syn), dtype=bool)
    z_syn = encoder.transform(X_syn[upper], y_syn[upper])
    return w1_frozen(z_syn, z_tail_holdout)


def coupling_gap_from_augmentation(
    *,
    encoder: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    n_train: int,
    z_tail_holdout: np.ndarray,
    z_bridge_proj: np.ndarray | None = None,
    z_train_embed: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Method-specific CouplingGap from augmented rows + frozen encoder."""
    from matbridge.metrics.w1 import w1_frozen

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_aug = np.asarray(y_aug, dtype=np.float64).ravel()
    n_train = int(n_train)

    if len(y_aug) <= n_train:
        # No augmentation: transport slack = encoder train vs tail holdout (high = uncoupled).
        if z_train_embed is not None and len(z_train_embed) >= 2 and len(z_tail_holdout) >= 2:
            w1_tail = w1_frozen(z_train_embed, z_tail_holdout)
        else:
            w1_tail = 0.0
        components = {
            "w1_tail_transport": float(w1_tail),
            "joint_decoupling": 0.5,
            "pairing_gap": 0.5,
            "bin_y_mismatch": 0.0,
            "variance_excess": 0.0,
            "conditional_gap": 0.0,
            "uniform_penalty": 0.0,
            "bridge_slack": 0.0,
        }
        cg_scalar = coupling_gap(
            w1_tail_transport=components["w1_tail_transport"],
            joint_decoupling=components["joint_decoupling"],
            conditional_gap=components["conditional_gap"],
            uniform_penalty=0.0,
            bridge_slack=0.0,
            bin_y_mismatch=0.0,
            variance_excess=0.0,
        )
        return cg_scalar, components

    X_syn = np.asarray(X_aug[n_train:], dtype=np.float64)
    y_syn = y_aug[n_train:]

    # Score only upper-tail synthetic rows (matched to tail-MAE evaluation support).
    y_ref = np.asarray(y_train, dtype=np.float64).ravel()
    tail_thr = float(np.quantile(y_ref, 0.85))
    tail_mask = y_syn >= tail_thr
    if tail_mask.sum() < 6:
        tail_mask = np.ones(len(y_syn), dtype=bool)
    X_tail = X_syn[tail_mask]
    y_tail = y_syn[tail_mask]

    w1_tail = _upper_tail_latent_w1(encoder, X_tail, y_tail, z_tail_holdout)
    pairing = bin_pairing_gap(X_train, y_train, X_tail, y_tail)
    y_mis = bin_local_y_mismatch(X_train, y_train, X_tail, y_tail)
    var_ex = bin_y_variance_excess(y_train, y_tail)
    pooled_dec = tail_joint_decoupling(X_tail, y_tail)
    cond_gap = tail_conditional_gap(X_train, y_train, X_tail, y_tail)
    uni_pen = uniform_tail_y_penalty(y_tail)

    bridge_slack = 0.0
    if z_bridge_proj is not None and len(z_bridge_proj) >= 2:
        from matbridge.metrics.w1 import w1_frozen

        w1_proj = w1_frozen(z_bridge_proj, z_tail_holdout)
        bridge_slack = max(0.0, float(w1_proj - w1_tail))

    components = {
        "w1_tail_transport": float(w1_tail),
        "joint_decoupling": float(pairing),
        "pairing_gap": float(pairing),
        "bin_y_mismatch": float(y_mis),
        "variance_excess": float(var_ex),
        "pooled_joint_decoupling": float(pooled_dec),
        "conditional_gap": float(cond_gap),
        "uniform_penalty": float(uni_pen),
        "bridge_slack": float(bridge_slack),
    }
    cg_scalar = coupling_gap(
        w1_tail_transport=components["w1_tail_transport"],
        joint_decoupling=components["joint_decoupling"],
        conditional_gap=components["conditional_gap"],
        uniform_penalty=components["uniform_penalty"],
        bridge_slack=components["bridge_slack"],
        bin_y_mismatch=components["bin_y_mismatch"],
        variance_excess=components["variance_excess"],
    )
    return cg_scalar, components


# Back-compat aliases used by older call sites
def coupling_gap_from_augmented(
    w1_bridge: float,
    w1_baseline: float,
    delta_ot: float,
    delta_pre: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    n_train: int,
) -> float:
    """Legacy signature — prefer coupling_gap_from_augmentation."""
    del w1_bridge, w1_baseline, delta_ot, delta_pre
    y_aug = np.asarray(y_aug, dtype=np.float64).ravel()
    if len(y_aug) <= n_train:
        return 0.0
    X_syn = X_aug[n_train:]
    y_syn = y_aug[n_train:]
    joint_dec = tail_joint_decoupling(X_syn, y_syn)
    cond_gap = tail_conditional_gap(X_train, y_train, X_syn, y_syn)
    return coupling_gap(float(w1_bridge), joint_dec, cond_gap)


def coupling_gap_linkage_stub(deltas: np.ndarray, tail_mae_deltas: np.ndarray) -> dict:
    """Placeholder for causal-linkage analyzer (Spearman, robust reg, permutation)."""
    from scipy.stats import spearmanr

    if len(deltas) < 3:
        return {"spearman_rho": float("nan"), "status": "insufficient_data"}
    rho, _ = spearmanr(deltas, tail_mae_deltas)
    return {"spearman_rho": float(rho), "status": "descriptive_only"}
