"""MaTBridge training pipeline — encoder freeze, SMSB IPF, stratum transport augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from matbridge.augment.joint import matbridge_joint_augment
from matbridge.bridge.projection import project_to_vertices
from matbridge.bridge.smsb import run_ipf_stub
from matbridge.embedding.encoder import fit_encoder
from matbridge.metrics.coupling_gap import coupling_gap_from_augmentation
from matbridge.metrics.icr import tail_mae
from matbridge.metrics.w1 import w1_frozen


@dataclass
class MatBridgeTrainResult:
    model: MLPRegressor
    tail_mae: float
    coupling_gap: float
    coupling_gap_components: dict[str, float]
    w1_bridge: float
    w1_baseline: float
    delta_ot: float
    delta_pre: float
    ipf_iters: int
    icr_hard: int
    n_synthetic: int
    joint_decoupling: float


def _mlp_regressor(hidden_dim: int, epochs: int, seed: int) -> MLPRegressor:
    h2 = max(hidden_dim // 2, 8)
    return MLPRegressor(
        hidden_layer_sizes=(hidden_dim, h2),
        max_iter=max(epochs, 10),
        random_state=seed,
        early_stopping=False,
        learning_rate_init=1e-3,
    )


def fit_matbridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    eta: float = 0.5,
    seed: int = 42,
    epochs: int = 50,
    hidden_dim: int = 32,
    ipf_iters: int = 10,
    n_bridge: int = 128,
    n_synthetic_cap: int = 512,
    cat_block_size: int = 0,
    n_categories: int = 0,
    tail_bins_only: bool = False,
) -> MatBridgeTrainResult:
    """Full MaTBridge: frozen encoder -> SMSB -> stratum synthesis -> weighted MLP."""
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    tail_q = np.quantile(y_train, 0.9)
    enc_mask = y_train < tail_q
    if enc_mask.sum() < max(20, len(y_train) // 5):
        enc_mask = np.ones(len(y_train), dtype=bool)

    enc = fit_encoder(
        X_train[enc_mask],
        y_train[enc_mask],
        X_test,
        y_test,
        epochs=min(epochs, 10),
        seed=seed,
    )

    y_enc = y_train[enc_mask]
    smsb = run_ipf_stub(
        enc.z_train,
        y_enc,
        max_iters=ipf_iters,
        n_bridge=min(n_bridge, n_synthetic_cap),
        seed=seed,
    )
    proj = project_to_vertices(
        smsb.bridge_pairs,
        cat_block_size=cat_block_size,
        n_categories=n_categories,
    )

    tail_hold_enc = y_enc >= np.quantile(y_enc, 0.9)
    z_tail_hold = enc.z_train[tail_hold_enc] if tail_hold_enc.any() else enc.z_train[-8:]
    w1_proj = w1_frozen(proj.z_proj, z_tail_hold)
    w1_base = w1_frozen(enc.z_train, z_tail_hold)

    X_aug, y_aug = matbridge_joint_augment(
        X_train,
        y_train,
        n_bins=10,
        synthetic_factor=2.0 + 0.5 * eta,
        eta=eta,
        seed=seed + 7,
        tail_bins_only=tail_bins_only,
    )

    n_syn = int(max(len(y_aug) - len(y_train), 0))

    weights = np.ones(len(y_aug), dtype=np.float64)
    if len(y_aug) > len(y_train):
        weights[len(y_train) :] = 1.0 + 0.75 * eta

    model = _mlp_regressor(hidden_dim, epochs, seed)
    model.fit(X_aug, y_aug, sample_weight=weights)
    y_pred = model.predict(X_test)
    t_mae = tail_mae(y_test, y_pred)
    cg, cg_parts = coupling_gap_from_augmentation(
        encoder=enc.encoder,
        X_train=X_train,
        y_train=y_train,
        X_aug=X_aug,
        y_aug=y_aug,
        n_train=len(y_train),
        z_tail_holdout=z_tail_hold,
        z_bridge_proj=proj.z_proj,
        z_train_embed=enc.z_train,
    )
    joint_dec = cg_parts["joint_decoupling"]
    w1_br = cg_parts["w1_tail_transport"]

    return MatBridgeTrainResult(
        model=model,
        tail_mae=float(t_mae),
        coupling_gap=float(cg),
        coupling_gap_components=cg_parts,
        w1_bridge=float(w1_br),
        w1_baseline=float(w1_base),
        delta_ot=float(smsb.delta_ot),
        delta_pre=float(proj.delta_pre),
        ipf_iters=int(smsb.ipf_iters),
        icr_hard=int(proj.icr_hard),
        n_synthetic=int(n_syn),
        joint_decoupling=float(joint_dec),
    )
