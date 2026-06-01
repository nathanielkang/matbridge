"""Projection Pi_vert — nearest valid one-hot per categorical block (stub)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionResult:
    z_proj: np.ndarray
    delta_pre: float
    icr_hard: int
    soft_violations: float


def project_to_vertices(
    z: np.ndarray,
    *,
    cat_block_size: int = 0,
    n_categories: int = 0,
) -> ProjectionResult:
    """Stub Pi_vert: clip continuous dims; round categorical block to one-hot."""
    z_proj = z.copy()
    if cat_block_size > 0 and n_categories > 0:
        start = max(z.shape[1] - cat_block_size, 0)
        block = z[:, start:]
        # Nearest one-hot per row (stub for mixed MDM reference).
        idx = np.argmax(block, axis=1)
        one_hot = np.zeros_like(block)
        one_hot[np.arange(len(block)), idx] = 1.0
        z_proj[:, start:] = one_hot
        soft = float(np.mean(np.abs(block - one_hot)))
        icr = 0  # hard ICR = 0 after projection by construction (Remark)
    else:
        soft = 0.0
        icr = 0

    delta_pre = float(np.mean(np.linalg.norm(z - z_proj, axis=1)))
    return ProjectionResult(
        z_proj=z_proj.astype(np.float64),
        delta_pre=delta_pre,
        icr_hard=icr,
        soft_violations=soft,
    )
