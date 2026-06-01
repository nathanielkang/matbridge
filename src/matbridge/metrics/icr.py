"""Tail MAE and ICR helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, tail_quantile: float = 0.9) -> float:
    thresh = np.quantile(y_true, tail_quantile)
    mask = y_true >= thresh
    if not mask.any():
        return float(mean_absolute_error(y_true, y_pred))
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))
