"""Bridge-off ablation — eta=0 (no bridge augmentation)."""

from __future__ import annotations

from matbridge.baselines.erm import ERMBaseline


class BridgeOffBaseline(ERMBaseline):
    name = "bridge_off"
    eta = 0.0
