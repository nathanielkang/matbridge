#!/usr/bin/env python
"""MaTBridge (MB1) smoke test — 2-epoch synthetic pipeline PASS."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG = ROOT / "configs" / "smoke.yaml"
RESULTS = ROOT / "results" / "pilot_smoke.json"


def main() -> int:
    ok = True
    for path in (ROOT / "configs", ROOT / "results", SRC / "matbridge"):
        if not path.exists():
            print(f"[FAIL] missing: {path.relative_to(ROOT)}")
            ok = False
        else:
            print(f"[PASS] exists: {path.relative_to(ROOT)}")

    try:
        import numpy as np  # noqa: F401
        import yaml  # noqa: F401
        import pandas as pd  # noqa: F401
        from sklearn.neural_network import MLPRegressor  # noqa: F401

        print("[PASS] imports numpy, yaml, pandas, sklearn")
    except ImportError as exc:
        print(f"[FAIL] missing dependency: {exc}")
        return 1

    from matbridge.data.datasets import ALL_DATASETS, get_dataset_list

    n_loaders = len(get_dataset_list())
    if n_loaders != 12:
        print(f"[FAIL] expected 12 dataset loaders, got {n_loaders}")
        ok = False
    else:
        print(f"[PASS] 12 dataset loaders registered: {ALL_DATASETS}")

    from matbridge.baselines import BASELINE_REGISTRY

    required_baselines = {
        "erm", "fds", "denseloss",
        "copula_gaussian_joint", "copula_vine_joint",
        "bridge_off", "tabddpm_aug", "tabdiff_aug", "smoter",
    }
    missing = required_baselines - set(BASELINE_REGISTRY)
    if missing:
        print(f"[FAIL] missing baselines: {sorted(missing)}")
        ok = False
    else:
        print(f"[PASS] baselines registered: {sorted(BASELINE_REGISTRY)}")

    if not CONFIG.exists():
        print(f"[FAIL] smoke config missing: {CONFIG.relative_to(ROOT)}")
        return 1

    from matbridge.train.smoke_runner import run_smoke

    print("[INFO] running 2-epoch synthetic smoke pipeline...")
    result = run_smoke(CONFIG, RESULTS)
    print(f"[INFO] wrote {RESULTS.relative_to(ROOT)}")
    for key in (
        "dataset", "icr_hard", "w1_frozen", "w1_tabddpm",
        "tail_mae", "copula_tail_mae", "coupling_gap", "matched_eta", "status",
    ):
        print(f"  {key}: {result.get(key)}")

    if result.get("status") != "PASS":
        print("[FAIL] smoke pipeline did not PASS")
        ok = False
    else:
        print("[PASS] smoke pipeline PASS")

    if result.get("icr_hard", 1) != 0:
        print("[WARN] icr_hard != 0 (expected 0 after Pi_vert)")
    if not result.get("matched_eta", False):
        print("[FAIL] matched_eta parity broken")
        ok = False
    else:
        print("[PASS] matched_eta parity OK")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
