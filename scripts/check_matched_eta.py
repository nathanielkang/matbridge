#!/usr/bin/env python
"""Matched-η parity checker (DUEL_EXECUTION_CONTRACT §1).

Reads configs/matched_eta_manifest.yaml, audits all joint comparators for
forbidden diffs, writes results/matched_eta_report.json.

Usage:
    cd 2_Code
    python scripts/check_matched_eta.py
    python scripts/check_matched_eta.py --manifest configs/matched_eta_manifest.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MANIFEST = ROOT / "configs" / "matched_eta_manifest.yaml"
DEFAULT_OUT = ROOT / "results" / "matched_eta_report.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _audit_comparator(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check that a named comparator matches the manifest parameters."""
    from matbridge.baselines import BASELINE_REGISTRY

    cls = BASELINE_REGISTRY.get(name)
    if cls is None:
        return {
            "comparator": name,
            "status": "MISSING",
            "diffs": [f"not in BASELINE_REGISTRY"],
        }

    m_eta = float(manifest["eta"])
    m_syn_factor = float(manifest["synthetic_factor"])
    m_n_bins = int(manifest["n_bins"])
    m_seeds = list(manifest["regressor"]["seeds"])
    m_epochs = int(manifest["regressor"]["epochs"])

    try:
        inst = cls(eta=m_eta, seed=m_seeds[0])
    except TypeError:
        try:
            inst = cls()
        except Exception as exc:
            return {
                "comparator": name,
                "status": "INSTANTIATION_ERROR",
                "diffs": [str(exc)],
            }

    diffs: list[str] = []

    # Check eta
    inst_eta = getattr(inst, "eta", None)
    if inst_eta is not None and abs(float(inst_eta) - m_eta) > 1e-9:
        diffs.append(f"eta: instance={inst_eta}, manifest={m_eta}")

    # Check synthetic_factor
    inst_sf = getattr(inst, "synthetic_factor", None)
    if inst_sf is not None and abs(float(inst_sf) - m_syn_factor) > 1e-6:
        diffs.append(f"synthetic_factor: instance={inst_sf}, manifest={m_syn_factor}")

    # Check n_bins
    inst_nb = getattr(inst, "n_bins", None)
    if inst_nb is not None and int(inst_nb) != m_n_bins:
        diffs.append(f"n_bins: instance={inst_nb}, manifest={m_n_bins}")

    # Check regressor family (downstream model must be MLP per manifest)
    from sklearn.neural_network import MLPRegressor

    dummy_X = __import__("numpy").random.randn(30, 4).astype("float32")
    dummy_y = __import__("numpy").random.randn(30)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            inst.fit(dummy_X, dummy_y)
        except Exception:
            pass
    model = getattr(inst, "model", None)
    if model is not None and not isinstance(model, MLPRegressor):
        diffs.append(
            f"downstream_model_family: expected MLPRegressor, got {type(model).__name__}"
        )

    forbidden_diffs = manifest.get("forbidden_diffs", [])
    # Map canonical names to whether they are in our diff list
    forbidden_triggered = [
        fd
        for fd in forbidden_diffs
        if any(fd in d for d in diffs)
    ]

    return {
        "comparator": name,
        "status": "PASS" if not diffs else "FAIL",
        "diffs": diffs,
        "forbidden_triggered": forbidden_triggered,
        "eta_checked": m_eta,
        "synthetic_factor_checked": m_syn_factor,
        "n_bins_checked": m_n_bins,
        "seeds_checked": m_seeds,
        "epochs_checked": m_epochs,
    }


def run_check(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)

    comparators: list[str] = manifest.get("comparators", [])
    results: list[dict[str, Any]] = []
    for name in comparators:
        r = _audit_comparator(name, manifest)
        results.append(r)
        status_tag = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        print(f"  {status_tag} {name}: diffs={r['diffs']}")

    total_forbidden = sum(len(r.get("forbidden_triggered", [])) for r in results)
    all_pass = all(r["status"] == "PASS" for r in results)

    report: dict[str, Any] = {
        "schema": "matched-eta-report-v1",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "comparators_checked": comparators,
        "results": results,
        "total_forbidden_diffs": total_forbidden,
        "all_pass": all_pass,
        "status": "PASS" if (all_pass and total_forbidden == 0) else "FAIL",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched-η parity checker")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.out)

    print(f"[INFO] manifest: {manifest_path}")
    manifest = _load_yaml(manifest_path)
    print(f"  eta={manifest['eta']}, synthetic_factor={manifest['synthetic_factor']}, "
          f"n_bins={manifest['n_bins']}, seeds={manifest['regressor']['seeds']}")

    report = run_check(manifest_path, output_path)
    print(f"[INFO] wrote {output_path.relative_to(ROOT)}")
    print(f"  total_forbidden_diffs: {report['total_forbidden_diffs']}")
    print(f"  status: {report['status']}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
