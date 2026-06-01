#!/usr/bin/env python
"""R6 CouplingGap causal-linkage analyzer (DUEL_EXECUTION_CONTRACT §causal-linkage).

Reads results/pilot_r6_summary.json, computes all three required tests:
  1. Spearman ρ ≥ 0.50 between ΔCouplingGap and Δtail-MAE
  2. Robust-regression slope CI for Δtail-MAE on ΔCouplingGap excludes 0
  3. Permutation test p ≤ 0.05 under shuffled CouplingGap deltas

Writes results/coupling_gap_linkage.json with PASS/FAIL per clause.

Usage:
    cd 2_Code
    python scripts/analyze_coupling_gap.py
    python scripts/analyze_coupling_gap.py --pilot results/pilot_r6_summary.json
    python scripts/analyze_coupling_gap.py --n-permutations 999

Exit codes: 0=all 3 tests pass, 1=at least one test fails or insufficient data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_PILOT = ROOT / "results" / "pilot_r6_summary.json"
LINKAGE_PILOT = ROOT / "results" / "pilot_linkage_dgp.json"
DEFAULT_OUT = ROOT / "results" / "coupling_gap_linkage.json"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_deltas(
    pilot: dict[str, Any],
    *,
    mode: str = "pooled",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (ΔCouplingGap, Δtail-MAE) pairs from pilot summary.

    ΔCouplingGap = coupling_gap(copula_gaussian) − coupling_gap(matbridge)
    Δtail-MAE    = tail_mae(copula_gaussian) − tail_mae(matbridge)

    mode=pooled: one pair per (dataset, seed, eta) record (matched-η sweep).
    mode=eta_contrast: Δ at η_hi minus Δ at η_lo within each (dataset, seed) cell.
    """
    if mode == "eta_contrast":
        return _extract_eta_contrast_deltas(pilot)
    delta_cg: list[float] = []
    delta_tmae: list[float] = []

    for dr in pilot.get("dataset_results", []):
        for sr in dr.get("seed_records", []):
            mb = sr.get("matbridge", {})
            gauss = sr.get("copula_gaussian_joint", {})
            mb_cg = mb.get("coupling_gap")
            g_cg = gauss.get("coupling_gap")
            mb_tm = mb.get("tail_mae")
            g_tm = gauss.get("tail_mae")
            if None in (mb_cg, g_cg, mb_tm, g_tm):
                continue
            delta_cg.append(float(g_cg) - float(mb_cg))
            delta_tmae.append(float(g_tm) - float(mb_tm))

    return np.array(delta_cg), np.array(delta_tmae)


def _extract_eta_contrast_deltas(pilot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Matched-η intervention: change in (copula−matbridge) gap from η_lo to η_hi."""
    from collections import defaultdict

    etas = sorted({float(e) for e in pilot.get("etas", [0.35, 0.65])})
    eta_lo = min(etas)
    eta_hi = max(etas)
    groups: dict[tuple[str, int], dict[float, dict]] = defaultdict(dict)

    for dr in pilot.get("dataset_results", []):
        ds = dr.get("dataset", "")
        for sr in dr.get("seed_records", []):
            key = (ds, int(sr["seed"]))
            groups[key][float(sr["eta"])] = sr

    delta_cg: list[float] = []
    delta_tmae: list[float] = []
    for _key, by_eta in groups.items():
        if eta_lo not in by_eta or eta_hi not in by_eta:
            continue
        lo, hi = by_eta[eta_lo], by_eta[eta_hi]

        def pair_delta(rec: dict[str, Any], method_a: str, method_b: str, field: str) -> float:
            return float(rec[method_a][field]) - float(rec[method_b][field])

        dcg_hi = pair_delta(hi, "copula_gaussian_joint", "matbridge", "coupling_gap")
        dcg_lo = pair_delta(lo, "copula_gaussian_joint", "matbridge", "coupling_gap")
        dtm_hi = pair_delta(hi, "copula_gaussian_joint", "matbridge", "tail_mae")
        dtm_lo = pair_delta(lo, "copula_gaussian_joint", "matbridge", "tail_mae")
        delta_cg.append(dcg_hi - dcg_lo)
        delta_tmae.append(dtm_hi - dtm_lo)

    return np.array(delta_cg), np.array(delta_tmae)


def _spearman_test(delta_cg: np.ndarray, delta_tmae: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr

    if len(delta_cg) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "pass": False, "reason": "insufficient_data"}
    rho, p = spearmanr(delta_cg, delta_tmae)
    result = {
        "rho": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "threshold_rho": 0.50,
        "pass": bool(float(rho) >= 0.50),
    }
    return result


def _robust_regression_ci(
    delta_cg: np.ndarray,
    delta_tmae: np.ndarray,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Theil-Sen slope with bootstrap CI (robust to outliers)."""
    from scipy.stats import theilslopes

    if len(delta_cg) < 3:
        return {
            "slope": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_excludes_zero": False,
            "pass": False,
            "reason": "insufficient_data",
        }

    rng = np.random.default_rng(0)
    n = len(delta_cg)
    slopes: list[float] = []
    for _ in range(4999):
        idx = rng.integers(0, n, size=n)
        x, y = delta_cg[idx], delta_tmae[idx]
        if np.std(x) < 1e-12:
            continue
        try:
            res = theilslopes(y, x, alpha=1 - ci_level)
            slopes.append(float(res.slope))
        except Exception:
            slopes.append(float(np.polyfit(x, y, 1)[0]))

    if not slopes:
        return {
            "slope": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_excludes_zero": False,
            "pass": False,
            "reason": "fit_error",
        }

    slopes_arr = np.array(slopes)
    alpha = (1 - ci_level) / 2
    ci_low = float(np.percentile(slopes_arr, 100 * alpha))
    ci_high = float(np.percentile(slopes_arr, 100 * (1 - alpha)))
    slope = float(np.median(slopes_arr))
    ci_excludes_zero = not (ci_low <= 0.0 <= ci_high)
    return {
        "slope": round(slope, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "ci_level": ci_level,
        "ci_excludes_zero": ci_excludes_zero,
        "pass": bool(ci_excludes_zero),
    }


def _permutation_test(
    delta_cg: np.ndarray,
    delta_tmae: np.ndarray,
    n_permutations: int = 999,
    seed: int = 0,
) -> dict[str, Any]:
    """One-sided permutation test: H0 = shuffled CouplingGap has the same Spearman ρ."""
    from scipy.stats import spearmanr

    if len(delta_cg) < 3:
        return {
            "observed_rho": float("nan"),
            "p_value": float("nan"),
            "n_permutations": n_permutations,
            "pass": False,
            "reason": "insufficient_data",
        }

    rng = np.random.default_rng(seed)
    obs_rho = float(spearmanr(delta_cg, delta_tmae).statistic)
    null_rhos = []
    for _ in range(n_permutations):
        perm = rng.permutation(delta_cg)
        null_rho = float(spearmanr(perm, delta_tmae).statistic)
        null_rhos.append(null_rho)

    null_arr = np.array(null_rhos)
    # One-sided: how often null ρ >= observed ρ
    p_value = float(np.mean(null_arr >= obs_rho))

    return {
        "observed_rho": round(obs_rho, 4),
        "null_rho_median": round(float(np.median(null_arr)), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
        "threshold_p": 0.05,
        "pass": bool(p_value <= 0.05),
    }


def run_analysis(
    pilot_path: Path,
    output_path: Path,
    n_permutations: int = 999,
    *,
    extraction_mode: str = "auto",
) -> dict[str, Any]:
    pilot = _load_json(pilot_path)
    mode = extraction_mode
    if mode == "auto":
        # Pooled (dataset × seed × η) is the primary matched-η linkage design.
        mode = "pooled"
    delta_cg, delta_tmae = _extract_deltas(pilot, mode=mode)

    print(f"  Observations extracted: {len(delta_cg)}")
    if len(delta_cg) == 0:
        report = {
            "schema": "coupling-gap-linkage-v1",
            "n_observations": 0,
            "status": "FAIL",
            "reason": "no data extracted from pilot",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    print(f"  delta_cg range: [{delta_cg.min():.4f}, {delta_cg.max():.4f}]")
    print(f"  delta_tmae range: [{delta_tmae.min():.4f}, {delta_tmae.max():.4f}]")

    spearman = _spearman_test(delta_cg, delta_tmae)
    robust_ci = _robust_regression_ci(delta_cg, delta_tmae)
    permutation = _permutation_test(delta_cg, delta_tmae, n_permutations=n_permutations)

    status_tag = "[PASS]" if spearman["pass"] else "[FAIL]"
    print(f"  {status_tag} Spearman rho={spearman['rho']} (threshold >=0.50)")
    status_tag = "[PASS]" if robust_ci["pass"] else "[FAIL]"
    print(f"  {status_tag} Robust CI=[{robust_ci['ci_low']}, {robust_ci['ci_high']}] (excludes 0: {robust_ci['ci_excludes_zero']})")
    status_tag = "[PASS]" if permutation["pass"] else "[FAIL]"
    print(f"  {status_tag} Permutation p={permutation['p_value']} (threshold <=0.05)")

    all_pass = spearman["pass"] and robust_ci["pass"] and permutation["pass"]

    if not all_pass:
        if len(delta_cg) < 5:
            note = ("WARNING: fewer than 5 observations - run more datasets/seeds "
                    "for reliable causal-linkage evidence (Day-5 full run).")
        else:
            note = "CouplingGap is descriptive-only; no novelty credit per DUEL_EXECUTION_CONTRACT."
    else:
        note = "All three causal-linkage tests passed. R6 Clause 3 PASS."

    report: dict[str, Any] = {
        "schema": "coupling-gap-linkage-v1",
        "pilot_source": str(pilot_path.relative_to(ROOT)),
        "extraction_mode": mode,
        "n_observations": int(len(delta_cg)),
        "delta_cg_stats": {
            "mean": round(float(delta_cg.mean()), 4),
            "std": round(float(delta_cg.std()), 4),
            "min": round(float(delta_cg.min()), 4),
            "max": round(float(delta_cg.max()), 4),
        },
        "delta_tmae_stats": {
            "mean": round(float(delta_tmae.mean()), 4),
            "std": round(float(delta_tmae.std()), 4),
        },
        "test1_spearman": spearman,
        "test2_robust_regression_ci": robust_ci,
        "test3_permutation": permutation,
        "all_tests_pass": all_pass,
        "status": "PASS" if all_pass else "FAIL",
        "note": note,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="CouplingGap causal-linkage analyzer")
    parser.add_argument("--pilot", default=None, help="Pilot JSON (default: pilot_r6 or --linkage file)")
    parser.add_argument("--linkage", action="store_true", help="Analyze pilot_linkage_dgp.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument(
        "--extraction-mode",
        choices=("auto", "pooled", "eta_contrast"),
        default="auto",
        help="auto: eta_contrast when pilot has multiple etas, else pooled",
    )
    args = parser.parse_args()

    if args.pilot:
        pilot_path = Path(args.pilot)
    elif args.linkage:
        pilot_path = LINKAGE_PILOT
    else:
        pilot_path = DEFAULT_PILOT
    output_path = Path(args.out)

    if not pilot_path.exists():
        print(f"[ERROR] pilot file not found: {pilot_path}")
        print("  Run `python scripts/run_pilot.py` first.")
        return 1

    print(f"[INFO] CouplingGap causal-linkage analyzer")
    print(f"  pilot: {pilot_path.relative_to(ROOT)}")
    print(f"  n_permutations: {args.n_permutations}")

    report = run_analysis(
        pilot_path,
        output_path,
        n_permutations=args.n_permutations,
        extraction_mode=args.extraction_mode,
    )
    print(f"\n[INFO] wrote {output_path.relative_to(ROOT)}")
    print(f"  status: {report['status']}")
    if "note" in report:
        print(f"  note: {report['note']}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
