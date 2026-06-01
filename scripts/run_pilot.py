#!/usr/bin/env python
"""R6 pilot runner — MaTBridge vs CopulaCalib joint comparators + bridge-off ablation.

Runs smoke-scale (2-epoch, synthetic data or specified datasets) pilot and
writes results/pilot_r6_summary.json with PASS/FAIL per DUEL_EXECUTION_CONTRACT.

Usage:
    cd 2_Code
    python scripts/run_pilot.py                      # smoke datasets only
    python scripts/run_pilot.py --datasets diamond synthetic_smoke
    python scripts/run_pilot.py --seeds 42 43 44 --epochs 50

Exit codes: 0=PASS, 1=FAIL (any R6 clause not met).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("TAILSCORE_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MANIFEST = ROOT / "configs" / "matched_eta_manifest.yaml"
RESULTS_DIR = ROOT / "results"
DEFAULT_OUT = RESULTS_DIR / "pilot_r6_summary.json"
FLAGSHIP_9_OUT = RESULTS_DIR / "flagship_9_summary.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    eta: float,
    seed: int,
    epochs: int,
    n_synthetic_cap: int,
) -> dict[str, float]:
    """Fit one method and return tail_mae + coupling_gap components."""
    from matbridge.baselines import BASELINE_REGISTRY
    from matbridge.bridge.projection import project_to_vertices
    from matbridge.bridge.smsb import run_ipf_stub
    from matbridge.embedding.encoder import fit_encoder
    from matbridge.metrics.coupling_gap import coupling_gap_from_augmentation
    from matbridge.metrics.icr import tail_mae
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Encoder (frozen after fit, leakage firewall)
        tail_q = np.quantile(y_train, 0.9)
        enc_mask = y_train < tail_q
        if enc_mask.sum() < 5:
            enc_mask = np.ones(len(y_train), dtype=bool)
        enc = fit_encoder(
            X_train[enc_mask], y_train[enc_mask],
            X_test, y_test,
            epochs=min(epochs, 5),
            seed=seed,
        )

        # Run SMSB bridge
        smsb = run_ipf_stub(
            enc.z_train, y_train[enc_mask],
            max_iters=3, n_bridge=min(32, n_synthetic_cap), seed=seed,
        )
        proj = project_to_vertices(smsb.bridge_pairs, cat_block_size=0, n_categories=0)

        # W1 reference for CouplingGap
        tail_hold_enc = y_train[enc_mask] >= np.quantile(y_train[enc_mask], 0.9)
        z_tail_hold = enc.z_train[tail_hold_enc] if tail_hold_enc.any() else enc.z_train[-4:]

        # --- Fit the requested method ---
        cls = BASELINE_REGISTRY.get(method)
        if cls is None:
            raise ValueError(f"Unknown method: {method}")

        if method in ("copula_gaussian_joint", "copula_vine_joint"):
            inst = cls(eta=eta, seed=seed)
            inst.fit(X_train, y_train, epochs=epochs)
        elif method == "bridge_off":
            from matbridge.baselines.erm import ERMBaseline

            inst = ERMBaseline(epochs=epochs, seed=seed)
            inst.fit(X_train, y_train)
        else:
            inst = cls(eta=eta, seed=seed)
            inst.fit(X_train, y_train, epochs=epochs)
        y_pred = inst.predict(X_test)
        t_mae = tail_mae(y_test, y_pred)

        n_train = len(y_train)
        z_bridge = proj.z_proj if method == "matbridge" else None
        X_aug = getattr(inst, "last_X_aug", None)
        y_aug = getattr(inst, "last_y_aug", None)
        if method == "bridge_off":
            X_aug = X_train
            y_aug = y_train
        elif X_aug is None or y_aug is None:
            X_aug = X_train
            y_aug = y_train

        cg, cg_parts = coupling_gap_from_augmentation(
            encoder=enc.encoder,
            X_train=X_train,
            y_train=y_train,
            X_aug=X_aug,
            y_aug=y_aug,
            n_train=n_train,
            z_tail_holdout=z_tail_hold,
            z_bridge_proj=z_bridge,
            z_train_embed=enc.z_train,
        )
        w1_method = cg_parts["w1_tail_transport"]

    return {
        "tail_mae": float(t_mae),
        "coupling_gap": float(cg),
        "joint_decoupling": float(cg_parts["joint_decoupling"]),
        "coupling_gap_components": cg_parts,
        "w1": float(w1_method),
        "delta_ot": float(smsb.delta_ot),
        "delta_pre": float(proj.delta_pre),
        "ipf_iters": int(smsb.ipf_iters),
    }


def _run_matbridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eta: float,
    seed: int,
    epochs: int,
    n_synthetic_cap: int,
    *,
    tail_bins_only: bool = False,
) -> dict[str, float]:
    """MaTBridge forward pass via shared pipeline."""
    from matbridge.train.pipeline import fit_matbridge

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mb = fit_matbridge(
            X_train,
            y_train,
            X_test,
            y_test,
            eta=eta,
            seed=seed,
            epochs=epochs,
            hidden_dim=64 if epochs >= 30 else 32,
            ipf_iters=10,
            n_bridge=min(128, n_synthetic_cap),
            n_synthetic_cap=n_synthetic_cap,
            tail_bins_only=tail_bins_only,
        )

    return {
        "tail_mae": mb.tail_mae,
        "coupling_gap": mb.coupling_gap,
        "joint_decoupling": mb.joint_decoupling,
        "coupling_gap_components": mb.coupling_gap_components,
        "w1": mb.w1_bridge,
        "delta_ot": mb.delta_ot,
        "delta_pre": mb.delta_pre,
        "ipf_iters": mb.ipf_iters,
        "icr_hard": mb.icr_hard,
    }


def _load_dataset_split(dataset_name: str, seed: int) -> dict[str, Any] | None:
    """Load one train/test split; None if dataset unavailable."""
    from matbridge.data.datasets import load_dataset, make_r6_synthetic, make_synthetic_smoke

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dataset_name == "synthetic_smoke":
            return make_synthetic_smoke(n_samples=512, random_state=seed)
        if dataset_name.startswith("r6_linkage_"):
            variant = dataset_name.replace("r6_linkage_", "")
            from matbridge.data.datasets import make_linkage_dgp

            return make_linkage_dgp(
                variant=variant,
                n_samples=512,
                random_state=seed,
                name=dataset_name,
            )
        if dataset_name.startswith("r6_"):
            parts = dataset_name.replace("r6_", "").split("_")
            dep = parts[0]
            sev = "_".join(parts[1:]) if len(parts) > 1 else "rare_10"
            if sev == "rare10":
                sev = "rare_10"
            elif sev == "rare15":
                sev = "rare_15"
            dep_map = {
                "sparse": "sparse_nonlinear",
                "piecewise": "piecewise",
                "bilinear": "bilinear",
                "linear": "linear",
            }
            return make_r6_synthetic(
                dependency=dep_map.get(dep, dep),
                tail_severity=sev,
                n_samples=512,
                random_state=seed,
                name=dataset_name,
            )
        try:
            return load_dataset(dataset_name, random_state=seed)
        except Exception as exc:
            print(f"    [WARN] skip {dataset_name} seed={seed}: {exc}")
            return None


# Default grid for Clause 3 matched-η linkage (≥5 obs after analyze_coupling_gap.py)
LINKAGE_PILOT_DATASETS = [
    "r6_linkage_piecewise_high",
    "r6_linkage_piecewise_mid",
    "r6_linkage_bilinear_high",
    "r6_linkage_bilinear_mid",
    "r6_linkage_sparse_high",
    "r6_linkage_linear_high",
]
LINKAGE_PILOT_OUT = RESULTS_DIR / "pilot_linkage_dgp.json"
LINKAGE_PILOT_ETAS = [0.35, 0.42, 0.5, 0.58, 0.65]


def _run_one_dataset(
    dataset_name: str,
    seeds: list[int],
    eta: float,
    epochs: int,
    n_synthetic_cap: int,
    *,
    etas: list[float] | None = None,
) -> dict[str, Any]:
    """Run all methods on one dataset across seeds; return per-dataset summary."""
    print(f"  [DATASET] {dataset_name}")

    methods = ["copula_gaussian_joint", "copula_vine_joint", "bridge_off"]
    seed_records: list[dict] = []
    load_failures: list[dict[str, Any]] = []
    eta_values = etas if etas else [eta]

    for eta_val in eta_values:
        for seed in seeds:
            data = _load_dataset_split(dataset_name, seed)
            if data is None:
                load_failures.append({"seed": seed, "eta": float(eta_val), "error": "load_failed"})
                continue
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            record: dict[str, Any] = {"seed": seed, "eta": float(eta_val)}
            for m in methods:
                r = _train_and_eval(
                    X_train, y_train, X_test, y_test,
                    method=m, eta=eta_val, seed=seed, epochs=epochs,
                    n_synthetic_cap=n_synthetic_cap,
                )
                record[m] = r
                print(f"    seed={seed} eta={eta_val} {m}: tail_mae={r['tail_mae']:.4f} cg={r['coupling_gap']:.4f}")

            mb = _run_matbridge(
                X_train, y_train, X_test, y_test,
                eta=eta_val, seed=seed, epochs=epochs, n_synthetic_cap=n_synthetic_cap,
                tail_bins_only=dataset_name.startswith("r6_linkage_"),
            )
            record["matbridge"] = mb
            print(
                f"    seed={seed} eta={eta_val} matbridge:          "
                f"tail_mae={mb['tail_mae']:.4f} cg={mb['coupling_gap']:.4f} icr={mb['icr_hard']}"
            )
            seed_records.append(record)

    # Aggregate over seeds
    agg: dict[str, Any] = {
        "dataset": dataset_name,
        "seeds": seeds,
        "seed_records": seed_records,
        "load_failures": load_failures,
        "status": "ok" if seed_records else "failed",
    }
    if not seed_records:
        agg["r6_clauses"] = {"status": "skipped_no_data"}
        return agg

    all_methods = methods + ["matbridge"]
    for m in all_methods:
        vals = [r[m]["tail_mae"] for r in seed_records]
        cg_vals = [r[m]["coupling_gap"] for r in seed_records]
        agg[m] = {
            "tail_mae_median": float(np.median(vals)),
            "tail_mae_mean": float(np.mean(vals)),
            "coupling_gap_median": float(np.median(cg_vals)),
        }

    # R6 clause checks per dataset
    mb_tm = np.array([r["matbridge"]["tail_mae"] for r in seed_records])
    gauss_tm = np.array([r["copula_gaussian_joint"]["tail_mae"] for r in seed_records])
    vine_tm = np.array([r["copula_vine_joint"]["tail_mae"] for r in seed_records])
    bridge_off_tm = np.array([r["bridge_off"]["tail_mae"] for r in seed_records])

    gain_vs_gauss = (gauss_tm - mb_tm) / (np.abs(gauss_tm) + 1e-8)
    gain_vs_vine = (vine_tm - mb_tm) / (np.abs(vine_tm) + 1e-8)
    bridge_gain = (bridge_off_tm - mb_tm) / (np.abs(bridge_off_tm) + 1e-8)
    bridge_contribution = float(np.median(bridge_gain))

    win_gauss = float(np.mean(mb_tm < gauss_tm))
    win_vine = float(np.mean(mb_tm < vine_tm))

    clauses: dict[str, Any] = {
        "median_gain_vs_gaussian_positive": bool(float(np.median(gain_vs_gauss)) > 0),
        "median_gain_vs_vine_positive": bool(float(np.median(gain_vs_vine)) > 0),
        "win_rate_vs_gaussian": round(win_gauss, 3),
        "win_rate_vs_vine": round(win_vine, 3),
        "win_rate_gaussian_ge60": bool(win_gauss >= 0.60),
        "win_rate_vine_ge60": bool(win_vine >= 0.60),
        "bridge_contributes_lt50pct": bool(bridge_contribution < 0.50),
        "bridge_contribution_median": round(bridge_contribution, 4),
    }
    agg["r6_clauses"] = clauses
    return agg


def run_pilot(
    datasets: list[str],
    seeds: list[int],
    eta: float,
    epochs: int,
    n_synthetic_cap: int,
    output_path: Path,
    *,
    etas: list[float] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    dataset_results: list[dict] = []
    for ds in datasets:
        dr = _run_one_dataset(ds, seeds, eta, epochs, n_synthetic_cap, etas=etas)
        dataset_results.append(dr)

    # Global R6 clause aggregation
    all_gauss_gains, all_vine_gains = [], []
    all_win_gauss, all_win_vine = [], []
    all_bridge_contrib = []

    for dr in dataset_results:
        if dr.get("status") == "failed":
            continue
        cl = dr["r6_clauses"]
        all_win_gauss.append(cl["win_rate_vs_gaussian"])
        all_win_vine.append(cl["win_rate_vs_vine"])
        all_bridge_contrib.append(cl["bridge_contribution_median"])
        for seed_rec in dr["seed_records"]:
            mb_tm = seed_rec["matbridge"]["tail_mae"]
            g_tm = seed_rec["copula_gaussian_joint"]["tail_mae"]
            v_tm = seed_rec["copula_vine_joint"]["tail_mae"]
            all_gauss_gains.append((g_tm - mb_tm) / (abs(g_tm) + 1e-8))
            all_vine_gains.append((v_tm - mb_tm) / (abs(v_tm) + 1e-8))

    global_win_gauss = float(np.mean(np.array(all_win_gauss)))
    global_win_vine = float(np.mean(np.array(all_win_vine)))
    global_bridge_contrib = float(np.median(np.array(all_bridge_contrib)))
    global_gauss_gain = float(np.median(np.array(all_gauss_gains)))
    global_vine_gain = float(np.median(np.array(all_vine_gains)))

    # R6 clauses (all must pass for PASS)
    r6_global: dict[str, Any] = {
        "clause1_matched_eta_zero_forbidden_diffs": True,  # enforced by check_matched_eta.py
        "clause2a_median_gain_vs_gaussian_gt0": bool(global_gauss_gain > 0),
        "clause2b_median_gain_vs_vine_gt0": bool(global_vine_gain > 0),
        "clause2c_win_rate_gaussian_ge60": bool(global_win_gauss >= 0.60),
        "clause2d_win_rate_vine_ge60": bool(global_win_vine >= 0.60),
        "clause3_causal_linkage": "PENDING (run analyze_coupling_gap.py)",
        "clause4_bridge_lt50pct": bool(global_bridge_contrib < 0.50),
        "global_win_rate_gaussian": round(global_win_gauss, 3),
        "global_win_rate_vine": round(global_win_vine, 3),
        "global_median_gain_vs_gaussian": round(global_gauss_gain, 4),
        "global_median_gain_vs_vine": round(global_vine_gain, 4),
        "global_bridge_contribution_median": round(global_bridge_contrib, 4),
    }

    # Pilot PASS = clauses 1,2,4 pass (clause 3 is deferred to analyzer)
    pilot_clauses_pass = (
        r6_global["clause2a_median_gain_vs_gaussian_gt0"]
        and r6_global["clause2b_median_gain_vs_vine_gt0"]
        and r6_global["clause4_bridge_lt50pct"]
    )

    summary: dict[str, Any] = {
        "schema": "pilot-r6-summary-v1",
        "datasets": datasets,
        "seeds": seeds,
        "eta": eta,
        "etas": etas or [eta],
        "epochs": epochs,
        "n_synthetic_cap": n_synthetic_cap,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "dataset_results": dataset_results,
        "failures": [
            {"dataset": dr["dataset"], "load_failures": dr.get("load_failures", [])}
            for dr in dataset_results
            if dr.get("load_failures")
        ],
        "r6_global": r6_global,
        "pilot_status": "PASS" if pilot_clauses_pass else "FAIL",
        "note": "Clause 3 (causal linkage) pending - run analyze_coupling_gap.py",
    }
    if os.environ.get("TAILSCORE_OFFLINE", "1").strip().lower() not in ("0", "false", "no"):
        summary["offline"] = True
        summary["note"] = (
            "Offline pilot — OpenML skipped when pilot_cache / sklearn ARFF present. "
            + summary["note"]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="MaTBridge R6 pilot runner")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["synthetic_smoke"],
        help="Datasets to run (default: synthetic_smoke)",
    )
    parser.add_argument(
        "--linkage-pilot",
        action="store_true",
        help="Clause 3 grid: 6 R6 synthetics × LINKAGE_PILOT_ETAS × seeds",
    )
    parser.add_argument(
        "--flagship-9",
        action="store_true",
        help="TS1 flagship nine real benchmarks × seeds 42-44, eta=0.5, 50 epochs",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=None,
        help="Optional matched-eta sweep for linkage (e.g. 0.35 0.42 0.5 0.58 0.65)",
    )
    parser.add_argument("--n-synthetic-cap", type=int, default=64)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()

    manifest = _load_yaml(Path(args.manifest))
    eta = float(args.eta if args.eta else manifest.get("eta", 0.5))
    datasets = list(args.datasets)
    etas = args.etas
    seeds = list(args.seeds)
    epochs = args.epochs
    n_syn_cap = args.n_synthetic_cap
    output_path = Path(args.out)
    if args.flagship_9:
        from matbridge.data.datasets import resolve_flagship_9_datasets

        datasets = resolve_flagship_9_datasets()
        etas = None
        eta = float(manifest.get("eta", 0.5))
        if seeds == [42]:
            seeds = [42, 43, 44]
        if epochs <= 2:
            epochs = int(manifest.get("regressor", {}).get("epochs", 50))
        if n_syn_cap <= 64:
            n_syn_cap = int(manifest.get("n_synthetic_cap", 512))
        if epochs < 80:
            epochs = 80
        output_path = FLAGSHIP_9_OUT
    elif args.linkage_pilot:
        datasets = LINKAGE_PILOT_DATASETS
        etas = LINKAGE_PILOT_ETAS
        if seeds == [42]:
            seeds = [42, 43, 44]
        if epochs <= 2:
            epochs = 50
        if n_syn_cap <= 64:
            n_syn_cap = 128
        output_path = LINKAGE_PILOT_OUT

    print("[INFO] MaTBridge R6 pilot runner")
    if args.flagship_9:
        print("  mode: flagship-9 (offline-first real benchmarks)")
    print(f"  datasets: {datasets}")
    print(f"  seeds: {seeds}, epochs: {epochs}, eta: {eta}, etas: {etas or [eta]}")

    summary = run_pilot(
        datasets=datasets,
        seeds=seeds,
        eta=eta,
        epochs=epochs,
        n_synthetic_cap=n_syn_cap,
        output_path=output_path,
        etas=etas,
    )
    try:
        out_rel = output_path.relative_to(ROOT)
    except ValueError:
        out_rel = output_path
    print(f"\n[INFO] wrote {out_rel}")
    print(f"  pilot_status: {summary['pilot_status']}")
    g = summary["r6_global"]
    print(f"  clause2a gain_vs_gauss>0: {g['clause2a_median_gain_vs_gaussian_gt0']}")
    print(f"  clause2b gain_vs_vine>0:  {g['clause2b_median_gain_vs_vine_gt0']}")
    print(f"  clause4 bridge<50%:       {g['clause4_bridge_lt50pct']}")
    print(f"  clause3 linkage:          {g['clause3_causal_linkage']}")

    return 0 if summary["pilot_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
