# MaTBridge — reference implementation

**MaTBridge** augments imbalanced tabular regression by learning a **stratum-marginal Schrödinger bridge (SMSB)** between head and tail target strata in a frozen mixed-type embedding. Bridge samples are projected to valid categorical rows, mixed with real data under a held-out-tail Wasserstein weight, and used to train a standard regressor.

This repository is the **reference implementation** accompanying the MaTBridge manuscript. It contains **source code only**. Datasets are loaded at runtime via OpenML and scikit-learn; running the scripts writes JSON summaries under `results/` on your machine (not tracked in git).

**Repository:** https://github.com/nathanielkang/matbridge

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick verification (smoke test)

```bash
python scripts/smoke_test.py
```

A successful run writes `results/pilot_smoke.json` locally (gitignored) and exits with code 0.

## Reproducing experiments

Benchmarks are loaded at runtime via OpenML and scikit-learn. Hyperparameters are frozen in `configs/`. Full runs write JSON summaries under `results/` on your machine only.

```bash
# Matched-augmentation parity audit
python scripts/check_matched_eta.py

# Pilot and flagship evaluation (see configs/smoke.yaml, matched_eta_manifest.yaml)
python scripts/run_pilot.py
python scripts/run_pilot.py --flagship-9

# Optional linkage diagnostic (requires prior pilot outputs)
python scripts/analyze_coupling_gap.py
```

## Package layout

| Path | Role |
|------|------|
| `src/matbridge/bridge/smsb.py` | Stratum-marginal Schrödinger bridge (IPF) |
| `src/matbridge/bridge/projection.py` | Categorical projection to valid rows |
| `src/matbridge/embedding/encoder.py` | Spectral-normalised embedding (frozen before transport) |
| `src/matbridge/augment/` | Bridge sampling and training-set augmentation |
| `src/matbridge/baselines/` | ERM, FDS, DenseLoss, SMOTER, TabDDPM/TabDiff aug, copula samplers |
| `src/matbridge/metrics/` | Tail MAE, Wasserstein-1, coupling diagnostics |
| `configs/` | Smoke and matched-η manifest YAML |
| `scripts/` | CLI entry points listed above |

## Scope

**Included:** MaTBridge, in-repo baselines, evaluation metrics, and configuration files needed to rerun the experiments described in the paper.

## Citation

If you use this code, please cite the associated MaTBridge manuscript when available.

## License

MIT License. See `LICENSE`.
