"""Tabular regression dataset loaders — same 12-dataset pool as TailScore (TS1).

Charter eight (PROPOSAL) + four OpenML extras for fair cross-line comparison.
Parity source: 26_InfoSci_TailScore/2_Code/src/tailscore/data/datasets.py
"""

from __future__ import annotations

import gzip
import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHARTER_DATASETS = [
    "meps_19",
    "acs_income",
    "brazilian_housing",
    "diamond",
    "house_16h",
    "cpu_small",
    "naval",
    "kin8nm",
]

OPENML_EXTRA_DATASETS = [
    "abalone",
    "space_ga",
    "yacht",
    "california_housing",
]

ALL_DATASETS = CHARTER_DATASETS + OPENML_EXTRA_DATASETS

OPENML_IDS: dict[str, list[int]] = {
    "abalone": [183],
    "diamond": [1197],
    "house_16h": [574, 41021],
    "cpu_small": [197, 573],
    "naval": [151],
    "kin8nm": [189],
    "space_ga": [201],
    "yacht": [198],
}

# TS1 flagship nine (charter pool minus kin8nm diagnostic).
FLAGSHIP_9_DATASETS = [
    "meps_19",
    "acs_income",
    "brazilian_housing",
    "diamond",
    "house_16h",
    "cpu_small",
    "naval",
    "abalone",
    "space_ga",
]

_CODE_ROOT = Path(__file__).resolve().parents[3]
_PILOT_CACHE_DIRS = [
    _CODE_ROOT / "data" / "pilot_cache",
    Path(__file__).resolve().parents[5]
    / "26_InfoSci_TailScore"
    / "2_Code"
    / "data"
    / "pilot_cache",
]
_TS1_SELECTION_JSON = (
    Path(__file__).resolve().parents[5]
    / "26_InfoSci_TailScore"
    / "2_Code"
    / "results"
    / "dataset_selection.json"
)
_SKLEARN_DL = (
    Path.home()
    / "scikit_learn_data"
    / "openml"
    / "openml.org"
    / "data"
    / "v1"
    / "download"
)

SKLEARN_ARFF_PATHS: dict[str, list[Path]] = {
    "cpu_small": [
        _SKLEARN_DL / "52751" / "cpu_act.arff.gz",
        _SKLEARN_DL / "53295" / "cpu_act.arff.gz",
        _SKLEARN_DL / "3634" / "cpu_act.arff.gz",
    ],
    "house_16h": [_SKLEARN_DL / "52752" / "house_16H.arff.gz"],
    "kin8nm": [_SKLEARN_DL / "3626" / "kin8nm.arff.gz"],
    "meps_19": [_SKLEARN_DL / "22120800" / "medical_cost.arff.gz"],
    "abalone": [_SKLEARN_DL / "22111820" / "abalone.arff.gz"],
}

CHARTER_SURROGATE_SPECS: dict[str, dict[str, int]] = {
    "diamond": dict(n_samples=5000, n_num=6, n_cat=3, cat_card=5, seed=1197),
    "acs_income": dict(n_samples=6000, n_num=10, n_cat=4, cat_card=8, seed=42178),
    "naval": dict(n_samples=8000, n_num=12, n_cat=0, cat_card=0, seed=151),
    "brazilian_housing": dict(n_samples=7000, n_num=6, n_cat=4, cat_card=5, seed=42688),
    "space_ga": dict(n_samples=4000, n_num=8, n_cat=0, cat_card=0, seed=201),
    "yacht": dict(n_samples=308, n_num=6, n_cat=0, cat_card=0, seed=198),
}


def _offline_only() -> bool:
    """Default offline-first (shared env var with TS1)."""
    return os.environ.get("TAILSCORE_OFFLINE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def resolve_flagship_9_datasets() -> list[str]:
    """Return TS1 selected_main when present, else charter nine minus kin8nm."""
    if _TS1_SELECTION_JSON.exists():
        try:
            payload = json.loads(_TS1_SELECTION_JSON.read_text(encoding="utf-8"))
            selected = [
                name for name in (payload.get("selected_main") or []) if name != "kin8nm"
            ]
            if len(selected) >= 9:
                return list(selected[:9])
        except (json.JSONDecodeError, OSError):
            pass
    return list(FLAGSHIP_9_DATASETS)


def get_dataset_list(scope: str = "all") -> list[str]:
    if scope == "charter":
        return list(CHARTER_DATASETS)
    if scope == "openml_extra":
        return list(OPENML_EXTRA_DATASETS)
    return list(ALL_DATASETS)


def load_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    key = name.lower().replace("-", "_").replace(" ", "_")
    loader = _LOADERS.get(key)
    if loader is None:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {ALL_DATASETS}")
    return loader(test_size=test_size, random_state=random_state)


def make_synthetic_smoke(
    n_samples: int = 512,
    n_num: int = 6,
    n_cat: int = 2,
    cat_card: int = 4,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Fast offline mixed-tabular smoke data with R6-style tail injection."""
    return make_r6_synthetic(
        dependency="piecewise",
        tail_severity="rare_10",
        cat_complexity="low",
        n_samples=n_samples,
        random_state=random_state,
        test_size=test_size,
        name="synthetic_smoke",
    )


def make_r6_synthetic(
    *,
    dependency: str,
    tail_severity: str,
    cat_complexity: str = "low",
    n_samples: int = 512,
    random_state: int = 42,
    test_size: float = 0.2,
    name: str | None = None,
) -> dict[str, Any]:
    """R6 authority grid cell (offline, imports TS1 generator when available)."""
    del test_size
    label = name or f"r6_{dependency}_{tail_severity}_{cat_complexity}"
    try:
        import sys
        from pathlib import Path

        ts1_src = (
            Path(__file__).resolve().parents[5]
            / "26_InfoSci_TailScore"
            / "2_Code"
            / "src"
        )
        ts1_str = str(ts1_src)
        if ts1_src.is_dir() and ts1_str not in sys.path:
            sys.path.insert(0, ts1_str)
        from tailscore.data.synthetic import make_r6_suite_dataset

        data = make_r6_suite_dataset(
            dependency=dependency,
            tail_severity=tail_severity,
            cat_complexity=cat_complexity,
            n_samples=n_samples,
            seed=random_state,
        )
        data["name"] = label
        data["generator"] = "tailscore_r6_suite"
        return data
    except Exception as exc:
        import warnings

        warnings.warn(f"make_r6_synthetic fallback for {label}: {exc}", stacklevel=2)
        return _fallback_synthetic_smoke(
            n_samples=n_samples,
            random_state=random_state,
            name=label,
        )


def _fallback_synthetic_smoke(
    n_samples: int,
    random_state: int,
    name: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    num = rng.standard_normal((n_samples, 6))
    cats = rng.integers(0, 4, size=(n_samples, 2))
    X_df = pd.DataFrame(num, columns=[f"num_{i}" for i in range(6)])
    for j in range(2):
        X_df[f"cat_{j}"] = cats[:, j].astype(str)
    w = rng.normal(size=(7, 1))
    latent = num @ w[:6] + 0.3 * cats.sum(axis=1, keepdims=True)
    y = np.exp(np.clip(latent.squeeze() + 0.5 * rng.standard_normal(n_samples), -3, 3))
    return _finalize(X_df, y, name, test_size=0.2, random_state=random_state)


def _charter_surrogate_frame(
    name: str,
    n_samples: int,
    n_num: int,
    n_cat: int,
    cat_card: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    num = rng.standard_normal((n_samples, n_num))
    X_df = pd.DataFrame(num, columns=[f"num_{i}" for i in range(n_num)])
    cat_effect = np.zeros(n_samples)
    if n_cat > 0 and cat_card > 0:
        cats = rng.integers(0, cat_card, size=(n_samples, n_cat))
        for j in range(n_cat):
            X_df[f"cat_{j}"] = cats[:, j].astype(str)
        cat_effect = 0.35 * cats.sum(axis=1)
    w = rng.normal(size=n_num)
    latent = num @ w + cat_effect
    noise = rng.standard_normal(n_samples)
    tail_mask = latent > np.quantile(latent, 0.85)
    noise[tail_mask] *= 2.5
    y = np.exp(0.25 * latent + 0.45 * noise) + 0.1 * rng.exponential(1.0, n_samples)
    target = "target"
    X_df[target] = y
    return X_df, target


def _load_arff_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data, _meta = arff.loadarff(fh)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    return df


def _try_pilot_cache(name: str, **kw) -> dict[str, Any] | None:
    for cache_dir in _PILOT_CACHE_DIRS:
        csv_path = cache_dir / f"{name}.csv"
        if not csv_path.exists():
            continue
        meta_path = cache_dir / f"{name}.meta.json"
        target = "target"
        source = "pilot_cache_csv"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target = meta.get("target", target)
            source = meta.get("source", source)
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            target = df.columns[-1]
        y = pd.to_numeric(df[target], errors="coerce").values
        X_df = df.drop(columns=[target])
        valid = ~np.isnan(y)
        return _finalize(
            X_df[valid].reset_index(drop=True),
            y[valid],
            name,
            charter=True,
            source=source,
            surrogate="surrogate" in source,
            **kw,
        )
    return None


def _try_arff_cache(
    name: str,
    paths: list[Path],
    *,
    tail_inject: bool = False,
    **kw,
) -> dict[str, Any] | None:
    for path in paths:
        if not path.exists():
            continue
        df = _load_arff_gz(path)
        target = df.columns[-1]
        y = pd.to_numeric(df[target], errors="coerce").values
        X_df = df.drop(columns=[target])
        valid = ~np.isnan(y)
        return _finalize(
            X_df[valid].reset_index(drop=True),
            y[valid],
            name,
            charter=True,
            tail_inject=tail_inject,
            source=f"sklearn_arff:{path.parent.name}",
            **kw,
        )
    return None


def _load_charter_dataset(
    name: str,
    *,
    arff_paths: list[Path] | None = None,
    openml_ids: list[int] | None = None,
    openml_name: str | None = None,
    surrogate_spec: dict[str, int] | None = None,
    tail_inject: bool = False,
    **kw,
) -> dict[str, Any]:
    """Offline-first loader: pilot CSV → local ARFF → OpenML (optional) → surrogate."""
    cached = _try_pilot_cache(name, tail_inject=tail_inject, **kw)
    if cached is not None:
        return cached

    if arff_paths:
        arff_hit = _try_arff_cache(name, arff_paths, tail_inject=tail_inject, **kw)
        if arff_hit is not None:
            return arff_hit

    if not _offline_only():
        try:
            if openml_name:
                return _load_openml_by_name(openml_name, name, charter=True, **kw)
            if openml_ids:
                return _load_openml(
                    name, openml_ids, charter=True, tail_inject=tail_inject, **kw
                )
        except Exception:
            pass

    if surrogate_spec is None and name in CHARTER_SURROGATE_SPECS:
        surrogate_spec = CHARTER_SURROGATE_SPECS[name]
    if surrogate_spec is not None:
        return _load_mixed_type_surrogate(name, **surrogate_spec, **kw)

    raise RuntimeError(
        f"Could not load charter dataset {name!r} offline "
        f"(cache dirs={_PILOT_CACHE_DIRS}, offline={_offline_only()})"
    )


def _stratified_split(X, y, test_size, random_state, n_bins: int = 10):
    from sklearn.preprocessing import KBinsDiscretizer

    y_arr = np.asarray(y).ravel()
    n_unique = len(np.unique(y_arr))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return train_test_split(X, y_arr, test_size=test_size, random_state=random_state)

    binner = KBinsDiscretizer(n_bins=actual_bins, encode="ordinal", strategy="quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_binned = binner.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(int)
    try:
        return train_test_split(
            X, y_arr, test_size=test_size, random_state=random_state, stratify=y_binned
        )
    except ValueError:
        return train_test_split(X, y_arr, test_size=test_size, random_state=random_state)


def _build_preprocessor(X_df: pd.DataFrame) -> tuple[ColumnTransformer, list[int]]:
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers, remainder="drop")
    n_num = len(num_cols)
    cat_indices = list(range(n_num, n_num + len(cat_cols)))
    return pre, cat_indices


def _finalize(
    X_df: pd.DataFrame,
    y: np.ndarray,
    name: str,
    test_size: float,
    random_state: int,
    *,
    charter: bool = False,
    tail_inject: bool = False,
    source: str | None = None,
    surrogate: bool = False,
) -> dict[str, Any]:
    pre, cat_indices = _build_preprocessor(X_df)
    X_train_raw, X_test_raw, y_train, y_test = _stratified_split(
        X_df, y, test_size, random_state
    )
    pre.fit(X_train_raw)
    X_train = pre.transform(X_train_raw).astype(np.float32)
    X_test = pre.transform(X_test_raw).astype(np.float32)

    if tail_inject:
        y_train, y_test = _inject_kin8nm_tail(y_train, y_test, random_state)

    try:
        feature_names = pre.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    return {
        "name": name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": np.asarray(y_train, dtype=np.float64),
        "y_test": np.asarray(y_test, dtype=np.float64),
        "feature_names": feature_names,
        "cat_indices": cat_indices,
        "charter": charter,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        **({"source": source} if source else {}),
        **({"surrogate": True} if surrogate else {}),
    }


def _inject_kin8nm_tail(y_train: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_tr = y_train.copy()
    y_te = y_test.copy()
    for arr in (y_tr, y_te):
        q95 = np.quantile(arr, 0.95)
        tail = arr >= q95
        arr[tail] = arr[tail] * (1.0 + 0.5 * rng.random(tail.sum()))
    return y_tr, y_te


def _load_openml(name: str, ids: list[int], *, charter: bool = False, tail_inject: bool = False, **kw):
    from sklearn.datasets import fetch_openml

    last_err: Exception | None = None
    for data_id in ids:
        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            frame = bunch.frame.copy()
            target = bunch.target_names[0] if bunch.target_names else bunch.target.name
            y = pd.to_numeric(frame[target], errors="coerce").values
            X_df = frame.drop(columns=[target])
            valid = ~np.isnan(y)
            return _finalize(
                X_df[valid].reset_index(drop=True),
                y[valid],
                name,
                charter=charter,
                tail_inject=tail_inject,
                **kw,
            )
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"OpenML load failed for {name} ids={ids}: {last_err}")


def _load_openml_by_name(openml_name: str, registry_name: str, *, charter: bool = False, **kw):
    from sklearn.datasets import fetch_openml

    data = fetch_openml(openml_name, version=1, as_frame=True, parser="auto")
    X_df = data.data.copy()
    y = pd.to_numeric(data.target, errors="coerce").values.astype(float)
    valid = ~np.isnan(y)
    return _finalize(
        X_df[valid].reset_index(drop=True),
        y[valid],
        registry_name,
        charter=charter,
        **kw,
    )


def _load_california_housing(**kw):
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    return _finalize(data.data, data.target.values, "california_housing", charter=False, **kw)


def _load_mixed_type_surrogate(
    name: str,
    n_samples: int,
    n_num: int,
    n_cat: int,
    cat_card: int,
    seed: int,
    **kw,
):
    X_df, target = _charter_surrogate_frame(name, n_samples, n_num, n_cat, cat_card, seed)
    y = X_df[target].values
    X_df = X_df.drop(columns=[target])
    out = _finalize(
        X_df,
        y,
        name,
        charter=True,
        source=f"charter_surrogate:seed={seed}",
        surrogate=True,
        **kw,
    )
    return out


def _load_meps_19(**kw):
    return _load_charter_dataset(
        "meps_19",
        arff_paths=SKLEARN_ARFF_PATHS.get("meps_19"),
        openml_name="MEPS",
        surrogate_spec=dict(n_samples=3000, n_num=8, n_cat=3, cat_card=6, seed=19),
        **kw,
    )


def _load_acs_income(**kw):
    return _load_charter_dataset(
        "acs_income",
        openml_ids=[42178, 43141],
        surrogate_spec=CHARTER_SURROGATE_SPECS["acs_income"],
        **kw,
    )


def _load_brazilian_housing(**kw):
    return _load_charter_dataset(
        "brazilian_housing",
        openml_ids=[42688],
        surrogate_spec=CHARTER_SURROGATE_SPECS["brazilian_housing"],
        **kw,
    )


def _load_diamond(**kw):
    return _load_charter_dataset(
        "diamond",
        openml_ids=OPENML_IDS["diamond"],
        surrogate_spec=CHARTER_SURROGATE_SPECS["diamond"],
        **kw,
    )


def _load_house_16h(**kw):
    return _load_charter_dataset(
        "house_16h",
        arff_paths=SKLEARN_ARFF_PATHS.get("house_16h"),
        openml_name="house_16H",
        openml_ids=OPENML_IDS["house_16h"],
        **kw,
    )


def _load_cpu_small(**kw):
    return _load_charter_dataset(
        "cpu_small",
        arff_paths=SKLEARN_ARFF_PATHS.get("cpu_small"),
        openml_name="cpu_small",
        openml_ids=OPENML_IDS["cpu_small"],
        **kw,
    )


def _load_naval(**kw):
    return _load_charter_dataset(
        "naval",
        openml_ids=OPENML_IDS["naval"],
        surrogate_spec=CHARTER_SURROGATE_SPECS["naval"],
        **kw,
    )


def _load_kin8nm(**kw):
    return _load_charter_dataset(
        "kin8nm",
        arff_paths=SKLEARN_ARFF_PATHS.get("kin8nm"),
        openml_ids=OPENML_IDS["kin8nm"],
        tail_inject=True,
        **kw,
    )


def _load_abalone(**kw):
    return _load_charter_dataset(
        "abalone",
        arff_paths=SKLEARN_ARFF_PATHS.get("abalone"),
        openml_name="abalone",
        openml_ids=OPENML_IDS["abalone"],
        **kw,
    )


def _load_space_ga(**kw):
    return _load_charter_dataset(
        "space_ga",
        openml_name="space_ga",
        openml_ids=OPENML_IDS["space_ga"],
        surrogate_spec=CHARTER_SURROGATE_SPECS["space_ga"],
        **kw,
    )


def _load_yacht(**kw):
    return _load_charter_dataset(
        "yacht",
        openml_name="yacht_hydrodynamics",
        openml_ids=OPENML_IDS["yacht"],
        surrogate_spec=CHARTER_SURROGATE_SPECS["yacht"],
        **kw,
    )


def make_linkage_dgp(
    *,
    variant: str = "piecewise_high",
    n_samples: int = 512,
    random_state: int = 42,
    test_size: float = 0.2,
    name: str | None = None,
) -> dict[str, Any]:
    """Clause-3 linkage DGP: strong within-bin (X,y) coupling in upper tail bins.

    CopulaCalib uniform-y breaks tail coupling; MaTBridge pair resampling preserves it.
    """
    label = name or f"r6_linkage_{variant}"
    rng = np.random.default_rng(random_state + hash(variant) % 997)
    n_num, n_cat, cat_card = 8, 3, 8
    num = rng.standard_normal((n_samples, n_num))
    cats = rng.integers(0, cat_card, size=(n_samples, n_cat))
    X_df = pd.DataFrame(num, columns=[f"num_{i}" for i in range(n_num)])
    for j in range(n_cat):
        X_df[f"cat_{j}"] = cats[:, j].astype(str)

    w = rng.normal(size=n_num)
    base = num @ w
    q70, q80 = np.quantile(base, [0.70, 0.80])
    tail = base >= q70
    ultra = base >= q80

    strength = {
        "piecewise_high": 2.2,
        "piecewise_mid": 1.6,
        "bilinear_high": 2.0,
        "bilinear_mid": 1.4,
        "sparse_high": 1.8,
        "linear_high": 1.5,
    }.get(variant, 1.8)

    cat_tail = (cats[:, 0] >= cat_card - 2).astype(float)
    cat_mid = cats[:, 1].astype(float) / max(cat_card - 1, 1)
    y = base.copy()
    y[tail] += strength * (0.6 + 0.4 * cat_tail[tail]) * (base[tail] - q70)
    y[ultra] += strength * 0.8 * cat_mid[ultra] * (base[ultra] - q80)
    y += 0.12 * rng.standard_normal(n_samples)
    y = np.exp(y / 2.8) + 0.08 * rng.exponential(1.0, n_samples)

    return _finalize(
        X_df,
        y,
        label,
        test_size=test_size,
        random_state=random_state,
        charter=True,
    )


def _linkage_loader(variant: str):
    def _fn(**kw):
        return make_linkage_dgp(variant=variant, name=f"r6_linkage_{variant}", **kw)

    return _fn


_LOADERS: dict[str, Callable[..., dict[str, Any]]] = {
    "meps_19": _load_meps_19,
    "acs_income": _load_acs_income,
    "brazilian_housing": _load_brazilian_housing,
    "diamond": _load_diamond,
    "house_16h": _load_house_16h,
    "cpu_small": _load_cpu_small,
    "naval": _load_naval,
    "kin8nm": _load_kin8nm,
    "abalone": _load_abalone,
    "space_ga": _load_space_ga,
    "yacht": _load_yacht,
    "california_housing": _load_california_housing,
    "synthetic_smoke": make_synthetic_smoke,
    "r6_piecewise_rare10": lambda **kw: make_r6_synthetic(
        dependency="piecewise", tail_severity="rare_10", name="r6_piecewise_rare10", **kw
    ),
    "r6_bilinear_rare10": lambda **kw: make_r6_synthetic(
        dependency="bilinear", tail_severity="rare_10", name="r6_bilinear_rare10", **kw
    ),
    "r6_sparse_rare10": lambda **kw: make_r6_synthetic(
        dependency="sparse_nonlinear", tail_severity="rare_10", name="r6_sparse_rare10", **kw
    ),
    "r6_sparse_rare15": lambda **kw: make_r6_synthetic(
        dependency="sparse_nonlinear", tail_severity="rare_15", name="r6_sparse_rare15", **kw
    ),
    "r6_linear_rare10": lambda **kw: make_r6_synthetic(
        dependency="linear", tail_severity="rare_10", name="r6_linear_rare10", **kw
    ),
    "r6_linear_rare15": lambda **kw: make_r6_synthetic(
        dependency="linear", tail_severity="rare_15", name="r6_linear_rare15", **kw
    ),
    "r6_piecewise_rare15": lambda **kw: make_r6_synthetic(
        dependency="piecewise", tail_severity="rare_15", name="r6_piecewise_rare15", **kw
    ),
    "r6_linkage_piecewise_high": _linkage_loader("piecewise_high"),
    "r6_linkage_piecewise_mid": _linkage_loader("piecewise_mid"),
    "r6_linkage_bilinear_high": _linkage_loader("bilinear_high"),
    "r6_linkage_bilinear_mid": _linkage_loader("bilinear_mid"),
    "r6_linkage_sparse_high": _linkage_loader("sparse_high"),
    "r6_linkage_linear_high": _linkage_loader("linear_high"),
}
