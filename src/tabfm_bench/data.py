"""Loaders for the FinBench (yuweiyin/FinBench) financial risk suite.

Loaded via direct file download (huggingface_hub.hf_hub_download) rather
than `datasets.load_dataset` -- FinBench ships as a legacy HF "dataset
script" (FinBench.py), and current `datasets` versions (we use 5.0.0) have
fully dropped support for that pattern, failing with:
    RuntimeError: Dataset scripts are no longer supported, but found FinBench.py
The underlying data is plain per-config .npy files, so we load those
directly instead.

Each split is exposed in two representations:
  - X_{split}: pre-encoded array (categoricals integer-encoded, numerics
    standardized) -- what the classical baselines (XGBoost/LightGBM/logreg)
    use, matching how a team would realistically feed them.
  - X_{split}_df: a pandas DataFrame with real column names and real
    category strings (reconstructed from stat_dict.json's cat_str mapping),
    raw (unstandardized) numeric values -- what TabFM/SAP-RPT use, matching
    how those models are documented to be used (they're built to handle
    real-world mixed-type tables, not opaque pre-encoded arrays). See
    OBJECTIVES.md's "Production Constraint" and RQ1 discussion.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "yuweiyin/FinBench"


@dataclass
class FinBenchSplit:
    config: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame
    cat_idx: list
    num_idx: list
    col_name: list


def _download(config: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=REPO_ID, filename=f"data/{config}/{filename}", repo_type="dataset"
    )


def _raw_dataframe(
    X_unscale: np.ndarray, cat_idx: list, cat_str: list, col_name: list
) -> pd.DataFrame:
    """Reconstructs a human-readable DataFrame from FinBench's unscaled
    array: categorical columns are still integer-coded even in the
    "unscale" file, so we map each code back to its real string label via
    stat_dict.json's cat_str (cat_str[i] lists the labels for cat_idx[i]).

    Builds each column as its own typed Series and assembles them with
    pd.concat, rather than constructing one DataFrame from the raw object
    array and coercing dtypes on slices afterward. Two reasons:
      - Positional (not name-based) throughout -- at least one FinBench
        config (cf2) has duplicate column names in its own stat_dict.json,
        which breaks name-based lookups with an ambiguous "Columns must be
        same length as key" error.
      - `df.iloc[:, positions] = df.iloc[:, positions].astype(float)` on a
        DataFrame built from a single object-dtype array mutates values in
        place but does NOT change the column block's dtype -- every column
        (including numeric ones) was still reported as dtype `object`
        downstream, which caused SAP-RPT/TabFM's type inference to treat
        numeric features as strings ("dtype is object... converting to
        str"). Casting each column as an individual Series does properly
        change its dtype.
    """
    cat_labels = dict(zip(cat_idx, cat_str))
    columns = []
    for i, name in enumerate(col_name):
        raw_col = X_unscale[:, i]
        if i in cat_labels:
            code_to_label = dict(enumerate(cat_labels[i]))
            series = pd.Series(raw_col, name=name).astype(int).map(code_to_label)
        else:
            series = pd.Series(raw_col, name=name).astype(float)
        columns.append(series)
    return pd.concat(columns, axis=1)


def load_finbench(config: str) -> FinBenchSplit:
    """Load one FinBench task (e.g. "cd1", "ld2", "cf1", "cc3")."""

    def _load_encoded(split_name: str):
        X = np.load(_download(config, f"X_{split_name}.npy"), allow_pickle=True)
        X = X.astype(np.float64)
        y = np.load(_download(config, f"y_{split_name}.npy"), allow_pickle=True)
        return X, y

    def _load_unscaled(split_name: str):
        return np.load(_download(config, f"X_{split_name}_unscale.npy"), allow_pickle=True)

    X_train, y_train = _load_encoded("train")
    X_val, y_val = _load_encoded("val")
    X_test, y_test = _load_encoded("test")

    with open(_download(config, "stat_dict.json")) as f:
        stat = json.load(f)
    cat_idx = stat.get("cat_idx", [])
    num_idx = stat.get("num_idx", [])
    col_name = stat.get("col_name", [])
    cat_str = stat.get("cat_str", [])

    X_train_df = _raw_dataframe(_load_unscaled("train"), cat_idx, cat_str, col_name)
    X_val_df = _raw_dataframe(_load_unscaled("val"), cat_idx, cat_str, col_name)
    X_test_df = _raw_dataframe(_load_unscaled("test"), cat_idx, cat_str, col_name)

    return FinBenchSplit(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        cat_idx=cat_idx,
        num_idx=num_idx,
        col_name=col_name,
    )


def list_configs() -> list[str]:
    return ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
