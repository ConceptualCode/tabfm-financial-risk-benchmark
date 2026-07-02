"""Loaders for the FinBench (yuweiyin/FinBench) financial risk suite."""

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset


@dataclass
class FinBenchSplit:
    config: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    cat_idx: list
    num_idx: list
    col_name: list


def load_finbench(config: str) -> FinBenchSplit:
    """Load one FinBench task (e.g. "cd1", "ld2", "cf1", "cc3").

    Returns train/validation/test arrays plus the categorical/numerical
    column indices FinBench ships with, so downstream models (TabFM,
    LightGBM) can be told which columns are categorical.
    """
    ds = load_dataset("yuweiyin/FinBench", config)

    def _extract(split):
        rows = ds[split]
        X = np.asarray(rows["X_ml"])
        y = np.asarray(rows["y"])
        return X, y

    X_train, y_train = _extract("train")
    X_val, y_val = _extract("validation")
    X_test, y_test = _extract("test")

    meta = ds["train"][0]
    cat_idx = meta.get("cat_idx", [])
    num_idx = meta.get("num_idx", [])
    col_name = meta.get("col_name", [])

    return FinBenchSplit(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        cat_idx=cat_idx,
        num_idx=num_idx,
        col_name=col_name,
    )


def list_configs() -> list[str]:
    return ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
