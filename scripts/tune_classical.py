"""Real hyperparameter search for the classical baselines (XGBoost,
LightGBM, logistic regression) -- these were previously run at a single
fixed, hand-picked config identical across all 10 datasets, which the
article calls "tuned" but which OBJECTIVES.md itself discloses was never an
actual search ("No hyperparameter search beyond sane defaults for
baselines"). This script runs one.

Uses the FinBench validation split (X_val/y_val) that load_finbench already
provides per dataset but that the original benchmark run never referenced --
confirmed by grep across run.py/run_benchmark.py before writing this. For
each (dataset, model), it draws N random hyperparameter sets, fits each on
train, scores AUC-ROC on val (the metric the article's tables report), keeps
the best, then refits that config on train and scores on the held-out test
set using the same metrics/fairness/cost pipeline as run.py's run_single --
so results_tuned.csv is directly comparable, column for column, to the
original results.csv.

Writes to results/results_tuned.jsonl / .csv -- a new file, not an
overwrite, so the original fixed-default numbers stay available for a
before/after comparison rather than being silently replaced.

TabFM is out of scope here on purpose: it's zero-shot, there is no
hyperparameter to search over, and its numbers don't change based on
anything in this script.

Usage:
    python scripts/tune_classical.py
    python scripts/tune_classical.py --datasets cd1 ld1 --models xgboost
    python scripts/tune_classical.py --n-trials 50 --fresh
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from tabfm_bench.cost import measure_fit_and_predict
from tabfm_bench.data import load_finbench
from tabfm_bench.fairness import disparate_impact_metrics
from tabfm_bench.metrics import (
    compute_metrics,
    cost_weighted_score,
    find_optimal_threshold,
    reliability_curve,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TABFM_RESULTS_DIR", ROOT / "results"))
RESULTS_JSONL = RESULTS_DIR / "results_tuned.jsonl"
BEST_PARAMS_JSONL = RESULTS_DIR / "tuned_best_params.jsonl"

SEED = 42
DEFAULT_COST_FALSE_NEGATIVE = 5.0
DEFAULT_COST_FALSE_POSITIVE = 1.0


# ---------------------------------------------------------------------------
# Search spaces -- ranges as drafted and shared for review before this
# script was written; see the conversation history / commit message for the
# rationale behind each range.
# ---------------------------------------------------------------------------
def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_xgboost_params(rng, y_train):
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return {
        "n_estimators": int(rng.integers(100, 801)),
        "max_depth": int(rng.integers(3, 11)),
        "learning_rate": _log_uniform(rng, 0.01, 0.3),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "min_child_weight": int(rng.integers(1, 11)),
        "reg_alpha": _log_uniform(rng, 1e-3, 10) if rng.random() > 0.2 else 0.0,
        "reg_lambda": _log_uniform(rng, 0.1, 20),
        "scale_pos_weight": float(rng.choice([1.0, scale_pos_weight])),
    }


def sample_lightgbm_params(rng):
    return {
        "n_estimators": int(rng.integers(100, 801)),
        "num_leaves": int(rng.integers(15, 256)),
        "max_depth": int(rng.choice([-1, 4, 6, 8, 10])),
        "learning_rate": _log_uniform(rng, 0.01, 0.3),
        "bagging_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_freq": 1,
        "feature_fraction": float(rng.uniform(0.6, 1.0)),
        "min_child_samples": int(rng.integers(5, 51)),
        "reg_alpha": _log_uniform(rng, 1e-3, 10) if rng.random() > 0.2 else 0.0,
        "reg_lambda": _log_uniform(rng, 1e-3, 10) if rng.random() > 0.2 else 0.0,
    }


def sample_logreg_params(rng):
    # sklearn >=1.8 deprecates `penalty=` in favor of specifying `l1_ratio`
    # directly against the default (elasticnet-capable) penalty: l1_ratio=0
    # is pure L2, l1_ratio=1 is pure L1, anything between is elasticnet.
    #
    # `class_weight` deliberately left out of this search: selecting purely
    # by validation AUC-ROC let a first pass pick `class_weight="balanced"`
    # on most datasets, which reweights the loss to treat classes as equally
    # frequent -- a real ranking boost on imbalanced data, but it distorts
    # predicted probabilities away from the true base rate as a direct side
    # effect. Confirmed empirically: that run's calibration error blew up
    # 10x-30x on several datasets (e.g. cf2: 0.013 -> 0.386) while AUC-ROC
    # barely moved (+0.0005 mean). AUC-ROC-only selection has no way to see
    # that trade and will walk into it every time, so the lever is removed
    # rather than the objective patched.
    l1_ratio = [0.0, 1.0, round(float(rng.uniform(0.1, 0.9)), 3)][rng.integers(0, 3)]
    return {
        "C": _log_uniform(rng, 0.001, 100),
        "l1_ratio": l1_ratio,
        "solver": "saga",
    }


# ---------------------------------------------------------------------------
# Model builders -- mirror models.py's preprocessing exactly (same
# categorical-feature handling for LightGBM, same ColumnTransformer for
# logreg) but accept a sampled hyperparameter dict instead of a hardcoded
# config, so the *only* thing this changes is tuning, not preprocessing.
# ---------------------------------------------------------------------------
class _CategoricalAwareLGBM:
    def __init__(self, model, cat_idx):
        self._model = model
        self._cat_idx = cat_idx or []

    def fit(self, X, y):
        cat_feature = self._cat_idx if self._cat_idx else "auto"
        self._model.fit(X, y, categorical_feature=cat_feature)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def build_xgboost_tuned(params):
    from xgboost import XGBClassifier

    return XGBClassifier(
        **params, eval_metric="logloss", n_jobs=-1, random_state=SEED
    )


def build_lightgbm_tuned(params, cat_idx):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**params, n_jobs=-1, random_state=SEED, verbose=-1)
    return _CategoricalAwareLGBM(model, cat_idx)


def build_logreg_tuned(params, cat_idx, num_idx):
    cat_idx = cat_idx or []
    num_idx = num_idx or []
    cat_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("cat", cat_pipeline, cat_idx), ("num", num_pipeline, num_idx)]
    )
    return make_pipeline(
        preprocessor, LogisticRegression(max_iter=2000, random_state=SEED, **params)
    )


BUILDERS = {
    "xgboost": lambda params, split: build_xgboost_tuned(params),
    "lightgbm": lambda params, split: build_lightgbm_tuned(params, split.cat_idx),
    "logreg": lambda params, split: build_logreg_tuned(params, split.cat_idx, split.num_idx),
}
SAMPLERS = {
    "xgboost": lambda rng, split: sample_xgboost_params(rng, split.y_train),
    "lightgbm": lambda rng, split: sample_lightgbm_params(rng),
    "logreg": lambda rng, split: sample_logreg_params(rng),
}
N_TRIALS_DEFAULT = {"xgboost": 50, "lightgbm": 50, "logreg": 20}


def _search(model_name, split, n_trials, seed):
    """Random search selected by validation AUC-ROC. Returns (best_params,
    best_val_auc). All 3 classical models use split.X_train/X_val/X_test
    (pre-encoded arrays) -- same input representation run.py already used
    for them, unchanged; only the hyperparameters and selection process are
    new.
    """
    rng = np.random.default_rng(seed)
    best_params, best_score = None, -np.inf

    for _ in range(n_trials):
        params = SAMPLERS[model_name](rng, split)
        model = BUILDERS[model_name](params, split)
        model.fit(split.X_train, split.y_train)
        val_proba = model.predict_proba(split.X_val)[:, 1]
        score = roc_auc_score(split.y_val, val_proba)
        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score


def _evaluate_on_test(model_name, params, split, protected_attribute):
    """Refits the winning config on train, scores on test -- same metrics,
    calibration, threshold-economics, and fairness pipeline as
    run.py's run_single, so results_tuned.csv lines up column-for-column
    with the original results.csv.
    """
    model = BUILDERS[model_name](params, split)
    cost = measure_fit_and_predict(model, split.X_train, split.y_train, split.X_test)
    proba = cost.pop("proba")[:, 1]

    metrics = compute_metrics(split.y_test, proba)
    predicted_bins, observed_freqs = reliability_curve(split.y_test, proba)
    best_threshold, min_expected_cost = find_optimal_threshold(
        split.y_test, proba, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    cost_at_naive_threshold = cost_weighted_score(
        split.y_test, proba, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )

    result = {
        "dataset": split.config,
        "model": model_name,
        **metrics,
        **cost,
        "best_threshold": best_threshold,
        "min_expected_cost": min_expected_cost,
        "cost_at_naive_0.5_threshold": cost_at_naive_threshold,
        "calibration_predicted": predicted_bins.tolist(),
        "calibration_observed": observed_freqs.tolist(),
    }

    if protected_attribute is not None:
        y_pred_at_best = (proba >= best_threshold).astype(int)
        group_labels = split.X_test_df[protected_attribute].to_numpy()
        fairness = disparate_impact_metrics(split.y_test, y_pred_at_best, group_labels)
        result["fairness_protected_attribute"] = protected_attribute
        result.update({f"fairness_{k}": v for k, v in fairness.items()})

    return result


def _append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _export_csv(jsonl_path: Path, csv_path: Path, dedup_keys=None):
    rows = _read_jsonl(jsonl_path)
    if not rows:
        return
    df = pd.json_normalize(rows)
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


def load_config():
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]
    protected_attrs = {d["config"]: d.get("protected_attribute") for d in cfg["datasets"]}
    classical_models = ["xgboost", "lightgbm", "logreg"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=all_datasets)
    parser.add_argument("--models", nargs="*", default=classical_models, choices=classical_models)
    parser.add_argument("--n-trials", type=int, default=None,
                         help="Overrides the per-model default (xgboost/lightgbm=50, logreg=20) "
                              "for every model given.")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        RESULTS_JSONL.unlink(missing_ok=True)
        BEST_PARAMS_JSONL.unlink(missing_ok=True)

    for dataset_config in tqdm(args.datasets, desc="datasets"):
        split = load_finbench(dataset_config)

        for model_name in tqdm(args.models, desc=dataset_config, leave=False):
            n_trials = args.n_trials or N_TRIALS_DEFAULT[model_name]
            try:
                best_params, best_val_auc = _search(model_name, split, n_trials, seed=SEED)
                _append_jsonl(BEST_PARAMS_JSONL, {
                    "dataset": dataset_config,
                    "model": model_name,
                    "n_trials": n_trials,
                    "best_val_auc_roc": best_val_auc,
                    "best_params": best_params,
                })

                result = _evaluate_on_test(
                    model_name, best_params, split, protected_attrs.get(dataset_config)
                )
                _append_jsonl(RESULTS_JSONL, result)
            except Exception as e:
                tqdm.write(f"FAILED: {dataset_config}/{model_name} -- {type(e).__name__}: {e}")
                continue

    _export_csv(RESULTS_JSONL, RESULTS_DIR / "results_tuned.csv", dedup_keys=["dataset", "model"])
    _export_csv(BEST_PARAMS_JSONL, RESULTS_DIR / "tuned_best_params.csv",
                dedup_keys=["dataset", "model"])


if __name__ == "__main__":
    main()
