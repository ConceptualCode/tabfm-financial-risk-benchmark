"""Re-runs the missingness robustness sweep (see run_robustness.py) for the
3 classical models using the tuned hyperparameters from tune_classical.py's
results/tuned_best_params.jsonl, instead of models.py's fixed defaults.

The original robustness.csv was generated before any real hyperparameter
search existed, so its XGBoost/LightGBM/logreg numbers reflect the same
single fixed config now known to be a hand-picked default rather than a
tuned one (see tune_classical.py's docstring). This script re-fits each
classical model with its per-dataset winning config and re-runs the same
0%/5%/20%/50% missingness sweep, so the robustness numbers are consistent
with the rest of the tuned results.

TabFM is untouched and not re-run here: it's zero-shot, nothing about it
changes based on classical-model tuning, and its robustness numbers already
match what's in the article.

Writes to results/robustness_tuned.csv -- a new file, not an overwrite of
the original.

Usage:
    python scripts/run_robustness_tuned.py
    python scripts/run_robustness_tuned.py --datasets cd1 ld1
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from tabfm_bench.data import load_finbench
from tabfm_bench.metrics import compute_metrics
from tabfm_bench.robustness import inject_missingness
from tune_classical import BUILDERS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TABFM_RESULTS_DIR", ROOT / "results"))
BEST_PARAMS_JSONL = RESULTS_DIR / "tuned_best_params.jsonl"
ROBUSTNESS_TUNED_JSONL = RESULTS_DIR / "robustness_tuned.jsonl"

DEFAULT_MISSING_RATES = [0.0, 0.05, 0.2, 0.5]
CLASSICAL_MODELS = ["xgboost", "lightgbm", "logreg"]


def _load_best_params() -> dict:
    """Returns {(dataset, model): params_dict}, keeping the last (most
    recent) entry per key in case tune_classical.py was re-run for a subset
    of models (as it was here, to fix the logreg class_weight issue).
    """
    if not BEST_PARAMS_JSONL.exists():
        raise FileNotFoundError(
            f"{BEST_PARAMS_JSONL} not found -- run tune_classical.py first."
        )
    best = {}
    with open(BEST_PARAMS_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            best[(row["dataset"], row["model"])] = row["best_params"]
    return best


def _append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config():
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=all_datasets)
    parser.add_argument("--models", nargs="*", default=CLASSICAL_MODELS, choices=CLASSICAL_MODELS)
    parser.add_argument("--missing-rates", nargs="*", type=float, default=DEFAULT_MISSING_RATES)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        ROBUSTNESS_TUNED_JSONL.unlink(missing_ok=True)

    best_params = _load_best_params()

    for dataset_config in tqdm(args.datasets, desc="datasets"):
        split = load_finbench(dataset_config)

        for model_name in tqdm(args.models, desc=dataset_config, leave=False):
            key = (dataset_config, model_name)
            if key not in best_params:
                tqdm.write(f"SKIPPED: no tuned params for {dataset_config}/{model_name}")
                continue

            try:
                model = BUILDERS[model_name](best_params[key], split)
                model.fit(split.X_train, split.y_train)
            except Exception as e:
                tqdm.write(f"FIT FAILED: {dataset_config}/{model_name} -- {type(e).__name__}: {e}")
                continue

            for missing_rate in args.missing_rates:
                try:
                    X_test_masked = inject_missingness(split.X_test, missing_rate)
                    proba = model.predict_proba(X_test_masked)[:, 1]
                    metrics = compute_metrics(split.y_test, proba)
                    _append_jsonl(
                        ROBUSTNESS_TUNED_JSONL,
                        {
                            "dataset": dataset_config,
                            "model": model_name,
                            "missing_rate": missing_rate,
                            **metrics,
                        },
                    )
                except Exception as e:
                    tqdm.write(
                        f"PREDICT FAILED: {dataset_config}/{model_name}@{missing_rate} "
                        f"-- {type(e).__name__}: {e}"
                    )

    rows = _read_jsonl(ROBUSTNESS_TUNED_JSONL)
    if rows:
        df = pd.json_normalize(rows)
        df = df.drop_duplicates(subset=["dataset", "model", "missing_rate"], keep="last")
        df.to_csv(RESULTS_DIR / "robustness_tuned.csv", index=False)
        print(f"Wrote {len(df)} rows to {RESULTS_DIR / 'robustness_tuned.csv'}")


if __name__ == "__main__":
    main()
