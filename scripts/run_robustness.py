"""CLI entrypoint: measures how each model's performance degrades as an
increasing fraction of test-time features are masked to missing (RQ7).

Trains each model normally (clean data), then evaluates it repeatedly
against the same test set with 0%, 5%, 20%, 50% of features masked to
missing -- isolating "does this model degrade gracefully" from "can this
model be trained on incomplete data" (a different, not-tested-here
question).

Usage:
    python scripts/run_robustness.py
    python scripts/run_robustness.py --datasets cd1 --models xgboost lightgbm logreg
    python scripts/run_robustness.py --missing-rates 0.0 0.1 0.3
    python scripts/run_robustness.py --fresh
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
from tabfm_bench.models import RAW_INPUT_MODELS, get_model
from tabfm_bench.robustness import inject_missingness

ROOT = Path(__file__).resolve().parents[1]
# Same TABFM_RESULTS_DIR override as run_benchmark.py -- e.g. a Google
# Drive-mounted path in Colab, so this survives a session disconnect too.
# Missing this was a real bug: a completed 3-hour robustness run landed
# only on the Colab VM's ephemeral local disk and was lost on restart.
RESULTS_DIR = Path(os.environ.get("TABFM_RESULTS_DIR", ROOT / "results"))
ROBUSTNESS_JSONL = RESULTS_DIR / "robustness.jsonl"

DEFAULT_MISSING_RATES = [0.0, 0.05, 0.2, 0.5]


def load_config():
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)


def _append_jsonl(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]
    all_models = cfg["models"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=all_datasets)
    parser.add_argument("--models", nargs="*", default=all_models)
    parser.add_argument(
        "--missing-rates", nargs="*", type=float, default=DEFAULT_MISSING_RATES
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing results/robustness.jsonl first instead of appending.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        ROBUSTNESS_JSONL.unlink(missing_ok=True)

    for dataset_config in tqdm(args.datasets, desc="datasets"):
        split = load_finbench(dataset_config)

        for model_name in tqdm(args.models, desc=dataset_config, leave=False):
            try:
                model = get_model(
                    model_name,
                    cat_idx=split.cat_idx,
                    num_idx=split.num_idx,
                    col_name=split.col_name,
                )
                if model_name in RAW_INPUT_MODELS:
                    X_train, X_test = split.X_train_df, split.X_test_df
                else:
                    X_train, X_test = split.X_train, split.X_test
                model.fit(X_train, split.y_train)
            except Exception as e:
                tqdm.write(f"FIT FAILED: {dataset_config}/{model_name} -- {type(e).__name__}: {e}")
                continue

            for missing_rate in args.missing_rates:
                try:
                    X_test_masked = inject_missingness(X_test, missing_rate)
                    proba = model.predict_proba(X_test_masked)[:, 1]
                    metrics = compute_metrics(split.y_test, proba)
                    _append_jsonl(
                        ROBUSTNESS_JSONL,
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
                finally:
                    # Calling predict_proba 4x in a row on the same TabFM/
                    # SAP-RPT instance (once per missing rate) was observed
                    # to accumulate GPU memory across calls rather than
                    # releasing it -- free bytes dropped from run to run on
                    # the *same* fitted model (e.g. 1.57 GiB -> 981 MiB free
                    # between two calls on cd2), causing OOMs on datasets
                    # that would have been fine for a single prediction
                    # (ld2, only 11 columns, failed here despite cd1's 9
                    # columns succeeding in the main benchmark's single-call
                    # path). Releasing cached memory after every call
                    # prevents that accumulation.
                    if model_name in RAW_INPUT_MODELS:
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass

    rows = _read_jsonl(ROBUSTNESS_JSONL)
    if rows:
        df = pd.json_normalize(rows)
        df = df.drop_duplicates(subset=["dataset", "model", "missing_rate"], keep="last")
        df.to_csv(RESULTS_DIR / "robustness.csv", index=False)
        print(f"Wrote {len(df)} rows to {RESULTS_DIR / 'robustness.csv'}")


if __name__ == "__main__":
    main()
