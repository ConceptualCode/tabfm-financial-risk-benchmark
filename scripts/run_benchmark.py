"""CLI entrypoint: runs every (dataset, model) pair in configs/datasets.yaml.

Writes incrementally to results/*.jsonl (one JSON object appended to disk
immediately after each result is computed), then exports results/*.csv from
whatever's accumulated. Previously all results only existed in memory and
were written once at the very end of the run -- a crash, an OOM on one
model, or a Colab session disconnect at any point before that final line
lost everything computed so far. Incremental writes mean an interruption
only costs the remaining unfinished work. A single (dataset, model)
failure is also caught and logged rather than aborting the whole grid.

RESULTS_DIR can be overridden via the TABFM_RESULTS_DIR env var -- e.g. a
Google Drive-mounted path in Colab, so writes survive a session disconnect
without needing to remember to run a download step.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --datasets cd1 ld1 --models tabfm xgboost
    python scripts/run_benchmark.py --fresh          # clear prior results.jsonl first
    python scripts/run_benchmark.py --export-only     # just regenerate CSVs from existing jsonl
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from tabfm_bench.agreement import compare_predictions
from tabfm_bench.data import load_finbench
from tabfm_bench.explain import compare_shap_agreement, get_shap_values
from tabfm_bench.models import RAW_INPUT_MODELS
from tabfm_bench.run import run_single

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TABFM_RESULTS_DIR", ROOT / "results"))

RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
SHAP_JSONL = RESULTS_DIR / "shap_agreement.jsonl"
AGREEMENT_JSONL = RESULTS_DIR / "prediction_agreement.jsonl"


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


def _export_csv(jsonl_path: Path, csv_path: Path, dedup_keys: list = None):
    """Regenerates a CSV from a jsonl file. Rerunning the same (dataset,
    model) pair without --fresh appends a new row rather than replacing the
    old one -- dedup_keys keeps only the most recent row per key group
    (e.g. ["dataset", "model"]), on the assumption a later attempt
    supersedes an earlier one (a code fix, a different setting, etc.).
    """
    rows = _read_jsonl(jsonl_path)
    if not rows:
        return
    df = pd.json_normalize(rows)
    if dedup_keys:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
        if len(df) != before:
            print(
                f"{csv_path.name}: deduplicated {before} -> {len(df)} rows "
                f"(kept the latest attempt per {dedup_keys})"
            )
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


def main():
    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]
    all_models = cfg["models"]
    protected_attrs = {d["config"]: d.get("protected_attribute") for d in cfg["datasets"]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=all_datasets)
    parser.add_argument("--models", nargs="*", default=all_models)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing results/*.jsonl first, instead of appending to a prior run "
        "(without this, rerunning the same dataset/model pair adds a duplicate row).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip running any models; just regenerate results/*.csv from existing "
        "results/*.jsonl (useful after an interrupted run).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        _export_csv(RESULTS_JSONL, RESULTS_DIR / "results.csv", dedup_keys=["dataset", "model"])
        _export_csv(
            SHAP_JSONL,
            RESULTS_DIR / "shap_agreement.csv",
            dedup_keys=["dataset", "model_a", "model_b"],
        )
        _export_csv(
            AGREEMENT_JSONL,
            RESULTS_DIR / "prediction_agreement.csv",
            dedup_keys=["dataset", "model_a", "model_b"],
        )
        return

    if args.fresh:
        for path in (RESULTS_JSONL, SHAP_JSONL, AGREEMENT_JSONL):
            path.unlink(missing_ok=True)

    for dataset_config in tqdm(args.datasets, desc="datasets"):
        fitted_models = {}
        shap_values = {}
        full_probas = {}
        split = load_finbench(dataset_config) if not args.skip_shap else None

        for model_name in tqdm(args.models, desc=dataset_config, leave=False):
            try:
                if args.skip_shap:
                    result = run_single(
                        dataset_config,
                        model_name,
                        protected_attribute=protected_attrs.get(dataset_config),
                    )
                else:
                    # Reuse the model run_single already built and fit,
                    # rather than building/fitting a second, independent
                    # instance for SHAP -- for TabFM specifically, two
                    # ~6.5GB+ instances (each with 32-member ensemble
                    # overhead) resident on GPU at once caused a real CUDA
                    # OOM, not just wasted time.
                    result, fitted_models[model_name] = run_single(
                        dataset_config,
                        model_name,
                        protected_attribute=protected_attrs.get(dataset_config),
                        return_model=True,
                    )
                _append_jsonl(RESULTS_JSONL, result)
            except Exception as e:
                tqdm.write(f"FAILED: {dataset_config}/{model_name} -- {type(e).__name__}: {e}")
                continue

            if not args.skip_shap:
                try:
                    model = fitted_models[model_name]
                    if model_name in RAW_INPUT_MODELS:
                        X_train, X_test = split.X_train_df, split.X_test_df
                    else:
                        X_train, X_test = split.X_train, split.X_test
                    full_probas[model_name] = model.predict_proba(X_test)[:, 1]
                    shap_values[model_name] = get_shap_values(
                        model, model_name, X_train, X_test[:50]
                    )
                except Exception as e:
                    tqdm.write(
                        f"SHAP FAILED: {dataset_config}/{model_name} -- {type(e).__name__}: {e}"
                    )

        if not args.skip_shap and len(shap_values) > 1:
            names = list(shap_values)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    shap_row = {
                        "dataset": dataset_config,
                        "model_a": names[i],
                        "model_b": names[j],
                        "shap_rank_agreement": compare_shap_agreement(
                            shap_values[names[i]], shap_values[names[j]]
                        ),
                    }
                    _append_jsonl(SHAP_JSONL, shap_row)

                    agreement_row = {
                        "dataset": dataset_config,
                        "model_a": names[i],
                        "model_b": names[j],
                        **compare_predictions(
                            split.y_test, full_probas[names[i]], full_probas[names[j]]
                        ),
                    }
                    _append_jsonl(AGREEMENT_JSONL, agreement_row)

    _export_csv(RESULTS_JSONL, RESULTS_DIR / "results.csv", dedup_keys=["dataset", "model"])
    _export_csv(
        SHAP_JSONL,
        RESULTS_DIR / "shap_agreement.csv",
        dedup_keys=["dataset", "model_a", "model_b"],
    )
    _export_csv(
        AGREEMENT_JSONL,
        RESULTS_DIR / "prediction_agreement.csv",
        dedup_keys=["dataset", "model_a", "model_b"],
    )


if __name__ == "__main__":
    main()
