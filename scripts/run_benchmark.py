"""CLI entrypoint: runs every (dataset, model) pair in configs/datasets.yaml
and writes results/results.csv plus results/shap_agreement.csv.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --datasets cd1 ld1 --models tabfm xgboost
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from tabfm_bench.agreement import compare_predictions
from tabfm_bench.data import load_finbench
from tabfm_bench.explain import compare_shap_agreement, get_shap_values
from tabfm_bench.models import RAW_INPUT_MODELS, get_model
from tabfm_bench.run import run_single

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_config():
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]
    all_models = cfg["models"]
    protected_attrs = {d["config"]: d.get("protected_attribute") for d in cfg["datasets"]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=all_datasets)
    parser.add_argument("--models", nargs="*", default=all_models)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    results = []
    shap_rows = []
    prediction_agreement_rows = []

    for dataset_config in tqdm(args.datasets, desc="datasets"):
        fitted_models = {}
        shap_values = {}
        full_probas = {}
        split = load_finbench(dataset_config) if not args.skip_shap else None

        for model_name in tqdm(args.models, desc=dataset_config, leave=False):
            result = run_single(
                dataset_config, model_name, protected_attribute=protected_attrs.get(dataset_config)
            )
            results.append(result)

            if not args.skip_shap:
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
                fitted_models[model_name] = model
                full_probas[model_name] = model.predict_proba(X_test)[:, 1]
                shap_values[model_name] = get_shap_values(
                    model, model_name, X_train, X_test[:50]
                )

        if not args.skip_shap and len(shap_values) > 1:
            names = list(shap_values)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    agreement = compare_shap_agreement(
                        shap_values[names[i]], shap_values[names[j]]
                    )
                    shap_rows.append(
                        {
                            "dataset": dataset_config,
                            "model_a": names[i],
                            "model_b": names[j],
                            "shap_rank_agreement": agreement,
                        }
                    )
                    pred_agreement = compare_predictions(
                        split.y_test, full_probas[names[i]], full_probas[names[j]]
                    )
                    prediction_agreement_rows.append(
                        {
                            "dataset": dataset_config,
                            "model_a": names[i],
                            "model_b": names[j],
                            **pred_agreement,
                        }
                    )

    results_df = pd.json_normalize(results)
    results_df.to_csv(RESULTS_DIR / "results.csv", index=False)

    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(RESULTS_DIR / "shap_agreement.csv", index=False)

    if prediction_agreement_rows:
        pd.DataFrame(prediction_agreement_rows).to_csv(
            RESULTS_DIR / "prediction_agreement.csv", index=False
        )

    print(f"Wrote {len(results_df)} rows to {RESULTS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
