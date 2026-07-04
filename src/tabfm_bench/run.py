"""Runs one (dataset, model) pair end to end and returns a result dict
combining predictive metrics, calibration, cost-based threshold selection,
and fit/predict cost.

Explainability (SHAP) is run separately in scripts/run_benchmark.py since
it's slow and only needs to run once per (dataset, model) after the model
is already fit.
"""

from tabfm_bench.cost import measure_fit_and_predict
from tabfm_bench.data import load_finbench
from tabfm_bench.metrics import compute_metrics, cost_weighted_score, find_optimal_threshold, reliability_curve
from tabfm_bench.models import RAW_INPUT_MODELS, get_model

# Illustrative default cost ratio (missing an actual default costs 5x more
# than wrongly declining a good customer) -- NOT a real business number.
# Replace with actual loss-given-default / cost-of-declined-good-customer
# figures before treating the resulting threshold as a real recommendation.
DEFAULT_COST_FALSE_NEGATIVE = 5.0
DEFAULT_COST_FALSE_POSITIVE = 1.0


def run_single(dataset_config: str, model_name: str) -> dict:
    split = load_finbench(dataset_config)
    model = get_model(
        model_name, cat_idx=split.cat_idx, num_idx=split.num_idx, col_name=split.col_name
    )

    if model_name in RAW_INPUT_MODELS:
        # TabFM/SAP-RPT are documented to take raw, real-world tabular input
        # (real column names, real category strings) -- feeding them the
        # same pre-encoded array as the classical baselines would test them
        # outside their intended/production usage. See OBJECTIVES.md.
        X_train, X_test = split.X_train_df, split.X_test_df
    else:
        X_train, X_test = split.X_train, split.X_test

    cost = measure_fit_and_predict(model, X_train, split.y_train, X_test)
    proba = cost.pop("proba")[:, 1]

    metrics = compute_metrics(split.y_test, proba)
    predicted_bins, observed_freqs = reliability_curve(split.y_test, proba)

    best_threshold, min_expected_cost = find_optimal_threshold(
        split.y_test, proba, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    cost_at_naive_threshold = cost_weighted_score(
        split.y_test, proba, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )

    return {
        "dataset": dataset_config,
        "model": model_name,
        **metrics,
        **cost,
        "best_threshold": best_threshold,
        "min_expected_cost": min_expected_cost,
        "cost_at_naive_0.5_threshold": cost_at_naive_threshold,
        "calibration_predicted": predicted_bins.tolist(),
        "calibration_observed": observed_freqs.tolist(),
    }
