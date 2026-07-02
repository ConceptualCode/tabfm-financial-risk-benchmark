"""Runs one (dataset, model) pair end to end and returns a result dict
combining predictive metrics, calibration, and cost.

Explainability (SHAP) is run separately in scripts/run_benchmark.py since
it's slow and only needs to run once per (dataset, model) after the model
is already fit.
"""

from tabfm_bench.cost import measure_fit_and_predict
from tabfm_bench.data import load_finbench
from tabfm_bench.metrics import compute_metrics, reliability_curve
from tabfm_bench.models import get_model


def run_single(dataset_config: str, model_name: str) -> dict:
    split = load_finbench(dataset_config)
    model = get_model(model_name)

    cost = measure_fit_and_predict(model, split.X_train, split.y_train, split.X_test)
    proba = cost.pop("proba")[:, 1]

    metrics = compute_metrics(split.y_test, proba)
    predicted_bins, observed_freqs = reliability_curve(split.y_test, proba)

    return {
        "dataset": dataset_config,
        "model": model_name,
        **metrics,
        **cost,
        "calibration_predicted": predicted_bins.tolist(),
        "calibration_observed": observed_freqs.tolist(),
    }
