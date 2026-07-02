"""Smoke tests that don't require network access or heavy deps to pass."""

import numpy as np

from tabfm_bench.metrics import compute_metrics, cost_weighted_score, reliability_curve


def test_compute_metrics_runs():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
    metrics = compute_metrics(y_true, y_proba)
    assert set(metrics) == {"auc_roc", "pr_auc", "log_loss", "brier_score"}
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_reliability_curve_shapes_match():
    y_true = np.random.randint(0, 2, size=200)
    y_proba = np.random.uniform(0, 1, size=200)
    predicted, observed = reliability_curve(y_true, y_proba, n_bins=5)
    assert len(predicted) == len(observed)


def test_cost_weighted_score_zero_when_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])
    score = cost_weighted_score(
        y_true, y_proba, threshold=0.5, cost_false_negative=10, cost_false_positive=1
    )
    assert score == 0.0
