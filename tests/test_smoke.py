"""Smoke tests that don't require network access or heavy deps to pass."""

import numpy as np

from tabfm_bench.agreement import compare_predictions
from tabfm_bench.fairness import disparate_impact_metrics
from tabfm_bench.metrics import (
    compute_metrics,
    cost_weighted_score,
    find_optimal_threshold,
    reliability_curve,
)


def test_compute_metrics_runs():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
    metrics = compute_metrics(y_true, y_proba)
    assert set(metrics) == {
        "auc_roc", "pr_auc", "log_loss", "brier_score", "recall", "f1_score",
    }
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


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


def test_find_optimal_threshold_zero_cost_when_perfectly_separable():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.05, 0.95, 0.1, 0.9])
    best_threshold, min_cost = find_optimal_threshold(
        y_true, y_proba, cost_false_negative=5.0, cost_false_positive=1.0
    )
    assert 0.0 <= best_threshold <= 1.0
    assert min_cost == 0.0


def test_disparate_impact_ratio_is_one_when_groups_treated_identically():
    # Perfect predictions, split evenly (2 pos/2 neg) within each group ->
    # both groups get identical selection rate, TPR, and FPR.
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    group_labels = np.array(["female"] * 4 + ["male"] * 4)
    result = disparate_impact_metrics(y_true, y_pred, group_labels)
    assert result["disparate_impact_ratio"] == 1.0
    assert result["equalized_odds_gap"] == 0.0


def test_disparate_impact_rejects_more_than_two_groups():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    group_labels = np.array(["a", "b", "c", "a"])
    try:
        disparate_impact_metrics(y_true, y_pred, group_labels)
        assert False, "expected ValueError for 3 groups"
    except ValueError:
        pass


def test_compare_predictions_full_agreement():
    y_true = np.array([0, 1, 0, 1])
    proba_a = np.array([0.1, 0.9, 0.2, 0.8])
    proba_b = np.array([0.05, 0.95, 0.15, 0.85])
    result = compare_predictions(y_true, proba_a, proba_b)
    assert result["hard_decision_agreement_rate"] == 1.0
    assert result["accuracy_when_agree"] == 1.0
    assert np.isnan(result["accuracy_when_disagree"])
