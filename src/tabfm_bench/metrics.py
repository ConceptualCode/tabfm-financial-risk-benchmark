"""Predictive-performance and calibration metrics.

Accuracy/AUC only tell you whether the model ranks risky vs. safe cases
correctly. Brier score and the reliability curve tell you whether a
predicted "0.73" actually behaves like a 73% chance in reality -- see
OBJECTIVES.md research question 2.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """y_proba: predicted probability of the positive class (shape (n,)).

    auc_roc/pr_auc/log_loss/brier_score are threshold-free (they operate on
    the raw probabilities). recall/f1_score need hard 0/1 decisions, so
    `threshold` converts y_proba -> y_pred first -- 0.5 is a neutral default,
    not a tuned operating point (see cost_weighted_score below for picking a
    threshold based on actual business costs instead).
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """Returns (predicted_bin_means, observed_frequencies) for a calibration plot."""
    observed, predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return predicted, observed


def cost_weighted_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_false_negative: float,
    cost_false_positive: float,
) -> float:
    """Expected monetary cost of using `threshold` to convert probabilities
    into accept/decline decisions. Lower is better.

    Framing predictions in cost terms (missed default vs. wrongly declined
    good customer) is how real credit teams judge models -- see
    OBJECTIVES.md research question 2/3 discussion.
    """
    y_pred = (y_proba >= threshold).astype(int)
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    return (
        false_negatives * cost_false_negative + false_positives * cost_false_positive
    ) / len(y_true)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
    n_thresholds: int = 99,
) -> tuple[float, float]:
    """Sweeps thresholds and returns (best_threshold, min_expected_cost).

    This is the actual production decision -- not "what's the accuracy,"
    but "at what threshold should you operate, and what does that cost."
    cost_false_negative/cost_false_positive are illustrative defaults (a 5:1
    ratio is used by run.py), not real business numbers -- replace with
    actual loss-given-default / cost-of-declined-good-customer figures
    before treating the resulting threshold as an actual recommendation.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    costs = [
        cost_weighted_score(y_true, y_proba, t, cost_false_negative, cost_false_positive)
        for t in thresholds
    ]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
