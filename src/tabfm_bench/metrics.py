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
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """y_proba: predicted probability of the positive class (shape (n,))."""
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
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
