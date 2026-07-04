"""Cross-model prediction agreement.

Two models can report near-identical AUC while disagreeing on individual
decisions -- a real production risk that model-vs-GBM benchmarks (the
existing SAP-RPT and TabFM literature) don't surface, since none of them
compare foundation models against each other. Whether TabFM and SAP-RPT
(trained on completely different data -- synthetic SCM vs. scraped
real-world tables) actually agree on individual applicants, not just
aggregate accuracy, is the question this module answers. See
OBJECTIVES.md's cross-model agreement research question.
"""

import numpy as np
from scipy.stats import pearsonr


def compare_predictions(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compares two models' predictions on the same test set.

    proba_correlation: do the two models' risk scores move together, even
    if their absolute values differ.
    hard_decision_agreement_rate: fraction of applicants both models would
    flag/approve the same way at `threshold`.
    accuracy_when_agree / accuracy_when_disagree: whether agreement between
    models is itself a useful confidence signal -- if accuracy is much
    higher on the "both models agree" subset, that agreement is worth
    surfacing operationally (e.g. auto-decide when models agree, route
    disagreements to human review).
    """
    y_true = np.asarray(y_true)
    pred_a = (np.asarray(proba_a) >= threshold).astype(int)
    pred_b = (np.asarray(proba_b) >= threshold).astype(int)
    agree = pred_a == pred_b

    def _accuracy(mask):
        if not mask.any():
            return float("nan")
        return float(np.mean(pred_a[mask] == y_true[mask]))

    return {
        "proba_correlation": float(pearsonr(proba_a, proba_b)[0]),
        "hard_decision_agreement_rate": float(np.mean(agree)),
        "accuracy_when_agree": _accuracy(agree),
        "accuracy_when_disagree": _accuracy(~agree),
    }
