"""Fair-lending style disparate-impact metrics.

TabFM is trained on synthetic SCM-generated data; SAP-RPT is trained on
scraped real-world tables (T4/TabLib). Whether those two very different
training corpora encode different bias patterns than a GBM trained fresh on
your own population is an untested question in the existing literature on
these two models -- see OBJECTIVES.md's fairness research question.
"""

import numpy as np


def disparate_impact_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, group_labels: np.ndarray
) -> dict:
    """Standard fair-lending metrics comparing outcomes across exactly two
    groups (e.g. "female"/"male").

    "Favorable" outcome here is y_pred == 0 (predicted not-risky/approved)
    -- these are default/fraud/churn risk models, so 1 means flagged/
    declined and 0 means approved.

    Returns per-group selection rates, the disparate impact ratio
    (unprivileged-group selection rate / privileged-group selection rate --
    the US EEOC "four-fifths rule" convention flags anything under 0.8 as a
    potential concern), and the equalized-odds gap (max absolute difference
    in true-positive rate and false-positive rate across groups -- 0 means
    the model errs equally often on both groups).

    The larger group is treated as "privileged" by convention -- not a
    claim about which group actually holds social privilege, just the
    standard reference-group choice these metrics use.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_labels = np.asarray(group_labels)
    groups = sorted(set(group_labels))
    if len(groups) != 2:
        raise ValueError(f"disparate_impact_metrics expects exactly 2 groups, got {groups}")

    privileged = max(groups, key=lambda g: np.sum(group_labels == g))
    unprivileged = [g for g in groups if g != privileged][0]

    def _selection_rate(group):
        mask = group_labels == group
        return float(np.mean(y_pred[mask] == 0))

    def _tpr_fpr(group):
        mask = group_labels == group
        y_t, y_p = y_true[mask], y_pred[mask]
        positives, negatives = y_t == 1, y_t == 0
        tpr = float(np.mean(y_p[positives] == 1)) if positives.any() else float("nan")
        fpr = float(np.mean(y_p[negatives] == 1)) if negatives.any() else float("nan")
        return tpr, fpr

    rate_priv = _selection_rate(privileged)
    rate_unpriv = _selection_rate(unprivileged)
    disparate_impact_ratio = rate_unpriv / rate_priv if rate_priv > 0 else float("nan")

    tpr_priv, fpr_priv = _tpr_fpr(privileged)
    tpr_unpriv, fpr_unpriv = _tpr_fpr(unprivileged)
    equalized_odds_gap = max(abs(tpr_priv - tpr_unpriv), abs(fpr_priv - fpr_unpriv))

    return {
        "privileged_group": str(privileged),
        "unprivileged_group": str(unprivileged),
        "selection_rate_privileged": rate_priv,
        "selection_rate_unprivileged": rate_unpriv,
        "disparate_impact_ratio": disparate_impact_ratio,
        "equalized_odds_gap": equalized_odds_gap,
    }
