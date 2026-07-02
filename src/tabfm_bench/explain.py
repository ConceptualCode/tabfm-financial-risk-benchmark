"""SHAP comparison across model types.

Tree models (XGBoost/LightGBM) get SHAP's fast exact TreeExplainer.
TabFM and logistic regression have no such shortcut, so they fall back to
model-agnostic KernelExplainer/PermutationExplainer, which is slower and
relies on approximations. Whether those approximations hold up on a
24-block causal transformer doing in-context learning is an open question
-- see OBJECTIVES.md research question 4.
"""

import numpy as np
import shap


def get_shap_values(model, model_name: str, X_background: np.ndarray, X_explain: np.ndarray):
    """Returns a (n_explain, n_features) array of SHAP values for the
    positive class.
    """
    if model_name in ("xgboost", "lightgbm"):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_explain)
        if isinstance(values, list):
            values = values[1]
        return values

    predict_fn = lambda X: model.predict_proba(X)[:, 1]
    background = shap.sample(X_background, min(100, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background)
    return explainer.shap_values(X_explain, nsamples="auto")


def compare_shap_agreement(shap_a: np.ndarray, shap_b: np.ndarray) -> float:
    """Rank correlation between two models' per-sample feature-importance
    orderings, averaged over samples. A quick, cheap signal for "do these
    two models agree on why a prediction happened."
    """
    from scipy.stats import spearmanr

    correlations = [
        spearmanr(shap_a[i], shap_b[i]).correlation for i in range(len(shap_a))
    ]
    return float(np.nanmean(correlations))
