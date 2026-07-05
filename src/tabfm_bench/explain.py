"""SHAP comparison across model types.

Tree models (XGBoost/LightGBM) get SHAP's fast exact TreeExplainer.
TabFM and logistic regression have no such shortcut, so they fall back to
model-agnostic KernelExplainer/PermutationExplainer, which is slower and
relies on approximations. Whether those approximations hold up on a
24-block causal transformer doing in-context learning is an open question
-- see OBJECTIVES.md research question 4.
"""

import numpy as np
import pandas as pd
import shap


def get_shap_values(model, model_name: str, X_background, X_explain):
    """Returns a (n_explain, n_features) array of SHAP values for the
    positive class.
    """
    if model_name in ("xgboost", "lightgbm"):
        # LightGBM is wrapped in _CategoricalAwareLGBM (models.py); SHAP's
        # TreeExplainer needs the real LGBMClassifier, not the wrapper.
        tree_model = getattr(model, "_model", model)
        explainer = shap.TreeExplainer(tree_model)
        values = explainer.shap_values(X_explain)
        if isinstance(values, list):
            values = values[1]
        return values

    predict_fn = lambda X: model.predict_proba(X)[:, 1]
    if isinstance(X_background, pd.DataFrame):
        # KernelExplainer perturbs data by converting it to a single
        # dtype=object numpy array internally, then hands that raw array
        # to predict_fn -- losing the per-column float64/string typing
        # TabFM/SAP-RPT expect (raw-input models, see RAW_INPUT_MODELS in
        # models.py). That's what causes "dtype is object, but first
        # non-null value is <class 'float'>. Converting to str." inside
        # their own type inference during the SHAP step specifically (not
        # during ordinary fit/predict, which uses the DataFrame directly).
        # Rebuilding a properly-typed DataFrame right before calling the
        # model fixes it.
        columns = X_background.columns
        dtypes = X_background.dtypes

        def predict_fn(X, _columns=columns, _dtypes=dtypes):
            df = pd.DataFrame(X, columns=_columns).astype(_dtypes.to_dict())
            return model.predict_proba(df)[:, 1]

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
