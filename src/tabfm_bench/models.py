"""Common fit/predict_proba interface across TabFM and the baselines.

Every model in MODEL_REGISTRY exposes:
    model.fit(X_train, y_train)
    model.predict_proba(X_test) -> (n_samples, 2) array
so the eval harness (run.py) can treat them interchangeably.
"""

import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_tabfm():
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    model = tabfm_v1_0_0.load(model_type="classification")
    return TabFMClassifier(model=model)


def build_sap_rpt(bagging=None, max_context_size=None):
    """SAP's zero-shot tabular in-context-learning model (formerly ConTextTab).

    Gated on HuggingFace -- requires requesting access at
    https://huggingface.co/SAP/sap-rpt-1-oss and authenticating
    (`huggingface-cli login` or HF_TOKEN) before this will load weights.

    Default `bagging`/`max_context_size` (8 / 8192) match the model card's
    recommended settings, which is also what drives its ~80GB GPU
    recommendation (that figure is about inference-time activation memory
    across bagged passes, not checkpoint size -- the weights themselves are
    only ~65MB). Reduce both for constrained hardware, at some cost to
    robustness -- itself a relevant data point for RQ3 (real inference cost
    of in-context learning).

    Overridable via SAP_RPT_BAGGING / SAP_RPT_MAX_CONTEXT_SIZE env vars so
    scripts (e.g. the Colab notebook) can constrain memory use without a
    code change.
    """
    from sap_rpt_oss import SAP_RPT_OSS_Classifier

    if bagging is None:
        bagging = int(os.environ.get("SAP_RPT_BAGGING", 8))
    if max_context_size is None:
        max_context_size = int(os.environ.get("SAP_RPT_MAX_CONTEXT_SIZE", 8192))

    return SAP_RPT_OSS_Classifier(bagging=bagging, max_context_size=max_context_size)


def build_xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        n_jobs=-1,
    )


def build_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


MODEL_REGISTRY = {
    "tabfm": build_tabfm,
    "sap_rpt": build_sap_rpt,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "logreg": build_logreg,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
