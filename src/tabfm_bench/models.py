"""Common fit/predict_proba interface across TabFM and the baselines.

Every model in MODEL_REGISTRY exposes:
    model.fit(X_train, y_train)
    model.predict_proba(X_test) -> (n_samples, 2) array
so the eval harness (run.py) can treat them interchangeably.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_tabfm():
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    model = tabfm_v1_0_0.load(model_type="classification")
    return TabFMClassifier(model=model)


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
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "logreg": build_logreg,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
