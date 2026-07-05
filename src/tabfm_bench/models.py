"""Common fit/predict_proba interface across TabFM and the baselines.

Every model in MODEL_REGISTRY exposes:
    model.fit(X_train, y_train)
    model.predict_proba(X_test) -> (n_samples, 2) array
so the eval harness (run.py) can treat them interchangeably.

Each model gets realistic, production-grade preprocessing for what it
actually is -- not a strawman, not an unfair advantage:
  - TabFM/SAP-RPT (RAW_INPUT_MODELS) are documented to accept raw tables
    (real column names, real category values) and handle mixed types
    internally -- run.py feeds them FinBenchSplit.X_{split}_df accordingly.
  - LightGBM gets its categorical columns flagged via `categorical_feature`
    so it can use native categorical splits instead of treating a
    label-encoded int as ordinal.
  - Logistic regression one-hot encodes categoricals (a linear model
    shouldn't be handed an arbitrary integer code and asked to fit a
    coefficient to it) and standardizes numerics.
  - XGBoost is left on the plain pre-encoded array -- native categorical
    support in XGBoost requires pandas `category` dtype columns
    (`enable_categorical=True`), which is a larger change than this pass
    covers; treating the encoded ints as numeric is a common, accepted
    simplification for tree models but is a known limitation here.

Random seeds are fixed on every trainable baseline (`random_state=42`) so
results are reproducible run to run -- important both for basic scientific
hygiene and because non-determinism in a credit decision is a real
production/compliance concern, not just an engineering nuisance.
"""

import os

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_INPUT_MODELS = {"tabfm", "sap_rpt"}

SEED = 42


TABFM_HF_REPO = "google/tabfm-1.0.0-pytorch"


def build_tabfm(cat_idx=None, num_idx=None, col_name=None, n_estimators=None):
    """Loads TabFM by going straight to the real checkpoint file.

    tabfm==1.0.0's own tabfm_v1_0_0.load() hardcodes looking for
    "<model_type>/pytorch_model.bin" after a snapshot_download, but the
    actual HF repo (google/tabfm-1.0.0-pytorch) only ships
    "<model_type>/model.safetensors" -- confirmed against the HF Hub API's
    file listing. Calling load() raises FileNotFoundError every time,
    regardless of hardware. This replicates load()'s working logic (build
    the model from ClassificationConfig, load a state dict, .eval()) but
    points at the file that actually exists, using safetensors instead of
    torch.load's pickle format.

    Loads via a "meta device + assign=True" pattern rather than the
    straightforward construct-then-copy path: building the model under
    `torch.device("meta")` allocates its structure with no real memory,
    then `load_state_dict(..., assign=True)` replaces its parameters with
    the loaded tensors directly (loaded straight onto the target device)
    instead of copying data into a separately-materialized, full-size CPU
    model first. Measured directly: the old construct-then-copy path used
    ~6.4GB of RSS; this path used ~176MB -- the difference between a
    machine with a real GPU but limited system RAM being able to run this
    at all, or not (this is the exact fix for the 8GB-GPU/<12GB-RAM laptop
    scenario). Falls back to the straightforward path if meta-device
    construction doesn't work cleanly with TabFM's specific module
    internals -- untested on real GPU hardware, since this machine has none.

    Explicitly moves the model to CUDA if available -- load()'s original
    device handling was never wired up when this was written, which left
    TabFM silently running on CPU even with a GPU present (a 24-block
    transformer forward pass over the training context on CPU took ~19
    minutes on a single small dataset in testing; SAP-RPT, by contrast,
    auto-detects and uses CUDA internally, so it wasn't the bottleneck).

    n_estimators defaults to TabFMClassifier's own default (32) -- every
    prediction runs that many ensemble members through the model, not one,
    which is what actually dominates TabFM's ~4 minute-per-dataset cost on
    a T4 GPU (the one-time 6.5GB checkpoint download is separate and only
    happens once per session). Overridable via TABFM_N_ESTIMATORS for
    faster validation runs, at the cost of ensemble robustness -- the same
    tradeoff SAP_RPT_BAGGING already exposes for SAP-RPT.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from tabfm import TabFMClassifier
    from tabfm.src.pytorch.tabfm_v1_0_0 import ClassificationConfig, TabFM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = hf_hub_download(
        repo_id=TABFM_HF_REPO, filename="classification/model.safetensors"
    )

    try:
        with torch.device("meta"):
            model = TabFM(**ClassificationConfig().to_dict())
        state_dict = load_file(checkpoint_path, device=str(device))
        model.load_state_dict(state_dict, strict=True, assign=True)
    except Exception as e:
        print(f"[tabfm] meta-device load failed ({e}); falling back to construct-then-copy")
        model = TabFM(**ClassificationConfig().to_dict())
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

    model.eval()
    print(f"[tabfm] using device: {device}")

    if n_estimators is None:
        n_estimators = int(os.environ.get("TABFM_N_ESTIMATORS", 32))

    return TabFMClassifier(model=model, n_estimators=n_estimators)


def build_sap_rpt(
    cat_idx=None, num_idx=None, col_name=None, bagging=None, max_context_size=None
):
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

    classifier = SAP_RPT_OSS_Classifier(bagging=bagging, max_context_size=max_context_size)
    print(f"[sap_rpt] using device: {classifier.device}")
    return classifier


def build_xgboost(cat_idx=None, num_idx=None, col_name=None):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )


class _CategoricalAwareLGBM:
    """Wraps LGBMClassifier so categorical columns get LightGBM's native
    categorical split handling (`categorical_feature=`) instead of being
    silently treated as ordinal integers.
    """

    def __init__(self, model, cat_idx):
        self._model = model
        self._cat_idx = cat_idx or []

    def fit(self, X, y):
        cat_feature = self._cat_idx if self._cat_idx else "auto"
        self._model.fit(X, y, categorical_feature=cat_feature)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def build_lightgbm(cat_idx=None, num_idx=None, col_name=None):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=SEED,
    )
    return _CategoricalAwareLGBM(model, cat_idx)


def build_logreg(cat_idx=None, num_idx=None, col_name=None):
    cat_idx = cat_idx or []
    num_idx = num_idx or []
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
            ("num", StandardScaler(), num_idx),
        ]
    )
    return make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=SEED))


MODEL_REGISTRY = {
    "tabfm": build_tabfm,
    "sap_rpt": build_sap_rpt,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "logreg": build_logreg,
}


def get_model(name: str, *, cat_idx=None, num_idx=None, col_name=None, **kwargs):
    """**kwargs forwards model-specific overrides (e.g. n_estimators for
    tabfm, bagging/max_context_size for sap_rpt) straight to the builder --
    used by scripts/run_benchmark.py to request a smaller, cheaper instance
    specifically for the SHAP step, separate from the full-settings
    instance used for the real metrics.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cat_idx=cat_idx, num_idx=num_idx, col_name=col_name, **kwargs)
