"""Missing-data robustness: how gracefully does each model degrade as an
increasing fraction of test-time features go missing?

Real applicants/transactions in production often have incomplete profiles
-- a field wasn't collected, a system didn't report a value in time.
TabFM/SAP-RPT both advertise automatic handling of missing data as a
built-in feature; XGBoost/LightGBM also handle missing values natively.
This tests that claim directly, at prediction time only -- training data
stays intact, since the scenario being simulated is "a subset of applicants
show up with incomplete data at scoring time," not a data-quality problem
baked into training. See OBJECTIVES.md RQ7.
"""

import numpy as np
import pandas as pd


def inject_missingness(X, missing_rate: float, seed: int = 42):
    """Returns a copy of X with `missing_rate` fraction of cells set to
    missing (NaN), chosen uniformly at random across all cells.

    Works for both the pre-encoded numpy array (classical baselines) and
    the raw DataFrame (TabFM/SAP-RPT) -- same random mask logic either way,
    just applied through the appropriate indexer.
    """
    if missing_rate <= 0:
        return X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

    rng = np.random.default_rng(seed)
    n_rows, n_cols = X.shape
    mask = rng.random((n_rows, n_cols)) < missing_rate

    if isinstance(X, pd.DataFrame):
        X_masked = X.copy()
        for j, col in enumerate(X_masked.columns):
            X_masked.iloc[mask[:, j], j] = np.nan
        return X_masked

    X_masked = np.array(X, dtype=float, copy=True)
    X_masked[mask] = np.nan
    return X_masked
