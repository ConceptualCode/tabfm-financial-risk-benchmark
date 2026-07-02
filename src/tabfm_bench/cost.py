"""Wall-clock / memory measurement for fit and inference.

Foundation-model in-context learning re-processes the training context on
every prediction call, unlike a GBM which pays its cost once during
training. This module measures whether that shifts total cost of ownership
-- see OBJECTIVES.md research question 3.
"""

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class CostReport:
    wall_seconds: float
    peak_memory_mb: float


@contextmanager
def measure():
    tracemalloc.start()
    start = time.perf_counter()
    report = CostReport(wall_seconds=0.0, peak_memory_mb=0.0)
    try:
        yield report
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report.wall_seconds = elapsed
        report.peak_memory_mb = peak / (1024 * 1024)


def measure_fit_and_predict(model, X_train, y_train, X_test) -> dict:
    """Runs fit then predict_proba, timing each phase separately so we can
    distinguish "train-time cost" from "inference-time cost" per model.
    """
    with measure() as fit_report:
        model.fit(X_train, y_train)

    with measure() as predict_report:
        proba = model.predict_proba(X_test)

    return {
        "fit_seconds": fit_report.wall_seconds,
        "fit_peak_memory_mb": fit_report.peak_memory_mb,
        "predict_seconds": predict_report.wall_seconds,
        "predict_peak_memory_mb": predict_report.peak_memory_mb,
        "proba": proba,
    }
