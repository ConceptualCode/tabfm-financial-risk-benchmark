"""Generates the 4 TabFM-only figures embedded in ARTICLE_V2.md, for the
version of the article that removes SAP-RPT entirely. Reuses the same
palette/style as generate_paper_figures.py so all figures across both
documents stay visually consistent.

Classical-model (XGBoost/LightGBM/logreg) data comes from results_tuned.csv /
robustness_tuned.csv (tune_classical.py's real per-dataset hyperparameter
search), not the original fixed-default results.csv / robustness.csv --
TabFM is zero-shot and untouched by that search, so its rows still come from
the original files.

Usage:
    python scripts/generate_article_figures_tabfm_only.py
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "tabfm": "#2a78d6",
    "lightgbm": "#008300",
    "xgboost": "#eda100",
    "logreg": "#4a3aa7",
}
LABELS = {
    "tabfm": "TabFM",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "logreg": "Logistic regression",
}
MODEL_ORDER = ["tabfm", "lightgbm", "xgboost", "logreg"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def style_axes(ax):
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


N_FEATURES = {"cd1": 9, "cd2": 23, "ld1": 12, "ld2": 11, "ld3": 35,
              "cf1": 19, "cf2": 120, "cc1": 9, "cc2": 10, "cc3": 21}
N_TOTAL = {"cd1": 4043, "cd2": 27900, "ld1": 3128, "ld2": 26633, "ld3": 209708,
           "cf1": 7902, "cf2": 7999, "cc1": 6184, "cc2": 9300, "cc3": 6550}
DATASET_ORDER = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]


def _load_results():
    """TabFM from the original run (untouched by tuning); XGBoost/LightGBM/
    logreg from the tuned re-run. Concatenated so callers can treat it as one
    results table exactly like the pre-tuning scripts did.
    """
    orig = pd.read_csv(RESULTS / "results.csv")
    tabfm = orig[orig["model"] == "tabfm"]
    tuned = pd.read_csv(RESULTS / "results_tuned.csv")
    return pd.concat([tabfm, tuned], ignore_index=True)


def _load_robustness():
    orig = pd.read_csv(RESULTS / "robustness.csv")
    tabfm = orig[orig["model"] == "tabfm"]
    tuned = pd.read_csv(RESULTS / "robustness_tuned.csv")
    return pd.concat([tabfm, tuned], ignore_index=True)


def fig_calibration():
    df = _load_results()
    df = df[df["dataset"] == "cd1"]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=MUTED, zorder=1,
            label="Perfect calibration")
    for model in MODEL_ORDER:
        row = df[df["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        pred = ast.literal_eval(row["calibration_predicted"])
        obs = ast.literal_eval(row["calibration_observed"])
        ax.plot(pred, obs, marker="o", markersize=5, linewidth=2,
                 color=COLORS[model], label=LABELS[model], zorder=3)

    style_axes(ax)
    ax.set_xlim(-0.01, 0.35)
    ax.set_ylim(-0.01, 0.35)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed default rate (per bin)")
    ax.set_title("Calibration reliability diagram — cd1 (credit default)", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_calibration_cd1_tabfm.png", dpi=200)
    plt.close(fig)


def fig_cost_scaling():
    df = _load_results()
    df["n_cells"] = df["dataset"].map(N_TOTAL) * df["dataset"].map(N_FEATURES)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("n_cells")
        ax.scatter(sub["n_cells"], sub["predict_seconds"], s=64,
                    color=COLORS[model], label=LABELS[model], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax)
    ax.set_xlabel("Table size scored (rows × features, log scale)")
    ax.set_ylabel("Prediction time, full test set (seconds, log scale)")
    ax.set_title("Inference cost climbs with table size", loc="left", fontsize=12, color=INK)

    cf2_cells = N_TOTAL["cf2"] * N_FEATURES["cf2"]
    ax.annotate("cf2 (120 features):\nTabFM failed here",
                xy=(cf2_cells, 0.02), xytext=(cf2_cells * 0.09, 3.5),
                fontsize=8.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cost_scaling_tabfm.png", dpi=200)
    plt.close(fig)


def fig_dataset_size():
    df = _load_results()
    df["n_total"] = df["dataset"].map(N_TOTAL)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("n_total")
        ax.scatter(sub["n_total"], sub["auc_roc"], s=56,
                    color=COLORS[model], label=LABELS[model], alpha=0.9, zorder=3)

    ax.set_xscale("log")
    style_axes(ax)
    ax.set_xlabel("Dataset size, train + test rows (log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC vs. dataset size, per model", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_dataset_size_tabfm.png", dpi=200)
    plt.close(fig)


def fig_robustness_trajectory():
    df = _load_robustness()

    RATES = [0.0, 0.05, 0.2, 0.5]
    RATE_LABELS = ["0%", "5%", "20%", "50%"]
    RATE_POS = {r: i for i, r in enumerate(RATES)}

    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub_d = df[df["dataset"] == dataset]
        for model in MODEL_ORDER:
            sub = sub_d[sub_d["model"] == model].sort_values("missing_rate")
            if sub.empty:
                continue
            xpos = sub["missing_rate"].map(RATE_POS)
            ax.plot(xpos, sub["auc_roc"], marker="o", markersize=4, linewidth=1.8,
                    color=COLORS[model], zorder=3)
        for i in range(len(RATES)):
            ax.axvline(i, color=GRID, linewidth=0.8, zorder=1)
        style_axes(ax)
        ax.set_title(dataset, loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(RATES)))
        ax.set_xticklabels(RATE_LABELS)
        ax.set_xlim(-0.3, len(RATES) - 0.7)

    for ax in axes[5:]:
        ax.set_xlabel("Feature values missing at test time")
    axes[0].set_ylabel("AUC-ROC")
    axes[5].set_ylabel("AUC-ROC")

    handles = [plt.Line2D([0], [0], color=COLORS[m], linewidth=2, marker="o", markersize=4)
               for m in MODEL_ORDER]
    fig.legend(handles, [LABELS[m] for m in MODEL_ORDER], loc="upper center",
               ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("AUC-ROC at each tested missingness level, by dataset — 0%, 5%, 20%, 50% features missing",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIGURES / "fig_robustness_trajectory_tabfm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_calibration()
    fig_cost_scaling()
    fig_dataset_size()
    fig_robustness_trajectory()
    print(f"Wrote 4 TabFM-only figures to {FIGURES}")
