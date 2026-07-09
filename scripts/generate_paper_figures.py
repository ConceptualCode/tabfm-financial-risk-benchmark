"""Generates the static figures used in PAPER.md from results/*.csv.

Not part of the benchmark pipeline itself -- a one-off (rerunnable) step
that turns the raw result tables into the reliability, cost-scaling,
robustness, dataset-size, and fairness figures the paper references.

Usage:
    python scripts/generate_paper_figures.py
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Fixed categorical assignment -- same color for the same model in every
# figure in the paper. Palette: dataviz skill reference palette, slots 1/2/4/3/5.
COLORS = {
    "tabfm": "#2a78d6",     # blue
    "sap_rpt": "#1baf7a",   # aqua
    "lightgbm": "#008300",  # green
    "xgboost": "#eda100",   # yellow
    "logreg": "#4a3aa7",    # violet
}
LABELS = {
    "tabfm": "TabFM",
    "sap_rpt": "SAP-RPT",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "logreg": "Logistic regression",
}
MODEL_ORDER = ["tabfm", "sap_rpt", "lightgbm", "xgboost", "logreg"]

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


# ---------------------------------------------------------------------------
# Figure 1: RQ2 calibration reliability diagram (cd1 -- only dataset with
# all 5 models present)
# ---------------------------------------------------------------------------
def fig_calibration():
    df = pd.read_csv(RESULTS / "results.csv")
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
    fig.savefig(FIGURES / "fig_calibration_cd1.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: RQ3 cost-vs-width scaling (predict time vs. feature count)
# ---------------------------------------------------------------------------
N_FEATURES = {"cd1": 9, "cd2": 23, "ld1": 12, "ld2": 11, "ld3": 35,
              "cf1": 19, "cf2": 120, "cc1": 9, "cc2": 10, "cc3": 21}


def fig_cost_scaling():
    # Points are discrete datasets, not samples of one continuous function --
    # connecting them with a line would imply a trend between arbitrary
    # datasets. Scatter only, one marker per (model, dataset).
    df = pd.read_csv(RESULTS / "results.csv")
    df["n_features"] = df["dataset"].map(N_FEATURES)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for model in ["tabfm", "sap_rpt", "xgboost"]:
        sub = df[df["model"] == model].sort_values("n_features")
        ax.scatter(sub["n_features"], sub["predict_seconds"], s=64,
                    color=COLORS[model], label=LABELS[model], zorder=3)

    ax.set_yscale("log")
    style_axes(ax)
    ax.set_xlabel("Dataset width (number of features)")
    ax.set_ylabel("Prediction time, full test set (seconds, log scale)")
    ax.set_title("Inference cost scales with context width, not tree depth", loc="left", fontsize=12, color=INK)
    ax.annotate("cf2 (120 features):\nTabFM + SAP-RPT\nboth failed here", xy=(120, 0.018), xytext=(75, 3),
                fontsize=8.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cost_scaling.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: RQ7 missingness degradation curves (AUC and Brier, two panels)
# ---------------------------------------------------------------------------
def fig_robustness():
    df = pd.read_csv(RESULTS / "robustness.csv")
    agg = df.groupby(["model", "missing_rate"]).agg(
        auc_roc=("auc_roc", "mean"), brier_score=("brier_score", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for model in MODEL_ORDER:
        sub = agg[agg["model"] == model].sort_values("missing_rate")
        if sub.empty:
            continue
        axes[0].plot(sub["missing_rate"], sub["auc_roc"], marker="o", markersize=6,
                      linewidth=2, color=COLORS[model], label=LABELS[model], zorder=3)
        axes[1].plot(sub["missing_rate"], sub["brier_score"], marker="o", markersize=6,
                      linewidth=2, color=COLORS[model], label=LABELS[model], zorder=3)

    for ax, title, ylabel in [
        (axes[0], "AUC-ROC (higher = better)", "Mean AUC-ROC across datasets"),
        (axes[1], "Brier score (lower = better)", "Mean Brier score across datasets"),
    ]:
        style_axes(ax)
        ax.set_xlabel("Fraction of test-time features masked to missing")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12, color=INK)

    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Performance degradation under increasing test-time missingness", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "fig_robustness.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: RQ8 dataset-size sensitivity (AUC vs. total rows, log x)
# ---------------------------------------------------------------------------
N_TOTAL = {"cd1": 4043, "cd2": 27900, "ld1": 3128, "ld2": 26633, "ld3": 209708,
           "cf1": 7902, "cf2": 7999, "cc1": 6184, "cc2": 9300, "cc3": 6550}


def fig_dataset_size():
    # Scatter, not lines -- each dataset is a distinct task (different
    # difficulty, feature count, base rate), not a repeated sample of one
    # underlying function of size. A connecting line here would visually
    # claim a monotonic size effect that the AUC swings do not support.
    df = pd.read_csv(RESULTS / "results.csv")
    df["n_total"] = df["dataset"].map(N_TOTAL)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("n_total")
        ax.scatter(sub["n_total"], sub["auc_roc"], s=56,
                    color=COLORS[model], label=LABELS[model],
                    alpha=0.9, zorder=3)

    ax.set_xscale("log")
    style_axes(ax)
    ax.set_xlabel("Dataset size, train + test rows (log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC vs. dataset size, per model", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_dataset_size.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: RQ5 fairness -- disparate impact ratio by dataset x model
# ---------------------------------------------------------------------------
def fig_fairness():
    df = pd.read_csv(RESULTS / "results.csv")
    df = df.dropna(subset=["fairness_disparate_impact_ratio"])
    datasets = sorted(df["dataset"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    n_models = len(MODEL_ORDER)
    width = 0.15
    x = range(len(datasets))

    for i, model in enumerate(MODEL_ORDER):
        vals = []
        for d in datasets:
            row = df[(df["dataset"] == d) & (df["model"] == model)]
            vals.append(row["fairness_disparate_impact_ratio"].iloc[0] if not row.empty else None)
        xs = [xi + (i - n_models / 2) * width + width / 2 for xi in x]
        xs_plot = [xv for xv, v in zip(xs, vals) if v is not None]
        vals_plot = [v for v in vals if v is not None]
        ax.bar(xs_plot, vals_plot, width=width * 0.9, color=COLORS[model], label=LABELS[model], zorder=3)

    ax.axhline(0.8, color="#d03b3b", linewidth=1.5, linestyle="--", zorder=2)
    ax.text(-0.45, 0.815, "four-fifths threshold (0.8)", color="#d03b3b",
            fontsize=9, ha="left", va="bottom")

    style_axes(ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Disparate impact ratio (gender)")
    ax.set_title("Disparate impact ratio by model — below 0.8 flags EEOC concern", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fairness.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_calibration()
    fig_cost_scaling()
    fig_robustness()
    fig_dataset_size()
    fig_fairness()
    print(f"Wrote 5 figures to {FIGURES}")
