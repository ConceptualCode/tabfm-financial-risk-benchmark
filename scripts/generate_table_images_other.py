"""Renders the calibration, inference-cost, fairness, and robustness tables
from ARTICLE_V2.md as standalone PNGs, styled to match
generate_table_predictive_performance.py and the other article figures.
LinkedIn's article editor doesn't render Markdown tables, so these are
meant to be uploaded as images directly.

Highlighting picks the best value per row among the 4 model columns, and
highlights ALL models tied for best -- two rows in this article have a real
tie (inference cost cc1: LightGBM/logreg both 0.011s; robustness cc1:
LightGBM/logreg/TabFM all -8%), and highlighting only one winner there would
misrepresent the data.

Usage:
    python scripts/generate_table_images_other.py
"""

import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "tabfm": "#2a78d6",
    "lightgbm": "#008300",
    "xgboost": "#eda100",
    "logreg": "#4a3aa7",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
WIN_TINT = "#eaf2fc"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
})

MODEL_KEYS = ["xgboost", "lightgbm", "logreg", "tabfm"]


def render_table(title, subtitle, columns, model_col_start, rows, value_fmt,
                  higher_is_better, out_name, col_widths, muted_cols=()):
    """rows: list of (label_cols..., raw_values) where raw_values is a
    4-tuple (xgb, lgbm, logreg, tabfm), each a float or None. label_cols is
    everything before the model columns, already display-ready strings.
    """
    n_rows = len(rows) + 1
    fig, ax = plt.subplots(figsize=(11, 0.5 * n_rows + 0.6))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=14)

    cell_text = [columns]
    for *label_cols, raw in rows:
        formatted = [fmt for fmt in label_cols]
        for v in raw:
            formatted.append("—" if v is None else value_fmt(v))
        cell_text.append(formatted)

    table = ax.table(cellText=cell_text, cellLoc="center", loc="center",
                      colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color=SURFACE, weight="bold")
            continue
        cell.set_facecolor(SURFACE)
        if c == 0:
            cell.set_text_props(ha="left", color=INK)
        elif c in muted_cols:
            cell.set_text_props(color=MUTED)

    for r, (*label_cols, raw) in enumerate(rows, start=1):
        present = {MODEL_KEYS[i]: v for i, v in enumerate(raw) if v is not None}
        if not present:
            continue
        best_val = max(present.values()) if higher_is_better else min(present.values())
        winners = [k for k, v in present.items() if v == best_val]
        for model in winners:
            col = model_col_start + MODEL_KEYS.index(model)
            cell = table[r, col]
            cell.set_facecolor(WIN_TINT)
            cell.set_text_props(weight="bold", color=COLORS[model])

    if subtitle:
        fig.text(0.01, -0.02 / n_rows, subtitle, fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout()
    out = FIGURES / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_calibration():
    rows = [
        ("cd1", (0.024, 0.021, 0.011, 0.014)),
        ("cd2", (0.010, 0.012, 0.019, None)),
        ("ld1", (0.028, 0.041, 0.020, 0.015)),
        ("ld2", (0.009, 0.016, 0.021, None)),
        ("ld3", (0.006, 0.007, 0.009, None)),
        ("cf1", (0.004, 0.005, 0.003, 0.002)),
        ("cf2", (0.037, 0.016, 0.012, None)),
        ("cc1", (0.036, 0.023, 0.025, 0.021)),
        ("cc2", (0.019, 0.017, 0.025, 0.011)),
        ("cc3", (0.021, 0.017, 0.027, 0.020)),
    ]
    render_table(
        title="Calibration error — all 10 datasets",
        subtitle="Mean calibration error, lower is better. Dash = TabFM not evaluated within this evaluation's GPU budget.",
        columns=["Dataset", "XGBoost", "LightGBM", "Logistic Regression", "TabFM"],
        model_col_start=1,
        rows=rows,
        value_fmt=lambda v: f"{v:.3f}",
        higher_is_better=False,
        out_name="table_calibration.png",
        col_widths=[0.20, 0.20, 0.20, 0.24, 0.16],
    )


def fig_inference_cost():
    rows = [
        ("cd1", "4,043", "9", (0.008, 0.009, 0.007, 183)),
        ("cd2", "27,900", "23", (0.015, 0.079, 0.034, None)),
        ("ld1", "3,128", "12", (0.005, 0.014, 0.007, 156)),
        ("ld2", "26,633", "11", (0.026, 0.244, 0.019, None)),
        ("ld3", "209,708", "35", (0.464, 0.899, 0.574, None)),
        ("cf1", "7,902", "19", (0.003, 0.021, 0.010, 515)),
        ("cf2", "7,999", "120", (0.024, 0.028, 0.030, None)),
        ("cc1", "6,184", "9", (0.035, 0.011, 0.011, 339)),
        ("cc2", "9,300", "10", (0.011, 0.033, 0.009, 580)),
        ("cc3", "6,550", "21", (0.009, 0.030, 0.015, 400)),
    ]

    def fmt_seconds(v):
        return f"{v:.0f}s" if v >= 1 else f"{v:.3f}s"

    render_table(
        title="Inference cost — predict time on full test set, all 10 datasets",
        subtitle="Lower is better. Dash = TabFM exhausted available GPU memory on that dataset.",
        columns=["Dataset", "Rows", "Features", "XGBoost", "LightGBM", "Logistic Regression", "TabFM"],
        model_col_start=3,
        rows=rows,
        value_fmt=fmt_seconds,
        higher_is_better=False,
        out_name="table_inference_cost.png",
        col_widths=[0.13, 0.13, 0.11, 0.14, 0.14, 0.20, 0.15],
        muted_cols=(1, 2),
    )


def fig_fairness():
    rows = [
        ("cd1", (0.075, 0.169, 0.237, 0.296)),
        ("cd2", (0.063, 0.072, 0.167, None)),
        ("cf2", (0.116, 0.081, 0.008, None)),
        ("cc1", (0.085, 0.084, 0.202, 0.193)),
        ("cc2", (0.078, 0.066, 0.218, 0.085)),
        ("cc3", (0.059, 0.036, 0.034, 0.058)),
    ]
    render_table(
        title="Fairness — equalized-odds gap, 6 datasets with a clean binary protected attribute",
        subtitle="Lower is better. Dash = TabFM not evaluated within this evaluation's GPU budget.",
        columns=["Dataset", "XGBoost", "LightGBM", "Logistic Regression", "TabFM"],
        model_col_start=1,
        rows=rows,
        value_fmt=lambda v: f"{v:.3f}",
        higher_is_better=False,
        out_name="table_fairness.png",
        col_widths=[0.20, 0.20, 0.20, 0.24, 0.16],
    )


def fig_robustness():
    rows = [
        ("cd1", (-18, -4, -2, -9)),
        ("cd2", (-15, -16, -9, None)),
        ("ld1", (-30, -21, -11, -25)),
        ("ld2", (-26, -21, -14, None)),
        ("ld3", (-17, -9, -7, None)),
        ("cf1", (-2, -10, -8, 2)),
        ("cf2", (-18, -15, -13, None)),
        ("cc1", (-11, -8, -8, -8)),
        ("cc2", (-27, -16, -10, -20)),
        ("cc3", (-21, -16, -8, -10)),
    ]
    render_table(
        title="Robustness — change in AUC-ROC, 0% to 50% missing feature values",
        subtitle="Higher (less negative) is better. Dash = TabFM not evaluated within this evaluation's GPU budget.",
        columns=["Dataset", "XGBoost", "LightGBM", "Logistic Regression", "TabFM"],
        model_col_start=1,
        rows=rows,
        value_fmt=lambda v: f"{v:+.0f}%",
        higher_is_better=True,
        out_name="table_robustness.png",
        col_widths=[0.20, 0.20, 0.20, 0.24, 0.16],
    )


if __name__ == "__main__":
    fig_calibration()
    fig_inference_cost()
    fig_fairness()
    fig_robustness()
