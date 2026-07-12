"""Renders the predictive-performance (AUC-ROC) table from ARTICLE_V2.md as a
standalone PNG, styled to match the other article figures (same palette/
fonts as generate_article_figures_tabfm_only.py). LinkedIn's article editor
doesn't render Markdown tables, so this is meant to be uploaded as an image
directly.

Usage:
    python scripts/generate_table_predictive_performance.py
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

COLUMNS = ["Dataset", "Features", "XGBoost", "LightGBM", "Logistic Regression", "TabFM"]
COL_MODEL_KEY = [None, None, "xgboost", "lightgbm", "logreg", "tabfm"]

ROWS = [
    ("Credit Default 1 (cd1)", 9, 0.740, 0.741, 0.740, 0.761),
    ("Credit Default 2 (cd2)", 23, 0.779, 0.778, 0.762, None),
    ("Loan Default 1 (ld1)", 12, 0.909, 0.912, 0.813, 0.998),
    ("Loan Default 2 (ld2)", 11, 0.947, 0.948, 0.867, None),
    ("Loan Default 3 (ld3)", 35, 0.665, 0.664, 0.635, None),
    ("Credit Fraud 1 (cf1)", 19, 0.907, 0.946, 0.960, 0.928),
    ("Credit Fraud 2 (cf2)", 120, 0.717, 0.730, 0.752, None),
    ("Customer Churn 1 (cc1)", 9, 0.655, 0.659, 0.662, 0.665),
    ("Customer Churn 2 (cc2)", 10, 0.864, 0.870, 0.774, 0.878),
    ("Customer Churn 3 (cc3)", 21, 0.863, 0.867, 0.857, 0.872),
]


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def main():
    n_rows = len(ROWS) + 1  # + header
    n_cols = len(COLUMNS)

    fig, ax = plt.subplots(figsize=(11, 0.52 * n_rows + 0.6))
    ax.axis("off")
    ax.set_title("Predictive performance (AUC-ROC) — all 10 datasets", loc="left",
                  fontsize=13, color=INK, pad=14)

    cell_text = [COLUMNS]
    for name, feats, xgb, lgbm, logreg, tabfm in ROWS:
        cell_text.append([name, str(feats), fmt(xgb), fmt(lgbm), fmt(logreg), fmt(tabfm)])

    table = ax.table(cellText=cell_text, cellLoc="center", loc="center",
                      colWidths=[0.30, 0.11, 0.13, 0.13, 0.20, 0.13])
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
            cell.PAD = 0.02
        elif c == 1:
            cell.set_text_props(color=MUTED)

    # Highlight the best (max) score per row among the 4 model columns.
    for r, (name, feats, xgb, lgbm, logreg, tabfm) in enumerate(ROWS, start=1):
        values = {"xgboost": xgb, "lightgbm": lgbm, "logreg": logreg, "tabfm": tabfm}
        present = {k: v for k, v in values.items() if v is not None}
        best_model = max(present, key=present.get)
        best_col = COL_MODEL_KEY.index(best_model)
        cell = table[r, best_col]
        cell.set_facecolor(WIN_TINT)
        cell.set_text_props(weight="bold", color=COLORS[best_model])

    fig.tight_layout()
    out = FIGURES / "table_predictive_performance.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
