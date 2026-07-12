"""Generates a 1200x627 LinkedIn article cover image: the article title plus
a 7-item status scorecard across the evaluation's criteria, using the
dataviz skill's status palette (good/warning/critical). Each chip pairs an
icon with the status color rather than relying on color alone, since two of
the three status hues fall under 3:1 contrast on the light surface.

Statuses reflect the article's own findings (see ARTICLE_V2.md, "What this
means for a production decision" and the Findings section) -- not a new
judgment call made for this image.

Usage:
    python scripts/generate_cover_image.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
ACCENT = "#2a78d6"  # TabFM's series color elsewhere in the article's figures

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

# (label, status, icon) -- status per the article's own findings.
ROWS = [
    ("Accuracy", "good", "✓"),
    ("Calibration", "good", "✓"),
    ("Inference cost", "critical", "✗"),
    ("Explainability", "critical", "✗"),
    ("Fairness", "warning", "~"),
    ("Robustness", "warning", "~"),
    ("Licensing", "critical", "✗"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
})


def main():
    fig = plt.figure(figsize=(12, 6.27), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    # Left accent bar, ties the banner to the article's own figure palette
    ax.add_patch(Rectangle((0, 0), 0.012, 1, transform=ax.transAxes,
                            facecolor=ACCENT, linewidth=0, zorder=4))

    left_margin = 0.075

    # Title block
    ax.text(left_margin, 0.86, "Google's TabFM vs.\nTuned Classical ML Models",
             fontsize=33, fontweight="bold", color=INK, va="top", ha="left",
             linespacing=1.22)
    ax.text(left_margin, 0.52, "A Production-Readiness Evaluation on\n10 Financial Risk Datasets",
             fontsize=15.5, color=SECONDARY, va="top", ha="left", linespacing=1.4)

    # Divider rule between title block and scorecard
    ax.add_patch(Rectangle((left_margin, 0.365), 0.97 - left_margin, 0.0035,
                            transform=ax.transAxes, facecolor="#e1e0d9",
                            linewidth=0, zorder=2))

    ax.text(left_margin, 0.325, "WHERE IT WINS, AND WHERE IT DOESN'T",
             fontsize=10.5, color=MUTED, va="top", ha="left", fontweight="bold")

    # Scorecard row -- filled tinted chips, not just outlines
    n = len(ROWS)
    right = 0.965
    gap = 0.016
    chip_w = (right - left_margin - gap * (n - 1)) / n
    y0 = 0.055
    chip_h = 0.20

    for i, (label, status, icon) in enumerate(ROWS):
        x0 = left_margin + i * (chip_w + gap)
        color = STATUS[status]
        box = FancyBboxPatch((x0, y0), chip_w, chip_h,
                              boxstyle="round,pad=0.002,rounding_size=0.014",
                              linewidth=1.6, edgecolor=color, facecolor=color,
                              alpha=0.10, transform=ax.transAxes, zorder=2)
        ax.add_patch(box)
        # re-stroke the border at full opacity (the fill alpha above would
        # otherwise fade the edge too)
        border = FancyBboxPatch((x0, y0), chip_w, chip_h,
                                 boxstyle="round,pad=0.002,rounding_size=0.014",
                                 linewidth=1.6, edgecolor=color, facecolor="none",
                                 transform=ax.transAxes, zorder=3)
        ax.add_patch(border)
        cx = x0 + chip_w / 2
        ax.text(cx, y0 + chip_h * 0.66, icon, fontsize=19, color=color,
                 ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(cx, y0 + chip_h * 0.24, label, fontsize=9.2, color=INK,
                 ha="center", va="center", zorder=4)

    out = FIGURES / "cover_linkedin.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
