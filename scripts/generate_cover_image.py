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
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"

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

    # Title block
    ax.text(0.06, 0.90, "Google's TabFM vs.\nTuned Classical ML Models",
             fontsize=30, fontweight="bold", color=INK, va="top", ha="left",
             linespacing=1.25)
    ax.text(0.06, 0.58, "A Production-Readiness Evaluation on\n10 Financial Risk Datasets",
             fontsize=15, color=SECONDARY, va="top", ha="left", linespacing=1.4)

    # Scorecard row
    n = len(ROWS)
    left, right = 0.06, 0.97
    gap = 0.018
    chip_w = (right - left - gap * (n - 1)) / n
    y0 = 0.06
    chip_h = 0.22

    for i, (label, status, icon) in enumerate(ROWS):
        x0 = left + i * (chip_w + gap)
        color = STATUS[status]
        box = FancyBboxPatch((x0, y0), chip_w, chip_h,
                              boxstyle="round,pad=0.004,rounding_size=0.012",
                              linewidth=1.4, edgecolor=color, facecolor=SURFACE,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(box)
        cx = x0 + chip_w / 2
        ax.text(cx, y0 + chip_h * 0.62, icon, fontsize=17, color=color,
                 ha="center", va="center", fontweight="bold", zorder=3)
        ax.text(cx, y0 + chip_h * 0.22, label, fontsize=8.3, color=INK,
                 ha="center", va="center", zorder=3)

    ax.text(0.06, 0.355, "Where it wins, and where it doesn't", fontsize=11,
             color=MUTED, va="top", ha="left")

    out = FIGURES / "cover_linkedin.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
