"""Model workflow diagram for the paper.

    python modules/workflow_figure.py

Pure matplotlib - draws boxes and connectors, fits nothing, runs in seconds.
Style matches the pipeline diagrams in the other Research projects
(Student Depression Fig01, Car Sales Fig05).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import DPI, figure

# ------------------------------------------------------------------ palette
INK = "#26354f"
C = {
    "data":  "#d4e7f7",   # blue    - data stages
    "rep":   "#fbdbe1",   # pink    - feature representations
    "lin":   "#fdeacb",   # peach   - linear models
    "prob":  "#e8d5f5",   # violet  - probabilistic
    "tree":  "#d8f0dc",   # green   - bagged trees
    "boost": "#d3d8f7",   # indigo  - boosting
    "other": "#cde9f0",   # cyan    - instance based and neural
    "tune":  "#fbd9b8",   # orange  - tuning
    "eval":  "#d2ebe7",   # teal    - evaluation
}

LW, MS = 2.6, 11.0
FIGW, FIGH = 21.0, 12.0
YUNIT = FIGH / 100.0          # inches per y-axis unit, for text-block maths

fig, ax = plt.subplots(figsize=(FIGW, FIGH))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


# ------------------------------------------------------------------ drawing
def _lineheight(fs: float) -> float:
    """Height of one text line, in y-axis units."""
    return fs * 1.45 / 72.0 / YUNIT


def box(x, y, w, h, fill, head, body=None, hfs=21.0, bfs=15.5):
    """Rounded box with a bold heading and an optional smaller body block."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.3",
        facecolor=fill, edgecolor=INK, linewidth=2.6, zorder=2))

    if body is None:
        ax.text(x + w / 2, y + h / 2, head, ha="center", va="center",
                fontsize=hfs, fontweight="bold", color=INK,
                linespacing=1.45, zorder=3)
        return

    hl, bl = _lineheight(hfs), _lineheight(bfs)
    nb = body.count("\n") + 1
    gap = 1.1
    top = y + h / 2 + (hl + gap + nb * bl) / 2

    ax.text(x + w / 2, top - hl / 2, head, ha="center", va="center",
            fontsize=hfs, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, top - hl - gap - nb * bl / 2, body,
            ha="center", va="center", fontsize=bfs, fontweight="bold",
            color=INK, linespacing=1.5, zorder=3)


def vline(x, y1, y2, head=True):
    ax.plot([x, x], [y1, y2], color=INK, lw=LW, solid_capstyle="round", zorder=1)
    if head:
        ax.plot([x], [y2], marker="v", ms=MS, color=INK, zorder=1)


def hline(x1, x2, y):
    ax.plot([x1, x2], [y, y], color=INK, lw=LW, solid_capstyle="round", zorder=1)


def harrow(x1, x2, y):
    ax.plot([x1, x2 - 0.9], [y, y], color=INK, lw=LW,
            solid_capstyle="round", zorder=1)
    ax.plot([x2], [y], marker=">", ms=MS, color=INK, zorder=1)


# ------------------------------------------------------------------ geometry
NCOL, CW, CGAP, CX0 = 5, 16.72, 2.6, 3.0
colx = [CX0 + i * (CW + CGAP) for i in range(NCOL)]
cen = [x + CW / 2 for x in colx]
SPAN = NCOL * CW + (NCOL - 1) * CGAP          # 94.0

Y_DATA, H_DATA = 80.0, 11.0
BUS1 = 76.3
Y_REP, H_REP = 63.0, 11.0
BUS2 = 59.8
Y_FAM, H_FAM = 38.0, 18.0
BUS3 = 34.5
Y_TUNE, H_TUNE = 21.0, 9.5
Y_EVAL, H_EVAL = 6.5, 10.5

ax.text(50, 94.8, "Review Sentiment Classification Pipeline",
        ha="center", va="center", fontsize=34, fontweight="bold", color=INK)

# ---- row 1: data chain -----------------------------------------------------
DW = 27.0
dxs = [CX0, CX0 + (SPAN - DW) / 2, CX0 + SPAN - DW]
DATA = [("Dataset", "23,486 reviews"),
        ("Cleaning", "drop neutral, deduplicate\n19,813 reviews  |  88% positive"),
        ("Stratified Split", "70 / 15 / 15")]
for i, (x, (head, sub)) in enumerate(zip(dxs, DATA)):
    box(x, Y_DATA, DW, H_DATA, C["data"], head, sub, hfs=20.0, bfs=14.5)
    if i:
        harrow(dxs[i - 1] + DW + 1.4, x - 0.7, Y_DATA + H_DATA / 2)

# ---- row 2: two feature representations ------------------------------------
r1x, r1w = colx[0], 2 * CW + CGAP
r2x, r2w = colx[2], 3 * CW + 2 * CGAP
r1c, r2c = r1x + r1w / 2, r2x + r2w / 2

vline(dxs[2] + DW / 2, Y_DATA, BUS1, head=False)
hline(r1c, dxs[2] + DW / 2, BUS1)
for c in (r1c, r2c):
    vline(c, BUS1, Y_REP + H_REP + 0.5)

box(r1x, Y_REP, r1w, H_REP, C["rep"], "TF-IDF  (sparse)",
    "word 1-2 grams  +  char_wb 3-5 grams" + chr(10) + "96,755 features",
    hfs=20.0, bfs=14.5)
box(r2x, Y_REP, r2w, H_REP, C["rep"], "TruncatedSVD  (dense)",
    "TF-IDF projected to 300 components", hfs=20.0, bfs=14.5)

# ---- row 3: five model families -------------------------------------------
FAMS = [("Linear", "Logistic Regression\nLinear SVM", C["lin"]),
        ("Probabilistic", "Multinomial\nNaive Bayes", C["prob"]),
        ("Tree", "Random Forest\nExtra Trees", C["tree"]),
        ("Boosting", "XGBoost\nLightGBM\nCatBoost", C["boost"]),
        ("Other", "k-Nearest Neighbours\nMLP", C["other"])]

for lo, hi, src in ((0, 2, r1c), (2, 5, r2c)):
    vline(src, Y_REP, BUS2, head=False)
    hline(cen[lo], cen[hi - 1], BUS2)
    for c in cen[lo:hi]:
        vline(c, BUS2, Y_FAM + H_FAM + 0.5)

for x, (head, sub, fill) in zip(colx, FAMS):
    box(x, Y_FAM, CW, H_FAM, fill, head, sub, hfs=20.0, bfs=14.0)

# ---- row 4: tuning ---------------------------------------------------------
for c in cen:
    ax.plot([c, c], [Y_FAM, BUS3], color=INK, lw=LW, zorder=1)
hline(cen[0], cen[-1], BUS3)
vline(50, BUS3, Y_TUNE + H_TUNE + 0.5)

box(CX0, Y_TUNE, SPAN, H_TUNE, C["tune"], "Optuna Hyper-parameter Tuning",
    "TPE sampler   |   objective = validation macro-F1", hfs=23.0, bfs=16.0)

# ---- row 5: evaluation -----------------------------------------------------
vline(50, Y_TUNE, Y_EVAL + H_EVAL + 0.5)
box(CX0, Y_EVAL, SPAN, H_EVAL, C["eval"], "Evaluation",
    "Macro-F1   |   MCC   |   ROC-AUC   |   Accuracy", hfs=25.0, bfs=18.0)

# ------------------------------------------------------------------ save
for paper, dpi in ((True, DPI), (False, 120)):
    fig.savefig(figure("fig00_workflow.png", paper=paper),
                dpi=dpi, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("  saved fig00_workflow.png")
