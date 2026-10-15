"""Shared matplotlib style for training / eval plots.

Runs are colored by peak LR with a FIXED LR -> color map (the dataviz categorical palette, slots
in LR order), so an LR keeps its color in every figure and phase. The base model (R0) is a
dashed neutral line. A single-hue ordinal ramp was used while there were 5 LRs; with 6 (R6/R7
added) no blue ramp clears the visible-step check, so the categorical slots are used instead.
The 6 used slots pass the validator in this order (adjacent CVD dE >= 9.1, normal-vision dE >=
19.6); aqua / yellow / magenta are < 3:1 on the light surface, so every figure keeps a legend and
a table view (training_summary.md, summary.md).
"""
import re

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BASE = "#52514e"  # base model reference line (dashed)

# Peak LRs, low -> high, and their fixed colors (categorical slots 1-7: blue, orange, aqua,
# yellow, magenta, green, violet). 5e-4 (R5) was dropped; its slot stays reserved.
LRS = [2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 2e-4, 5e-4]
RAMP = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def run_lr(run):
    """Peak LR from a run name like R3_1h_lr5e-05, else None."""
    m = re.search(r"lr([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)", run)  # lr5e-05, lr0.0002
    return float(m.group(1)) if m else None


def run_color(run):
    """Fixed color for a run: its LR's ramp step (nearest LR in log space); base -> BASE."""
    lr = run_lr(run)
    if lr is None:
        return BASE
    import math
    i = min(range(len(LRS)), key=lambda k: abs(math.log(LRS[k]) - math.log(lr)))
    return RAMP[i]


def run_label(run):
    lr = run_lr(run)
    return f"lr {lr:.0e}".replace("e-0", "e-") if lr is not None else run


def setup():
    """Import pyplot with the house style applied; None if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelcolor": INK_2, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
    })
    return plt
