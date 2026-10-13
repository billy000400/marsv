"""Summary figure: interpolation distance vs transition width, five models (2x3 layout).

Panels 1-5: one scatter per model, final-layer transition width against interpolation
distance, monotonic curves only, Pearson r in the title.
Panel 6:    one Spearman coefficient per model (same filter), x-axis fixed to [-1, 1].

Reads existing CSVs only; runs no model.

Usage (from the direction root or anywhere else):
    python experiments/summary_distance_vs_width.py
    python experiments/summary_distance_vs_width.py --results-dir results/block0_resid_post \
        --out plots/block0_resid_post/summary_distance_vs_width --formats png pdf
"""
# ============================================================================
# 1. IMPORTS
# ============================================================================
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# 2. USER-EDITABLE CONFIGURATION
# ============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "results", "block0_resid_post")
DEFAULT_OUT = os.path.join(ROOT, "plots", "block0_resid_post", "summary_distance_vs_width")
DEFAULT_FORMATS = ["png", "pdf"]

# --- Models: (panel title name, CSV path relative to the results dir) -------
# EDIT HERE to rename a panel, reorder panels, or point at another file.
MODELS = [
    ("GPT-2 Large", "sweep.csv"),
    ("GPT-2 XL", os.path.join("gen", "gpt2-xl.csv")),
    ("Pythia 1.4B", os.path.join("gen", "pythia-1.4b.csv")),
    ("Llama 3.1-8B", os.path.join("gen", "llama-3.1-8b.csv")),
    ("Qwen3-8B", os.path.join("gen", "qwen3-8b-base.csv")),
]

# --- Data-format assumptions (the ONLY place they live) ---------------------
# Every CSV has one row per vocabulary token B. The run interpolates linearly from the
# token ' big' to token B at the last-position residual stream after transformer block 0
# and reads the model output at each interpolation position t in [0, 1].
#
# "interpolation distance" = column D: L2 distance between the ' big' endpoint and the
#     token-B endpoint, measured where the interpolation happens (block-0 output).
# "final layer"            = column W_final: the width read out at the LAST transformer
#     block. (W_mid, the middle-block width, is in the same files and is not used.)
# "transition width"       = W = t_0.9 - t_0.1, the span of t over which the normalized
#     readout curve goes from 10% to 90% of the way from its ' big' value to its B value.
# "monotonic curve"        = flag_nonmonotonic == 0.
#     ASSUMPTION: the CSVs hold ONE flag per token. It is set when EITHER the middle-block
#     OR the final-block curve crosses the 0.1 or 0.9 level more than once, or steps
#     backwards beyond the tolerance (see experiments/s3_plots.py load_sweep). There is no
#     final-block-only flag in these files, so a token whose final-block curve is monotonic
#     but whose middle-block curve is not is also dropped here.
COL_DISTANCE = "D"
COL_WIDTH = "W_final"          # EDIT HERE to plot another layer's width (e.g. "W_mid")
COL_FLAG = "flag_nonmonotonic"
MONOTONIC_FLAG_VALUE = "0"

# --- Figure layout ----------------------------------------------------------
FIGSIZE = (15.0, 8.6)          # inches (width, height)
NROWS, NCOLS = 2, 3
WSPACE = 0.22                  # horizontal gap between panels
HSPACE = 0.36                  # vertical gap between panels
DPI = 300                      # output resolution of raster formats

# --- Fonts ------------------------------------------------------------------
FONT_TITLE = 12
FONT_LABEL = 11
FONT_TICK = 9.5
FONT_ANNOT = 9

# --- Scatter style (panels 1-5) ---------------------------------------------
# Colour-vision-deficiency-safe palette (no green); the marker shape also changes per
# model, so no panel is identified by colour alone. EDIT HERE for colours / markers.
COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
MARKERS = ["o", "s", "^", "D", "v"]
MARKER_SIZE = 5                # scatter point area (points^2)
ALPHA = 0.08                   # point transparency
RASTERIZE_POINTS = True        # keeps the PDF small (50k-150k points per panel)

# --- Axes (panels 1-5) ------------------------------------------------------
X_LABEL = "Interpolation distance D (L2, block-0 output)"   # EDIT axis labels here
Y_LABEL = "Transition width W (final block)"
TITLE_FMT = "{name} (Pearson r = {r:.2f})"                   # EDIT panel titles here
TITLE_EMPTY_FMT = "{name} (no monotonic curves)"
SHARE_Y = True                 # EDIT HERE: False lets every scatter panel pick its own y range
Y_LIM = None                   # None = common range covering all plotted points; or (lo, hi)
Y_PAD_FRAC = 0.03              # padding around the automatic common y range
# Each model keeps its own x range. A model not listed here gets its full data range.
# Llama: 26 far tokens (D up to 16.5) would squeeze the cloud into a sliver, so the panel
# uses the same 1-2.5 window as the existing Llama figure; the panel prints how many points
# fall outside. Set X_LIMS = {} to draw every point.
# Correlations always use ALL monotonic curves, whether drawn or not.
X_LIMS = {"Llama 3.1-8B": (1.0, 2.5)}
GRID_ALPHA = 0.25
GRID_LW = 0.5

# --- Summary panel (panel 6) ------------------------------------------------
SUMMARY_TITLE = "Spearman rank correlation, D vs W (final block)"
SUMMARY_X_LABEL = r"Spearman correlation coefficient $\rho$"
SUMMARY_X_LIM = (-1.0, 1.0)
SUMMARY_X_MARGIN = 0.06        # blank space drawn outside +-1 so the reference lines are visible
SUMMARY_MARKER_SIZE = 90
REF_LINE_LW = 1.3              # lines at x = -1 and x = +1
ZERO_LINE_LW = 1.0             # line at x = 0
DRAW_ZERO_LINE = True
# EDIT summary-panel annotations here: (x position, text, horizontal alignment).
REF_ANNOTATIONS = [
    (-1.0, "-1 = perfect\nnegative monotonic\nrelationship", "left"),
    (1.0, "+1 = perfect\npositive monotonic\nrelationship", "right"),
]
ZERO_ANNOTATION = "0 = no monotonic relationship"
VALUE_FMT = "{rho:+.2f}"       # label printed next to each model's point


# ============================================================================
# 3. DATA LOADING
# ============================================================================
def load_csv(path):
    """Return (distance, width, is_monotonic) arrays for one model's CSV."""
    dist, width, mono = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dist.append(float(row[COL_DISTANCE]))
            width.append(float(row[COL_WIDTH]))
            mono.append(row[COL_FLAG] == MONOTONIC_FLAG_VALUE)
    return np.array(dist), np.array(width), np.array(mono, dtype=bool)


# ============================================================================
# 4. FILTERING / METRIC EXTRACTION
# ============================================================================
def select_monotonic(dist, width, mono):
    """EDIT filtering logic here. Keeps monotonic curves with finite distance and width."""
    keep = mono & np.isfinite(dist) & np.isfinite(width)
    return dist[keep], width[keep]


# ============================================================================
# 5. CORRELATIONS
# ============================================================================
def correlations(dist, width):
    """Pearson r and Spearman rho of width against distance; NaN when undefined."""
    if len(dist) < 3 or np.ptp(dist) == 0 or np.ptp(width) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(dist, width)[0]), float(spearmanr(dist, width)[0])


# ============================================================================
# 6. PLOTTING HELPERS
# ============================================================================
def style_axes(ax):
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scatter(ax, rec, color, marker):
    """One model's scatter panel. `rec` is a dict built in main()."""
    style_axes(ax)
    ax.set_xlabel(X_LABEL, fontsize=FONT_LABEL)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
    if rec["n"] == 0:  # nothing passed the filter: keep the panel, say so
        ax.set_title(TITLE_EMPTY_FMT.format(name=rec["name"]), fontsize=FONT_TITLE)
        ax.text(0.5, 0.5, "no qualifying monotonic curves", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_ANNOT)
        return
    ax.scatter(rec["dist"], rec["width"], s=MARKER_SIZE, alpha=ALPHA, color=color, marker=marker,
               linewidths=0, rasterized=RASTERIZE_POINTS)
    ax.set_title(TITLE_FMT.format(name=rec["name"], r=rec["pearson"]), fontsize=FONT_TITLE)
    note = f"n = {rec['n']:,}"
    if rec["name"] in X_LIMS:
        lo, hi = X_LIMS[rec["name"]]
        ax.set_xlim(lo, hi)
        hidden = int(((rec["dist"] < lo) | (rec["dist"] > hi)).sum())
        note += f"\n{hidden:,} beyond x range (counted in r)"
    ax.grid(alpha=GRID_ALPHA, lw=GRID_LW)
    ax.text(0.98, 0.97, note, ha="right", va="top", transform=ax.transAxes, fontsize=FONT_ANNOT)


def common_y_limits(records):
    """One y range for all scatter panels, covering every plotted point."""
    if Y_LIM is not None:
        return Y_LIM
    widths = [rec["width"] for rec in records if rec["n"] > 0]
    if not widths:
        return (0.0, 1.0)
    lo, hi = min(w.min() for w in widths), max(w.max() for w in widths)
    pad = Y_PAD_FRAC * (hi - lo)
    return (lo - pad, hi + pad)


def plot_summary(ax, records):
    """Panel 6: one Spearman coefficient per model on a fixed [-1, 1] axis."""
    style_axes(ax)
    n = len(records)
    ys = np.arange(n)[::-1]  # first model at the top
    lo, hi = SUMMARY_X_LIM
    for x in (lo, hi):
        ax.axvline(x, color="black", lw=REF_LINE_LW, ls="-")
    if DRAW_ZERO_LINE:
        ax.axvline(0.0, color="gray", lw=ZERO_LINE_LW, ls="--")
    for y, rec, color, marker in zip(ys, records, COLORS, MARKERS):
        if np.isnan(rec["spearman"]):  # no qualifying curves: no point, just a note
            ax.text(0.0, y, "no monotonic curves", ha="center", va="center", fontsize=FONT_ANNOT)
            continue
        ax.scatter(rec["spearman"], y, s=SUMMARY_MARKER_SIZE, color=color, marker=marker,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.annotate(VALUE_FMT.format(rho=rec["spearman"]), (rec["spearman"], y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=FONT_ANNOT)
    # -1 / +1 annotations sit in a band above the top model row, the 0 annotation below the last.
    y_text = n - 0.05
    for x, text, ha in REF_ANNOTATIONS:
        ax.text(x + (0.03 if ha == "left" else -0.03), y_text, text, ha=ha, va="center",
                fontsize=FONT_ANNOT)
    if DRAW_ZERO_LINE:
        ax.text(0.0, -0.75, ZERO_ANNOTATION, ha="center", va="center", fontsize=FONT_ANNOT,
                color="dimgray", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_yticks(ys)
    ax.set_yticklabels([rec["name"] for rec in records], fontsize=FONT_LABEL)
    ax.set_xlim(lo - SUMMARY_X_MARGIN, hi + SUMMARY_X_MARGIN)
    ax.set_xticks(np.linspace(lo, hi, 5))
    ax.set_ylim(-1.0, n + 0.5)
    ax.set_xlabel(SUMMARY_X_LABEL, fontsize=FONT_LABEL)
    ax.set_title(SUMMARY_TITLE, fontsize=FONT_TITLE)
    ax.grid(axis="y", alpha=GRID_ALPHA, lw=GRID_LW)


# ============================================================================
# 7. MAIN
# ============================================================================
def main(results_dir, out, formats):
    records = []
    for name, rel in MODELS:
        d_all, w_all, mono = load_csv(os.path.join(results_dir, rel))
        dist, width = select_monotonic(d_all, w_all, mono)
        r, rho = correlations(dist, width)
        records.append(dict(name=name, dist=dist, width=width, n=len(dist), pearson=r, spearman=rho))
        print(f"{name:14s} rows={len(d_all):7d} monotonic={len(dist):7d} "
              f"pearson_r={r:+.4f} spearman_rho={rho:+.4f}")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
    axes = axes.ravel()
    scatter_axes = axes[:len(records)]
    for ax, rec, color, marker in zip(scatter_axes, records, COLORS, MARKERS):
        plot_scatter(ax, rec, color, marker)
    if SHARE_Y:  # same y range on all scatter panels (tick labels kept on each for readability)
        ylim = common_y_limits(records)
        for ax in scatter_axes:
            ax.set_ylim(*ylim)
    plot_summary(axes[-1], records)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in formats:
        path = f"{out}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print("wrote", path)
    plt.close(fig)


# ============================================================================
# 8. CLI ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="directory holding sweep.csv and gen/<model>.csv")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS, help="e.g. png pdf")
    args = ap.parse_args()
    main(args.results_dir, args.out, args.formats)
