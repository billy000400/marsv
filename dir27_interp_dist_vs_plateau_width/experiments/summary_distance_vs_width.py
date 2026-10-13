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
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
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
# "monotonic curve"        = the FINAL-block readout curve y(t), sampled at 101 values of t,
#     (a) crosses the 0.1 level exactly once, (b) crosses the 0.9 level exactly once, and
#     (c) never drops by more than BACKSTEP_TOL = 0.05 between two neighbouring values of t.
#     y(t) runs from 0 at ' big' to 1 at token B, so 0.05 is 5% of the full rise.
#     Only the final-block curve is tested, to match "final layer only".
#     ASSUMPTION 1: the CSVs do not hold this final-block test. Their column
#     flag_nonmonotonic is 1 when EITHER the middle-block OR the final-block curve fails
#     (a)-(c). The final-block test is therefore rebuilt from the per-curve statistics that
#     the same sweep saved next to each CSV: <csv name without .csv>/shard_*.npz, array
#     "stats", columns STAT_NCROSS10 / STAT_NCROSS90 / STAT_BACKSTEP, matched to the CSV rows
#     by token id. D and W_final are still read from the CSV.
#     ASSUMPTION 2: set MONOTONIC_RULE = "csv_flag" to use the CSV column instead (stricter:
#     it also drops a token whose middle-block curve fails). That needs no shard files.
COL_ID = "token_id"
COL_DISTANCE = "D"
COL_WIDTH = "W_final"          # EDIT HERE to plot another layer's width (e.g. "W_mid")
COL_FLAG = "flag_nonmonotonic"
MONOTONIC_FLAG_VALUE = "0"
MONOTONIC_RULE = "final_block"  # "final_block" (shard statistics) or "csv_flag" (CSV column)
STAT_NCROSS10, STAT_NCROSS90, STAT_BACKSTEP = 9, 10, 11   # final-block columns of "stats"
BACKSTEP_TOL = 0.05
# Line printed under the figure so the filter is stated on the figure itself.
FILTER_NOTE = {
    "final_block": "Monotonic curve: the final-block readout curve crosses the 10% and the 90% "
                   "level once each and never drops by more than 0.05 between neighbouring "
                   "interpolation positions. Every point drawn is counted in r and in rho.",
    "csv_flag": "Monotonic curve: CSV column flag_nonmonotonic = 0 (middle-block and final-block "
                "readout curves both monotonic). Every point drawn is counted in r and in rho.",
}

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
# model, so no model is identified by colour alone. EDIT HERE for colours / markers.
COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
MARKERS = ["o", "s", "^", "D", "v"]
MARKER_SIZE = 3                # scatter point area (points^2)
ALPHA = 0.5                    # point transparency
RASTERIZE_POINTS = True        # keeps the PDF small (50k-150k points per panel)
# With 46k-133k points per panel, transparency alone leaves the middle of each cloud a
# solid blob. So each point is also coloured by how many points share its neighbourhood
# (log scale) and the crowded points are drawn last. False = one flat colour per model.
COLOR_BY_DENSITY = True
DENSITY_BINS = 120             # neighbourhood grid: bins per axis over the drawn range
DENSITY_CMAP = "cividis"       # sequential map designed for colour-vision deficiency
DENSITY_LABEL = "Local point density (log scale; each panel scaled to its own densest spot)"
DENSITY_TICKS = ["1 point", "densest"]
CBAR_BOX = [0.14, 0.015, 0.36, 0.014]   # density colour bar [x0, y0, width, height], figure fractions

# --- Axes (panels 1-5) ------------------------------------------------------
X_LABEL = "Interpolation distance D (L2, block-0 output)"   # EDIT axis labels here
Y_LABEL = "Transition width W (final block)"
TITLE_FMT = "{name} (Pearson r = {r:.2f})"                   # EDIT panel titles here
TITLE_EMPTY_FMT = "{name} (no monotonic curves)"
SHARE_Y = True                 # EDIT HERE: False lets every scatter panel pick its own y range
Y_LIM = None                   # None = common range covering all plotted points; or (lo, hi)
Y_PAD_FRAC = 0.03              # padding around the automatic common y range
# Every model keeps its own x range and every panel draws ALL of its monotonic curves.
# Llama: a few dozen monotonic tokens lie far from the rest (D up to 9.1, bulk at 1.4-2.2),
# so the full-range panel shows the bulk as a narrow strip. An inset repeats the bulk
# enlarged. Entry: name -> (inset x range, inset box [x0, y0, width, height] in axes
# fractions). The box sits where the panel has no points. Set X_ZOOM_INSETS = {} for none.
X_ZOOM_INSETS = {"Llama 3.1-8B": ((1.0, 2.5), [0.42, 0.60, 0.56, 0.37])}
INSET_TITLE = "enlarged: D from {lo:g} to {hi:g}"
NOTE_Y = 0.97                  # height of the "n = ..." text (axes fraction, right-aligned);
NOTE_Y_WITH_INSET = 0.03       # a panel with an inset prints it at this height instead
GRID_ALPHA = 0.25
GRID_LW = 0.5

# --- Summary panel (panel 6) ------------------------------------------------
SUMMARY_TITLE = "Spearman rank correlation, D vs W (final block)"
SUMMARY_X_LABEL = r"Spearman correlation coefficient $\rho$"
SUMMARY_X_LIM = (-1.0, 1.0)
SUMMARY_MARKER_SIZE = 90
REF_LINE_LW = 2.0              # lines at x = -1 and x = +1 (they sit on the panel edges)
ZERO_LINE_LW = 1.0             # line at x = 0
DRAW_ZERO_LINE = True
# EDIT summary-panel annotations here: (x position, text, horizontal alignment).
REF_ANNOTATIONS = [
    (-1.0, "-1 = perfect\nnegative monotonic\nrelationship", "left"),
    (1.0, "+1 = perfect\npositive monotonic\nrelationship", "right"),
]
ZERO_ANNOTATION = "0 = no monotonic relationship"
VALUE_FMT = "{rho:+.2f}"       # label printed beside each model's point
VALUE_OFFSET_PT = 11           # gap between a point and its label, away from the x = 0 line
FOOTNOTE_Y = -0.075            # figure-fraction height of the FILTER_NOTE line


# ============================================================================
# 3. DATA LOADING
# ============================================================================
def load_csv(path):
    """Return (token id, distance, width, CSV-flag-says-monotonic) arrays for one model's CSV."""
    ids, dist, width, mono = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row[COL_ID]))
            dist.append(float(row[COL_DISTANCE]))
            width.append(float(row[COL_WIDTH]))
            mono.append(row[COL_FLAG] == MONOTONIC_FLAG_VALUE)
    return np.array(ids), np.array(dist), np.array(width), np.array(mono, dtype=bool)


def load_final_block_monotonic(csv_path, ids):
    """Final-block monotonic test per CSV row, from the shard statistics beside the CSV."""
    shard_dir = os.path.splitext(csv_path)[0]
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz in {shard_dir}; set MONOTONIC_RULE = 'csv_flag' "
                                "to filter on the CSV column instead")
    shards = [np.load(p) for p in paths]
    shard_ids = np.concatenate([s["ids"] for s in shards])
    stats = np.concatenate([s["stats"] for s in shards])
    ok = ((stats[:, STAT_NCROSS10] == 1) & (stats[:, STAT_NCROSS90] == 1)
          & (stats[:, STAT_BACKSTEP] <= BACKSTEP_TOL))
    by_id = dict(zip(shard_ids.tolist(), ok.tolist()))
    return np.array([by_id[i] for i in ids.tolist()], dtype=bool)


# ============================================================================
# 4. FILTERING / METRIC EXTRACTION
# ============================================================================
def select_monotonic(csv_path, ids, dist, width, csv_mono):
    """EDIT filtering logic here. Keeps monotonic curves with finite distance and width."""
    mono = csv_mono if MONOTONIC_RULE == "csv_flag" else load_final_block_monotonic(csv_path, ids)
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


def density_order_and_colors(x, y, xlim, ylim):
    """Drawing order (sparse first) and a 0-1 colour value per point: log of the number of
    points in its grid cell, divided by the log of the fullest cell."""
    counts, xe, ye = np.histogram2d(x, y, bins=DENSITY_BINS, range=[xlim, ylim])
    ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, DENSITY_BINS - 1)
    iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, DENSITY_BINS - 1)
    c = np.maximum(counts[ix, iy], 1.0)
    value = np.log(c) / max(np.log(counts.max()), 1e-9)
    order = np.argsort(value)
    return order, value[order]


def draw_points(ax, x, y, xlim, ylim, color, marker):
    """Scatter the points inside xlim on `ax` and fix its limits."""
    inside = (x >= xlim[0]) & (x <= xlim[1])
    x, y = x[inside], y[inside]
    kw = dict(s=MARKER_SIZE, alpha=ALPHA, marker=marker, linewidths=0, rasterized=RASTERIZE_POINTS)
    if COLOR_BY_DENSITY and len(x):
        order, value = density_order_and_colors(x, y, xlim, ylim)
        ax.scatter(x[order], y[order], c=value, cmap=DENSITY_CMAP, vmin=0, vmax=1, **kw)
    else:
        ax.scatter(x, y, color=color, **kw)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_scatter(ax, rec, color, marker, ylim):
    """One model's scatter panel. `rec` is a dict built in main()."""
    style_axes(ax)
    ax.set_xlabel(X_LABEL, fontsize=FONT_LABEL)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
    ax.set_ylim(*ylim)
    if rec["n"] == 0:  # nothing passed the filter: keep the panel, say so
        ax.set_title(TITLE_EMPTY_FMT.format(name=rec["name"]), fontsize=FONT_TITLE)
        ax.text(0.5, 0.5, "no qualifying monotonic curves", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_ANNOT)
        return
    dist, width = rec["dist"], rec["width"]
    pad = 0.03 * max(np.ptp(dist), 1e-9)
    draw_points(ax, dist, width, (dist.min() - pad, dist.max() + pad), ylim, color, marker)
    ax.set_title(TITLE_FMT.format(name=rec["name"], r=rec["pearson"]), fontsize=FONT_TITLE)
    ax.grid(alpha=GRID_ALPHA, lw=GRID_LW)
    has_inset = rec["name"] in X_ZOOM_INSETS
    ax.text(0.98, NOTE_Y_WITH_INSET if has_inset else NOTE_Y, f"n = {rec['n']:,}", ha="right",
            va="bottom" if has_inset else "top", transform=ax.transAxes, fontsize=FONT_ANNOT)
    if has_inset:  # enlarged copy of the bulk; the main panel keeps every point
        zoom, box = X_ZOOM_INSETS[rec["name"]]
        inset = ax.inset_axes(box)
        draw_points(inset, dist, width, zoom, ylim, color, marker)
        inset.tick_params(labelsize=FONT_TICK - 2)
        inset.grid(alpha=GRID_ALPHA, lw=GRID_LW)
        inset.text(0.03, 0.95, INSET_TITLE.format(lo=zoom[0], hi=zoom[1]), ha="left", va="top",
                   transform=inset.transAxes, fontsize=FONT_ANNOT - 1)


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
    ax.spines["left"].set_visible(False)  # the x = -1 reference line is drawn in its place
    n = len(records)
    ys = np.arange(n)[::-1]  # first model at the top
    lo, hi = SUMMARY_X_LIM
    for x in (lo, hi):
        ax.axvline(x, color="black", lw=REF_LINE_LW, ls="-", clip_on=False)
    if DRAW_ZERO_LINE:
        ax.axvline(0.0, color="gray", lw=ZERO_LINE_LW, ls="--")
    for y, rec, color, marker in zip(ys, records, COLORS, MARKERS):
        if np.isnan(rec["spearman"]):  # no qualifying curves: no point, just a note
            ax.text(0.0, y, "no monotonic curves", ha="center", va="center", fontsize=FONT_ANNOT)
            continue
        ax.scatter(rec["spearman"], y, s=SUMMARY_MARKER_SIZE, color=color, marker=marker,
                   edgecolors="black", linewidths=0.7, zorder=3)
        side = -1 if rec["spearman"] < 0 else 1  # label on the side away from the x = 0 line
        ax.annotate(VALUE_FMT.format(rho=rec["spearman"]), (rec["spearman"], y),
                    xytext=(side * VALUE_OFFSET_PT, 0), textcoords="offset points",
                    ha="right" if side < 0 else "left", va="center", fontsize=FONT_ANNOT)
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
    ax.set_xlim(lo, hi)
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
        path = os.path.join(results_dir, rel)
        ids, d_all, w_all, csv_mono = load_csv(path)
        dist, width = select_monotonic(path, ids, d_all, w_all, csv_mono)
        r, rho = correlations(dist, width)
        records.append(dict(name=name, dist=dist, width=width, n=len(dist), pearson=r, spearman=rho))
        print(f"{name:14s} rows={len(d_all):7d} monotonic={len(dist):7d} "
              f"pearson_r={r:+.4f} spearman_rho={rho:+.4f}")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
    axes = axes.ravel()
    shared_ylim = common_y_limits(records)  # same y range on all scatter panels
    for ax, rec, color, marker in zip(axes, records, COLORS, MARKERS):
        ylim = shared_ylim if SHARE_Y or rec["n"] == 0 else common_y_limits([rec])
        plot_scatter(ax, rec, color, marker, ylim)
    plot_summary(axes[-1], records)
    if COLOR_BY_DENSITY:
        cbar = fig.colorbar(ScalarMappable(cmap=DENSITY_CMAP), cax=fig.add_axes(CBAR_BOX),
                            orientation="horizontal", ticks=[0, 1])
        cbar.ax.set_xticklabels(DENSITY_TICKS, fontsize=FONT_TICK)
        cbar.set_label(DENSITY_LABEL, fontsize=FONT_ANNOT)
    fig.text(0.5, FOOTNOTE_Y, FILTER_NOTE[MONOTONIC_RULE], ha="center", va="top",
             fontsize=FONT_ANNOT, wrap=True)

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
