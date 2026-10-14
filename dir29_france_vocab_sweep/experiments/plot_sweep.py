"""Draw the ' France' -> B sweep histogram (Figure 1) from the saved per-token CSV. Runs no model.

Style follows panel (a) of Direction 27's fig2_final_width_distribution.png (s3_plots.py).
To restyle the figure, edit the STYLE block and re-run; see ../PLOTTING.md.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "france_tokens.csv")
OUT = os.path.join(ROOT, "plots", "france_vocab_width_histogram.png")

# ---- STYLE: edit here ----
FIGSIZE = (6, 4.5)  # inches; one panel of Direction 27's 12 x 4.5 two-panel figure
DPI = 130
BINS = 200
COLOR = "#0072B2"  # CVD[0], blue
TITLE = '"The capital of [France]" -> [B] sweep'
XLABEL = "final-layer width W_final"
YLABEL = "number of tokens"
# --------------------------


def load_data(path=DATA):
    """Return W_final for every token B in the sweep (one row of the CSV per token)."""
    with open(path, newline="") as f:
        return np.array([float(r["W_final"]) for r in csv.DictReader(f)])


def plot(Wf, out=OUT):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(Wf, bins=BINS, color=COLOR)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    Wf = load_data()
    plot(Wf)
    print(f"{len(Wf)} tokens -> {OUT}")
