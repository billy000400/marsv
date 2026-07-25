"""Figures for the comma-to-every-other-character sweep (operator feedback #3)."""
import os, sys, json, string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD, REF_DIAG, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
S = json.load(open(os.path.join(RES, "comma_sweep_summary.json")))
RAW = np.load(os.path.join(RES, "comma_sweep_raw.npz"))
TS = RAW["ts"]
FINAL = "30000"
DIAG = 0.80          # width of the no-plateau straight-line reference
STRICT = 0.25        # frozen plateau rule threshold


def cls(c):
    if c in string.ascii_lowercase:
        return "lower-case letter"
    if c in string.ascii_uppercase:
        return "upper-case letter"
    if c in " \n":
        return "space / newline"
    return "punctuation or digit"


# Character classes: colour is always paired with a marker (scatter) or hatch (bars).
COLORS = {"lower-case letter": CVD[0], "upper-case letter": CVD[1],
          "space / newline": CVD[2], "punctuation or digit": CVD[3]}
MK = {"lower-case letter": "o", "upper-case letter": "s",
      "space / newline": "^", "punctuation or digit": "D"}
HA = {"lower-case letter": "//", "upper-case letter": "\\\\",
      "space / newline": "xx", "punctuation or digit": ".."}


def label(c):
    return {"\n": "\\n", " ": "␣"}.get(c, c)


def stoi_index(c):
    """Vocabulary index used as the raw-array key."""
    return VOCAB.index(c)


VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
final_rows = S["by_step"][FINAL]
chars = list(final_rows)
w_final = np.array([final_rows[c]["w"] for c in chars])

# ---- Figure 1: all 64 raw d(t) curves + width histogram -----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
norm = plt.Normalize(w_final.min(), w_final.max())
cm = plt.cm.viridis
order = np.argsort(-w_final)
for i in order:
    c = chars[i]
    d = RAW[f"s{FINAL}|L0|{stoi_index(c)}"]
    ax[0].plot(TS, d, lw=1.1, color=cm(norm(w_final[i])), alpha=0.85)
med = np.median(np.stack([RAW[f"s{FINAL}|L0|{stoi_index(c)}"] for c in chars]), axis=0)
ax[0].plot(TS, med, lw=2.8, color="k", label="median over the 64 pairs (thick solid)")
ax[0].plot([0, 1], [0, 1], label="straight line, no plateau (dashed)", **REF_DIAG)
ax[0].set_xlabel("interpolation step t  (0 = comma prompt, 1 = other-character prompt)")
ax[0].set_ylabel("relative distance d(t) in final-logit space")
ax[0].set_title("Comma → every other character, final checkpoint (step 30,000)")
ax[0].legend(fontsize=8, loc="upper left")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax[0], label="transition width")

ax[1].hist(w_final, bins=np.arange(0.2, 0.72, 0.03), color=CVD[0], edgecolor="w")
ax[1].axvline(STRICT, label=f"strict plateau rule, width ≤ {STRICT} (dotted)", **REF_RULE)
ax[1].axvline(DIAG, label=f"straight line, width = {DIAG} (dashed)", **REF_DIAG)
ax[1].axvline(np.median(w_final), color="k", lw=2.5,
              label=f"median = {np.median(w_final):.2f} (thick solid)")
ax[1].set_xlim(0.2, 0.85)
ax[1].set_xlabel("transition width  (fraction of the interpolation path)")
ax[1].set_ylabel("number of character pairs")
ax[1].set_title("Width of the transition, 64 pairs")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "comma_all_chars_curves.png"), dpi=130)
plt.close(fig)

# ---- Figure 2: width per target character ------------------------------------------------
srt = np.argsort(w_final)
fig, ax = plt.subplots(figsize=(13, 4.2))
for j, i in enumerate(srt):
    k = cls(chars[i])
    ax.bar(j, w_final[i], color=COLORS[k], hatch=HA[k], edgecolor="w", lw=0.6)
ax.set_xticks(range(len(srt)))
ax.set_xticklabels([label(chars[i]) for i in srt], fontsize=8)
ax.axhline(STRICT, label=f"strict plateau rule ≤ {STRICT} (dotted)", **REF_RULE)
ax.axhline(DIAG, label=f"straight line {DIAG} (dashed)", **REF_DIAG)
ax.set_ylim(0, 0.88)
ax.set_xlim(-0.8, len(srt) - 0.2)
ax.set_xlabel("other character of the pair (comma is always the first endpoint)")
ax.set_ylabel("transition width")
ax.set_title("Transition width for each comma → character pair (final checkpoint, interpolation block 0, final logits)")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], hatch=HA[k], edgecolor="w") for k in COLORS]
ax.legend(handles + [plt.Line2D([], [], **REF_RULE), plt.Line2D([], [], **REF_DIAG)],
          [f"{k} ({HA[k]})" for k in COLORS] +
          [f"strict plateau rule ≤ {STRICT} (dotted)", f"straight line {DIAG} (dashed)"],
          fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "comma_width_by_char.png"), dpi=130)
plt.close(fig)

# ---- Figure 3: depth control + emergence across training ----------------------------------
blocks = sorted(int(k) for k in S["by_block_final"])
wb = [np.array([v["w"] for v in S["by_block_final"][str(b)].values()]) for b in blocks]
steps = S["steps"]
wsx = [np.array([v["w"] for v in S["by_step"][str(s)].values()]) for s in steps]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
m = [np.median(x) for x in wb]
lo = [np.percentile(x, 25) for x in wb]
hi = [np.percentile(x, 75) for x in wb]
ax[0].plot(blocks, m, "o-", color=CVD[0], label="median over 64 pairs (solid, circles)")
ax[0].fill_between(blocks, lo, hi, color=CVD[0], alpha=0.25, hatch="//", label="inter-quartile range")
ax[0].axhline(DIAG, label=f"straight line {DIAG} (dashed)", **REF_DIAG)
ax[0].axhline(STRICT, label=f"strict plateau rule ≤ {STRICT} (dotted)", **REF_RULE)
ax[0].set_xlabel("interpolation block (residual stream after this block is replaced)")
ax[0].set_ylabel("transition width (median over 64 pairs)")
ax[0].set_title("Depth: fewer layers below the readout → wider transition")
ax[0].set_ylim(0, 0.9)
ax[0].legend(fontsize=8)

m = [np.median(x) for x in wsx]
lo = [np.percentile(x, 25) for x in wsx]
hi = [np.percentile(x, 75) for x in wsx]
x = [max(s, 1) for s in steps]
ax[1].plot(x, m, "s--", color=CVD[1], label="median over 64 pairs (dashed, squares)")
ax[1].fill_between(x, lo, hi, color=CVD[1], alpha=0.25, hatch="\\\\", label="inter-quartile range")
ax[1].axhline(DIAG, label=f"straight line {DIAG} (dashed)", **REF_DIAG)
ax[1].axhline(STRICT, label=f"strict plateau rule ≤ {STRICT} (dotted)", **REF_RULE)
ax[1].set_xscale("log")
ax[1].set_xlabel("training step (log scale; step 0 plotted at 1)")
ax[1].set_ylabel("transition width (median over 64 pairs)")
ax[1].set_title("Training: transition narrows early, then stays flat")
ax[1].set_ylim(0, 0.9)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "comma_depth_and_training.png"), dpi=130)
plt.close(fig)

# ---- Figure 4: what predicts the width? ---------------------------------------------------
from scipy.stats import spearmanr

ep = S["endpoints_final"]["per_char"]
p_next = np.array([ep[c]["p_next"] for c in chars])
sep = np.array([ep[c]["logit_sep"] for c in chars])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
r1 = spearmanr(w_final, p_next)
for k in COLORS:                       # one scatter call per class: colour AND marker differ
    idx = [i for i, c in enumerate(chars) if cls(c) == k]
    ax[0].scatter(np.maximum(p_next[idx], 1e-9), w_final[idx], c=COLORS[k], marker=MK[k],
                  s=38, edgecolor="w", lw=0.4)
    ax[1].scatter(sep[idx], w_final[idx], c=COLORS[k], marker=MK[k], s=38, edgecolor="w", lw=0.4)
ax[0].set_xscale("log")
ax[0].set_xlabel("model probability of the character after \"The house was \" (log scale)")
ax[0].set_ylabel("transition width")
ax[0].set_title(f"Width vs how likely the character is\nSpearman ρ = {r1.statistic:.2f} (p = {r1.pvalue:.1e}, n = 64)")
r2 = spearmanr(w_final, sep)
ax[1].set_xlabel("distance between the two endpoint logit vectors (L2)")
ax[1].set_ylabel("transition width")
ax[1].set_title(f"Width vs how far apart the endpoints are\nSpearman ρ = {r2.statistic:.2f} (p = {r2.pvalue:.1e}, n = 64)")
for a in ax:
    a.axhline(STRICT, **REF_RULE)
    a.axhline(DIAG, **REF_DIAG)
    a.set_ylim(0.15, 0.88)
handles = [plt.Line2D([], [], marker=MK[k], ls="", color=COLORS[k]) for k in COLORS]
ax[0].legend(handles + [plt.Line2D([], [], **REF_RULE), plt.Line2D([], [], **REF_DIAG)],
             [f"{k} ({MK[k]})" for k in COLORS] +
             [f"strict plateau rule ≤ {STRICT} (dotted)", f"straight line {DIAG} (dashed)"],
             fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "comma_width_vs_endpoints.png"), dpi=130)
plt.close(fig)

print("median width final:", round(float(np.median(w_final)), 3),
      "| strict pass:", int((w_final <= STRICT).sum()), "/", len(w_final),
      "| <=0.35:", int((w_final <= 0.35).sum()))
print("saved 3 figures")
