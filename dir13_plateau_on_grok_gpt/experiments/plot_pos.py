"""Figure for S24: what happens to the plateau when the readout is not the patched position.

Three panels on the final (step-30,000) reference checkpoint:
  left   median d(t) over the 150 pairs, one curve per readout offset k (IQR band on k=0 and k=8);
  middle median transition width w vs k, trained against the untrained baseline at the same k;
  right  median endpoint logit separation vs k -- the signal the readout still has to work with.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, LINESTYLES, MARKERS, REF_RULE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")

blob = json.load(open(os.path.join(RES, "pos_assay.json")))
raw = np.load(os.path.join(RES, "pos_assay_raw.npz"))
ts = raw["ts"]
KS = blob["ks"]
WHICH = "last" if "last" in blob["checkpoints"] else "matched"
C = blob["checkpoints"][WHICH]["conditions"]
CI = blob["checkpoints"]["init"]["conditions"]

fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.5))

# ---- left: median d(t) per k -------------------------------------------------------------------
for n, k in enumerate(KS):
    D = raw[f"{WHICH}|k{k}|final"]
    med = np.median(D, axis=0)
    ax[0].plot(ts, med, color=CVD[n], ls=LINESTYLES[n % 4], lw=2.0,
               marker=MARKERS[n % 4], markevery=7, ms=5, label=f"k = {k}")
    if k in (0, 8):
        ax[0].fill_between(ts, np.percentile(D, 25, axis=0), np.percentile(D, 75, axis=0),
                           color=CVD[n], alpha=0.16, lw=0)
ax[0].plot([0, 1], [0, 1], color="0.45", ls="--", lw=1.4)
ax[0].annotate("straight path\n(no plateau)", xy=(0.62, 0.62), xytext=(0.60, 0.30),
               fontsize=8, color="0.35", ha="center",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
ax[0].set_xlabel("interpolation position  $t$")
ax[0].set_ylabel("relative distance  $d(t)$  (final-logit space)")
ax[0].set_title("median path, readout $k$ characters after the patch", fontsize=10)
ax[0].legend(fontsize=8, loc="upper left", title="readout offset", title_fontsize=8)
ax[0].set_xlim(0, 1); ax[0].set_ylim(-0.02, 1.02)

# ---- middle: width vs k ------------------------------------------------------------------------
wt = [C[f"k{k}"]["read_final"]["median_w"] for k in KS]
lo = [C[f"k{k}"]["read_final"]["iqr_w"][0] for k in KS]
hi = [C[f"k{k}"]["read_final"]["iqr_w"][1] for k in KS]
wi = [CI[f"k{k}"]["read_final"]["median_w"] for k in KS]
ax[1].errorbar(KS, wt, yerr=[np.array(wt) - np.array(lo), np.array(hi) - np.array(wt)],
               color=CVD[0], ls="-", marker="o", ms=7, lw=2.0, capsize=4, label="trained (step 30,000)")
ax[1].plot(KS, wi, color=CVD[1], ls="--", marker="s", ms=7, lw=2.0, label="untrained (step 0)")
ax[1].axhline(C["k0"]["read_patch"]["median_w"], **REF_RULE)
ax[1].annotate("readout at the patched position\n(same forward pass, every $k$)",
               xy=(4.0, C["k0"]["read_patch"]["median_w"]), xytext=(1.4, 0.16), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
ax[1].set_xlabel("readout offset  $k$  (characters between patch and readout)")
ax[1].set_ylabel("median transition width  $w_{10\\to90}$")
ax[1].set_title("the plateau widens with distance but survives", fontsize=10)
ax[1].set_xticks(KS); ax[1].set_ylim(0, 1.0)
ax[1].legend(fontsize=8, loc="center right")

# ---- right: endpoint separation vs k -----------------------------------------------------------
sep = [C[f"k{k}"]["read_final"]["median_sep"] for k in KS]
stf = [100 * C[f"k{k}"]["read_final"]["strict_frac"] for k in KS]
ax[2].plot(KS, sep, color=CVD[0], ls="-", marker="o", ms=7, lw=2.0, label="endpoint separation (left axis)")
ax[2].set_xlabel("readout offset  $k$")
ax[2].set_ylabel("median $\\Vert x_A - x_B\\Vert_2$  (logit units)", color=CVD[0])
ax[2].set_xticks(KS)
ax2 = ax[2].twinx()
ax2.plot(KS, stf, color=CVD[1], ls="--", marker="s", ms=7, lw=2.0, label="strict plateau rate (right axis)")
ax2.set_ylabel("strict plateau rate (% of pairs)", color=CVD[1])
ax2.set_ylim(0, max(stf) * 1.35 + 1)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
ax[2].set_title("the signal the later readout has left", fontsize=10)

fig.suptitle("Interpolating a character that is not the position the readout reads "
             f"(reference character GPT, step {blob['checkpoints'][WHICH]['step']:,}, 150 pairs, "
             'context "The house was ", filler " and then")', fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(PLOTS, "pos_offset.png"), dpi=150)
plt.close(fig)
print("wrote plots/pos_offset.png")
