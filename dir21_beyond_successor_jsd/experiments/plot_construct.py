"""Figure 17: constructed top-heavy vs tail-heavy edits — how far apart in S, and what they do to width."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

r = json.load(open(f"{RESULTS}/mode_construct.json"))
old = json.load(open(f"{RESULTS}/mode_split.json"))
ks = list(r["tokens"])
S = r["summary"]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))

a = ax[0]
x = np.arange(len(ks))
rand = np.array([[np.min(old["tokens"][s]["scan"]["top_share"]),
                  np.max(old["tokens"][s]["scan"]["top_share"])] for s in old["tokens"]])
a.fill_between([-0.5, len(ks) - 0.5], rand[:, 0].mean(), rand[:, 1].mean(),
               color="0.8", hatch="//", edgecolor="0.5", linewidth=0.0, zorder=0,
               label="range spanned by 24 random directions")
for tag, c, m in (("top_heavy", CVD[1], "^"), ("tail_heavy", CVD[2], "s")):
    j = 0 if tag == "top_heavy" else 1
    pred = np.array([r["tokens"][s]["predicted_share"][j] for s in ks])
    got = np.array([r["tokens"][s]["edits"][tag]["top_share"] for s in ks])
    a.scatter(x, pred, s=54, marker=m, facecolor="none", edgecolor=c, linewidth=1.5,
              label=f"{tag.replace('_', '-')}: predicted (small step)")
    a.scatter(x, got, s=54, marker=m, color=c, edgecolor="black", linewidth=0.4, zorder=3,
              label=f"{tag.replace('_', '-')}: achieved at {r['target_bits']} bits")
    for xi, p, g in zip(x, pred, got):
        a.plot([xi, xi], [p, g], color=c, lw=0.9, ls=":")
a.axhline(old["summary"]["top_mass"], color="0.2", ls="-.", lw=1.4)
a.text(len(ks) - 0.6, old["summary"]["top_mass"], f"base mass in top-{r['topk']} = "
       f"{old['summary']['top_mass']:.2f} ", fontsize=8, color="0.2", ha="right", va="bottom")
a.set_xticks(x)
a.set_xticklabels([s.strip() for s in ks], rotation=60, ha="right", fontsize=7)
a.set_ylim(0, 1.02)
a.set_xlim(-0.5, len(ks) - 0.5)
a.set_ylabel(f"share $S$ of the output change on the top-{r['topk']} successors")
a.set_title("the split is steerable in the small-step regime only")
a.legend(fontsize=7, loc="upper center", ncol=2)

b = ax[1]
base = np.array([r["tokens"][s]["base_w"] for s in ks])
lim = [0.32, 0.78]
b.plot(lim, lim, ls=":", color="0.4", lw=1.2, label="no change")
for tag, c, m, ls in (("top_heavy", CVD[1], "^", "-"), ("tail_heavy", CVD[2], "s", "--")):
    w = np.array([r["tokens"][s]["edits"][tag]["w"] for s in ks])
    o = S["edits"][tag]
    b.scatter(base, w, s=46, marker=m, color=c, edgecolor="black", linewidth=0.4,
              label=f"{tag.replace('_', '-')} ($S$ = {o['top_share']:.2f}): "
                    f"$\\rho$ = {o['rho_base'][0]:+.2f}")
    b.plot(np.array(lim), np.polyval(np.polyfit(base, w, 1), np.array(lim)), color=c, lw=1.6, ls=ls)
b.set_xlim(*lim)
b.set_xlabel(r"anchor width $\hat w_u$ before the edit")
b.set_ylabel(r"anchor width $\hat w_u$ after the edit")
b.set_title(f"both edits matched at {r['target_bits']} bits of output movement")
b.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{PLOTS}/mode_construct.png", dpi=150)
plt.close(fig)
print("wrote plots/mode_construct.png")
