"""Figure 16: where a random embedding edit's output change lands, and what each kind of edit does."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, RESULTS

r = json.load(open(f"{RESULTS}/mode_split.json"))
ks = list(r["tokens"])
S = r["summary"]

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))

a = ax[0]
for k, s in enumerate(ks):
    sc = r["tokens"][s]["scan"]
    a.scatter(sc["bits"], sc["top_share"], s=14, marker="o", alpha=0.45,
              color=CVD[0], edgecolor="none", label="random direction, norm 1.8" if k == 0 else None)
for k, (tag, c, m) in enumerate((("top_heavy", CVD[1], "^"), ("tail_heavy", CVD[2], "s"))):
    e = [r["tokens"][s]["edits"][tag] for s in ks]
    a.scatter([d["bits"] for d in e], [d["top_share"] for d in e], s=48, marker=m, color=c,
              edgecolor="black", linewidth=0.5, zorder=3,
              label=f"{tag.replace('_', '-')} edit, rescaled to {r['target_bits']} bits")
a.axhline(S["top_mass"], ls="--", lw=1.2, color="0.25")
a.text(a.get_xlim()[1], S["top_mass"], f" base mass in top-{r['topk']} = {S['top_mass']:.2f}",
       va="bottom", ha="right", fontsize=8, color="0.25")
a.set_xscale("log")
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel("output movement of the edit (bits, log scale)")
a.set_ylabel(f"share of the JSD on the top-{r['topk']} successors")
a.set_title("where an embedding edit moves the output")
a.legend(fontsize=8, loc="lower left")

b = ax[1]
base = np.array([r["tokens"][s]["base_w"] for s in ks])
lim = [0.32, 0.78]
b.plot(lim, lim, ls=":", color="0.4", lw=1.2, label="no change")
for tag, c, m in (("top_heavy", CVD[1], "^"), ("tail_heavy", CVD[2], "s")):
    w = np.array([r["tokens"][s]["edits"][tag]["w"] for s in ks])
    rho = S["edits"][tag]["rho_base"][0]
    b.scatter(base, w, s=44, marker=m, color=c, edgecolor="black", linewidth=0.4,
              label=f"{tag.replace('_', '-')}: $\\rho$ = {rho:+.2f}")
    p = np.polyfit(base, w, 1)
    xs = np.array(lim)
    b.plot(xs, np.polyval(p, xs), color=c, lw=1.6,
           ls="-" if tag == "top_heavy" else "--")
b.set_xlim(*lim)
b.set_xlabel(r"anchor width $\hat w_u$ before the edit")
b.set_ylabel(r"anchor width $\hat w_u$ after the edit")
b.set_title(f"both edits move the output by {r['target_bits']} bits")
b.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{PLOTS}/mode_split.png", dpi=150)
plt.close(fig)
print("wrote plots/mode_split.png")
