"""Figure for the cross-checkpoint transplant (results/ckpt_transplant_summary.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PLOTS, RESULTS, CVD

S = json.load(open(f"{RESULTS}/ckpt_transplant_summary.json"))
EARLY, FINAL = S["early"], S["final"]
LABEL = {"base": "no write (baseline)", "self": "own $m_u$ written back",
         "donor": "donor $m_u$, as measured", "donor_scaled": "donor $m_u$, norm-matched",
         "shuffle": "donor $m_u$, token identity shuffled",
         "shuffle_scaled": "shuffled, norm-matched"}
ORDER = ["base", "self", "donor", "donor_scaled", "shuffle", "shuffle_scaled"]

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9), sharex=True)
for k, (recip, don) in enumerate([(EARLY, FINAL), (FINAL, EARLY)]):
    tag = f"{recip}<-{don}"
    rows = {r["cond"]: r for r in S["rows"] if r["direction"] == tag}
    conds = [c for c in ORDER if c in rows]
    y = np.arange(len(conds))[::-1]
    a = ax[k]

    lo = np.array([rows[c]["rho_donor"] - rows[c]["rho_donor_ci"][0] for c in conds])
    hi = np.array([rows[c]["rho_donor_ci"][1] - rows[c]["rho_donor"] for c in conds])
    a.errorbar([rows[c]["rho_donor"] for c in conds], y + 0.13, xerr=[lo, hi], fmt="o",
               color=CVD[0], ls="none", capsize=3, ms=7,
               label=f"agreement with the {don} ordering")
    a.plot([rows[c]["rho_recipient_base"] for c in conds], y - 0.13, "s", color=CVD[1],
           ms=7, mfc="none", mew=1.8, label=f"agreement with this model's own ({recip}) ordering")
    for c, yy in zip(conds, y):
        a.text(1.28, yy, f"{rows[c]['bits']:.3f}", fontsize=7.5, va="center", color="0.35")

    a.axvline(0, color="0.75", lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels([LABEL[c] for c in conds], fontsize=8.5)
    a.set_ylim(-0.6, len(conds) - 0.1)
    a.set_xlim(-0.4, 1.5)
    a.set_xticks(np.arange(-0.4, 1.05, 0.2))
    a.set_xlabel(r"Spearman $\rho$ over the 123 tokens")
    a.text(0.905, 1.01, "output shift\n(bits)", fontsize=7, color="0.35", va="bottom",
           ha="center", transform=a.transAxes)
    a.set_title(f"write $m_u$ from {don} into {recip}"
                + ("\n(does the ordering appear?)" if k == 0 else "\n(does it go away?)"),
                fontsize=10, pad=22)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="lower center", frameon=False,
               bbox_to_anchor=(0.29 + 0.5 * k, -0.005), ncol=1)

fig.suptitle("Transplanting the block-0 MLP output between training checkpoints of Pythia-410M",
             fontsize=11)
fig.subplots_adjust(left=0.19, right=0.985, top=0.79, bottom=0.28, wspace=0.55)
fig.savefig(f"{PLOTS}/ckpt_transplant.png", dpi=150)
plt.close(fig)
print(f"wrote {PLOTS}/ckpt_transplant.png")
