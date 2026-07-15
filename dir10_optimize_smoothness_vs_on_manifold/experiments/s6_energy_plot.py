"""Per-pair energy trade-off figure from results/allpairs_summary.json (no model needed).

Shows, for all seven adjacent weekday pairs, E_act vs E_out of the optimized-path family
(linear chord + coarse lambda grid + output-only) against the fitted centroid spline. The
point of the figure is that the spline sits in the dominated top-right corner (highest
activation AND behavior kinetic energy) for every pair.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
summary = json.load(open(os.path.join(HERE, "results", "allpairs_summary.json")))

fig, ax = plt.subplots(figsize=(7.2, 5.4))
cmap = plt.get_cmap("tab10")

for j, (pair, d) in enumerate(summary.items()):
    c = cmap(j % 10)
    lams = sorted(d["lam"].keys(), key=float)
    ea = [d["lam"][l]["E_act"] for l in lams] + [d["output_only"]["E_act"]]
    eo = [d["lam"][l]["E_out"] for l in lams] + [d["output_only"]["E_out"]]
    # optimized-path family (chord through output-only)
    ax.plot(ea, eo, "-o", color=c, ms=3.5, lw=1.0, alpha=0.85,
            label=pair.replace("-", "→"))
    # fitted centroid spline (the target) as a star in the dominated corner
    ax.plot(d["spline_E_act"], d["spline_E_out"], "*", color=c, ms=15,
            markeredgecolor="black", markeredgewidth=0.6)

ax.set_xlabel(r"$E_{\mathrm{act}}$  (activation kinetic energy)  —  lower = smoother")
ax.set_ylabel(r"$E_{\mathrm{out}}$  (behavior kinetic energy)  —  lower = smoother")
ax.set_title("Energy trade-off, all 7 adjacent weekday pairs\n"
             "lines = optimized paths (λ grid + output-only);  ★ = fitted centroid spline (target)")
ax.grid(True, alpha=0.3)
ax.annotate("fitted splines sit in the\ndominated top-right corner\n(highest E_act AND E_out)",
            xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)
fig.tight_layout()
out = os.path.join(HERE, "plots", "s6_allpairs_energy_tradeoff.png")
fig.savefig(out, dpi=130)
plt.close(fig)
print("wrote", out)

# quick numeric confirmation of domination for the report
n_dom = 0
for pair, d in summary.items():
    best_ea = min([d["lam"][l]["E_act"] for l in d["lam"]] + [d["output_only"]["E_act"]])
    best_eo = min([d["lam"][l]["E_out"] for l in d["lam"]] + [d["output_only"]["E_out"]])
    dominated = d["spline_E_act"] > best_ea and d["spline_E_out"] > best_eo
    n_dom += dominated
print(f"spline dominated in both energies: {n_dom}/{len(summary)} pairs")
