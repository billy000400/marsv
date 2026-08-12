"""Figure for the redefined basin criterion (operator feedback #6): what the rest ratio measures,
how the old and new criteria behave on null curves, and the per-character basin fraction."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvd_style import CVD, REF_DIAG, use_cvd  # noqa: E402
from matthew_assay import pava_isotonic  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
use_cvd()

B = json.load(open(os.path.join(RES, "basin_criterion.json")))
S = json.load(open(os.path.join(RES, "allpairs_summary.json")))
z = np.load(os.path.join(RES, "allpairs_raw.npz"))
ts = z["ts"]
DELTA, KAPPA = B["definition"]["delta"], B["definition"]["kappa"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

# --- A: what the rest length is, on a real trained curve vs the straight-line null -------------
ax = axes[0]
pairs = [(p["i"], p["j"], p["w"]) for p in S["final_block0"]]
i, j, _ = sorted(pairs, key=lambda p: p[2])[len(pairs) // 20]     # a typical-sharp trained pair
d = pava_isotonic(z[f"final|L0|d|{i}_{j}"].astype(float))
chars = S["chars"]
ax.plot(ts, d, color=CVD[0], ls="-", lw=2.2, marker="o", ms=3,
        label=f"trained pair ({chars[i]!r}→{chars[j]!r})")
ax.plot(ts, ts, **REF_DIAG, label="straight-line null $d(t)=t$")
ax.axhline(DELTA, color="0.7", lw=0.9)
ax.axhline(1 - DELTA, color="0.7", lw=0.9)
r_A = float(ts[np.argmax(d >= DELTA)])
ax.annotate("", xy=(r_A, 0.045), xytext=(0, 0.045),
            arrowprops=dict(arrowstyle="<->", color=CVD[0], lw=1.8))
ax.text(r_A / 2, 0.062, f"trained $r_A$={r_A:.2f}", color=CVD[0], ha="center", fontsize=9)
ax.annotate("", xy=(DELTA, 0.17), xytext=(0, 0.17),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.8))
ax.text(0.13, 0.165, f"null $r_A$={DELTA:.2f}", color="0.35", ha="left", fontsize=9)
ax.set_xlabel("interpolation position $t$")
ax.set_ylabel("relative distance $d(t)$  (isotonic)")
ax.set_title("A. Rest length $r$ vs the straight-line null", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# --- B: pass rate vs kappa, trained against every null ----------------------------------------
ax = axes[1]
kap = np.array(B["kappas"])
series = [("trained (block 0, step 30000)", "trained network", CVD[0], "-", "o"),
          ("untrained (step 0)", "untrained network (step 0)", CVD[1], "--", "s"),
          ("block-11 patch (trained)", "block-11 patch (near-linear)", CVD[2], "-.", "^"),
          ("line + noise sigma=0.05", "line + noise ($\\sigma$=0.05)", CVD[3], ":", "D")]
for key, lab, col, ls, mk in series:
    ax.plot(kap, B["pass_rate_vs_kappa"][key], color=col, ls=ls, marker=mk, ms=4, lw=2, label=lab)
ax.axvline(1.0, color="0.45", ls="--", lw=1.4)
ax.text(1.06, 0.52, "old criterion:\n$\\kappa=1$,\nthe null itself", fontsize=8, color="0.35")
ax.axvline(KAPPA, **{"color": "k", "ls": ":", "lw": 1.6})
ax.text(KAPPA + 0.08, 0.05, f"adopted $\\kappa$={KAPPA:g}", fontsize=9)
ax.set_xlabel("strictness $\\kappa$  (basin iff rest ratio $R \\geq \\kappa$)")
ax.set_ylabel("fraction of endpoints passing")
ax.set_title("B. Old criterion sits on the null; new one clears it", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(-0.03, 1.05)

# --- C: per-character basin fraction, sorted, with the untrained control ----------------------
ax = axes[2]
pc = sorted(B["per_char"], key=lambda p: p["phi_new"])
x = np.arange(len(pc))
lab = ["\\n" if p["char"] == "\n" else ("sp" if p["char"] == " " else p["char"]) for p in pc]
ax.bar(x, [p["phi_new"] for p in pc], color=CVD[0], edgecolor="k", lw=0.3, hatch="//",
       label="$\\phi(c)$, trained")
ax.plot(x, [p["phi_old"] for p in pc], color=CVD[1], ls="none", marker="D", ms=4,
        label="old (unvalidated) criterion")
ax.plot(x, [p["phi_new_untrained"] for p in pc], color=CVD[2], ls="none", marker="v", ms=4,
        label="$\\phi(c)$, untrained network")
ax.axhline(0.5, color="k", ls=":", lw=1.4)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6, rotation=90)
ax.set_xlabel("character $c$ (sorted by $\\phi$)")
ax.set_ylabel("basin fraction $\\phi(c)$")
ax.set_title("C. 59 / 65 characters hold a basin; the 6 failures are the rarest", fontsize=11)
ax.legend(fontsize=8, loc="lower right", framealpha=0.95, bbox_to_anchor=(1.0, 0.06))
ax.set_ylim(-0.05, 1.10); ax.set_xlim(-1, len(pc))

fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "basin_criterion.png"), dpi=150)
plt.close(fig)

# --- second figure: phi_new against training frequency ----------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.6))
f = np.array([p["train_count"] for p in B["per_char"]], dtype=float)
p_new = np.array([p["phi_new"] for p in B["per_char"]])
ax.scatter(f, p_new, s=26, color=CVD[0], marker="o", edgecolor="k", lw=0.3)
for p in B["per_char"]:
    if p["phi_new"] < 0.95:
        ax.annotate("\\n" if p["char"] == "\n" else p["char"], (p["train_count"], p["phi_new"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=8)
ax.axvline(1000, color="0.45", ls="--", lw=1.4)
ax.text(1100, 0.05, "1000 training\noccurrences", fontsize=8, color="0.35")
rho, pv = B["per_char_summary"]["spearman_phi_new_vs_train_count"]
ax.set_xscale("log")
ax.set_xlabel("training-set occurrences of $c$ (log scale)")
ax.set_ylabel("basin fraction $\\phi(c)$")
ax.set_title(f"Basin ownership tracks how often the character was seen\nSpearman $\\rho$={rho:.2f}, p={pv:.1e}, n=65", fontsize=10)
ax.set_ylim(-0.05, 1.08)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "basin_vs_frequency.png"), dpi=150)
plt.close(fig)
print("wrote plots/basin_criterion.png, plots/basin_vs_frequency.png")
