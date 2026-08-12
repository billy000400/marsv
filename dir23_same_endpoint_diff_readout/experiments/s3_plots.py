"""S3+S4: individual d(t) plots, overlay, immediate-prediction plot, transition comparison."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, PLOTS, READOUTS, RESULTS

D = np.load(os.path.join(RESULTS, "interp.npz"))
T = json.load(open(os.path.join(RESULTS, "transitions.json")))
alphas = D["alphas"]
NAMES = [n for n, _, _, _ in READOUTS]
PRIMARY = ["Capital", "Continent", "Currency", "Language"]
STYLE = {"Capital": "-", "Continent": "--", "Currency": ":", "Language": "-.", "Type": (0, (3, 1, 1, 1, 1, 1))}
MARK = {"Capital": "o", "Continent": "s", "Currency": "^", "Language": "D", "Type": "v"}
COL = {n: CVD[i] for i, n in enumerate(NAMES)}
COL["Type"] = "#7f7f7f"


def mark_thresholds(ax, st, color):
    for key, lv in (("t10", 0.1), ("t50", 0.5), ("t90", 0.9)):
        t = st[key]
        if t is None:
            continue
        ax.plot([t], [lv], marker="o", ms=7, mfc="white", mec=color, mew=1.8, zorder=5)
        ax.annotate(f"$t_{{{key[1:]}}}$={t:.3f}", (t, lv), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=color)


# --- individual d(t) plots ---------------------------------------------------
for name, _, ans_a, ans_b in READOUTS:
    d = D[f"d_{name}"]
    st = T[name]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot(alphas, alphas, color="#999999", lw=1.0, ls=(0, (1, 2)), label="linear reference $d=t$")
    ax.plot(alphas, d, color=COL[name], ls=STYLE[name], lw=2.0, label=f"{name} $d(t)$")
    mark_thresholds(ax, st, COL[name])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("interpolation position $t$  (0 = Japan, 1 = Germany)")
    ax.set_ylabel("normalized logit distance $d(t)$")
    ax.set_title(f"{name}: {ans_a.strip()} $\\rightarrow$ {ans_b.strip()}\n"
                 f"$t_{{50}}$={st['t50']:.3f}, width $w$={st['w']:.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"distance_{name.lower()}.png"), dpi=160)
    plt.close(fig)

# --- overlay ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(alphas, alphas, color="#999999", lw=1.0, ls=(0, (1, 2)), label="linear reference $d=t$")
for name in NAMES:
    ax.plot(alphas, D[f"d_{name}"], color=COL[name], ls=STYLE[name], lw=2.0,
            marker=MARK[name], markevery=10, ms=4, label=name)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("interpolation position $t$  (0 = Japan, 1 = Germany)")
ax.set_ylabel("normalized logit distance $d(t)$")
ax.set_title("All five readouts follow the same sharp transition", fontsize=11)
ax.legend(fontsize=9, loc="upper left", frameon=False)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "distance_overlay.png"), dpi=160)
plt.close(fig)

# --- immediate prediction ------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(alphas, D["immediate_p_newline"], color=CVD[0], ls="-", lw=2.0, marker="o",
        markevery=10, ms=4, label="$p(\\mathrm{newline})$ at the country position")
ax.axvline(np.mean([T[n]["t50"] for n in PRIMARY]), color="#7f7f7f", ls="--", lw=1.2,
           label="mean readout $t_{50}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("interpolation position $t$  (0 = Japan, 1 = Germany)")
ax.set_ylabel("probability of the next token")
ax.set_title("The immediate next-token prediction barely moves\n"
             "(newline is top-1 at every $t$)", fontsize=11)
ax.legend(fontsize=9, loc="lower left", frameon=False)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "immediate_prediction.png"), dpi=160)
plt.close(fig)

# --- transition comparison -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
order = PRIMARY + ["Type"]
for i, name in enumerate(order):
    st = T[name]
    y = len(order) - 1 - i
    c = COL[name]
    ax.hlines(y, st["t10"], st["t90"], color=c, lw=3.5, alpha=0.55)
    ax.plot([st["t10"], st["t90"]], [y, y], marker="|", ms=12, ls="none", color=c)
    ax.plot([st["t50"]], [y], marker=MARK[name], ms=9, color=c, mec="black", mew=0.7, zorder=5)
    ax.text(st["t90"] + 0.015, y, f"$t_{{50}}$={st['t50']:.3f},  $w$={st['w']:.3f}",
            va="center", fontsize=9, color=c)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order[::-1])
ax.set_xlim(0, 1)
ax.set_xlabel("interpolation position $t$  (0 = Japan, 1 = Germany)")
dt = T["delta_t50_primary"]
ax.set_title(f"Transition location per readout: marker = $t_{{50}}$, bar = $[t_{{10}},t_{{90}}]$\n"
             f"$\\Delta t_{{50}}$ across the four primary readouts = {dt:.3f}", fontsize=10)
ax.grid(axis="x", alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "transition_comparison.png"), dpi=160)
plt.close(fig)

print("saved:", sorted(os.listdir(PLOTS)))
