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

# --- transition comparison + top-1 token along the sweep ----------------------------
from transformers import AutoTokenizer  # noqa: E402

from common import crossings  # noqa: E402

tok = AutoTokenizer.from_pretrained("gpt2-large")
order = PRIMARY + ["Type"]
sw = {}      # observed top-1 switch per readout: grid interval and linearly interpolated crossing
for name, _, ans_a, ans_b in READOUTS:
    t1 = D[f"top1_{name}"]
    idx = np.where(t1[:-1] != t1[1:])[0]
    # top-1 is always the Japan or Germany answer, so the switch is where p_A - p_B crosses 0
    cross = crossings(alphas, D[f"p_A_{name}"] - D[f"p_B_{name}"], 0.0) if len(idx) else []
    sw[name] = {"grid": [(float(alphas[i]), float(alphas[i + 1])) for i in idx],
                "t_switch": cross[0] if cross else None}

with open(os.path.join(RESULTS, "top1_tokens.csv"), "w") as f:
    f.write("t," + ",".join(f"top1_{n},p_top1_{n}" for n in NAMES) + "\n")
    for i, t in enumerate(alphas):
        cells = []
        for n in NAMES:
            cells.append(repr(tok.decode([int(D[f"top1_{n}"][i])])))
            cells.append(f"{max(D[f'p_A_{n}'][i], D[f'p_B_{n}'][i]):.4f}")
        f.write(f"{t:.2f}," + ",".join(cells) + "\n")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), gridspec_kw={"width_ratios": [1.0, 1.0]})
for ax, (x0, x1), zoom in zip(axes, ((0.0, 1.0), (0.40, 0.50)), (False, True)):
    for i, name in enumerate(order):
        st = T[name]
        y = len(order) - 1 - i
        c = COL[name]
        yb, ys = y + 0.18, y - 0.18          # bar row (d(t) transition) and top-1 strip row
        ax.hlines(yb, st["t10"], st["t90"], color=c, lw=3.5, alpha=0.55)
        ax.plot([st["t10"], st["t90"]], [yb, yb], marker="|", ms=10, ls="none", color=c)
        ax.plot([st["t50"]], [yb], marker=MARK[name], ms=9, color=c, mec="black", mew=0.7, zorder=5)
        # top-1 strip: open squares = Japan-side token, filled squares = Germany-side token
        t1 = D[f"top1_{name}"]
        japan = t1 == t1[0]
        ms = 7 if zoom else 3.2
        ax.plot(alphas[japan], np.full(japan.sum(), ys), ls="none", marker="s", ms=ms,
                mfc="white", mec="#444444", mew=0.8)
        if (~japan).any():
            ax.plot(alphas[~japan], np.full((~japan).sum(), ys), ls="none", marker="s", ms=ms,
                    mfc="#222222", mec="#222222", mew=0.8)
        tok_a = tok.decode([int(t1[0])]).strip()
        tok_b = tok.decode([int(t1[-1])]).strip()
        s = sw[name]
        if s["t_switch"] is not None:
            ax.plot([s["t_switch"]] * 2, [ys - 0.12, yb + 0.12], color="black", lw=1.2, ls="--", zorder=6)
        if zoom:
            if s["t_switch"] is None:
                label = f"$t_{{50}}$={st['t50']:.3f}\ntop-1 switch: none"
            else:
                lo, hi = s["grid"][0]
                label = (f"$t_{{50}}$={st['t50']:.3f}\ntop-1 switch={s['t_switch']:.3f}\n"
                         f"(last '{tok_a}' $t$={lo:.2f},\n first '{tok_b}' $t$={hi:.2f})")
            ax.text(1.03, y, label, va="center", fontsize=8, transform=ax.get_yaxis_transform())
        elif s["t_switch"] is None:
            ax.text(0.5, ys - 0.22, f"top-1 = '{tok_a}' at every $t$ (no switch)", ha="center",
                    va="center", fontsize=8)
        else:
            ax.text(0.02, ys - 0.22, f"top-1 '{tok_a}'", ha="left", va="center", fontsize=8)
            ax.text(0.98, ys - 0.22, f"top-1 '{tok_b}'", ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1] if not zoom else [])
    ax.set_ylim(-0.7, len(order) - 0.45)
    ax.set_xlim(x0 - (0.003 if zoom else 0), x1 + (0.003 if zoom else 0))
    ax.set_xlabel("interpolation position $t$  (0 = Japan, 1 = Germany)")
    if zoom:
        ax.set_xticks(np.arange(0.40, 0.501, 0.01), minor=True)
        ax.grid(axis="x", which="both", alpha=0.3, lw=0.5)
    else:
        ax.grid(axis="x", alpha=0.25, lw=0.5)
dt = T["delta_t50_primary"]
axes[0].set_title(f"(a) full sweep;  $\\Delta t_{{50}}$ (four primary) = {dt:.3f}", fontsize=10)
axes[1].set_title("(b) zoom on $0.40 \\leq t \\leq 0.50$ (grid step 0.01)", fontsize=10)
fig.tight_layout()
fig.subplots_adjust(right=0.82, wspace=0.08)
fig.savefig(os.path.join(PLOTS, "transition_comparison.png"), dpi=160)
plt.close(fig)
for n in order:
    print(f"{n:10s} t50={T[n]['t50']:.4f} switch={sw[n]['t_switch']} grid={sw[n]['grid']} "
          f"d_at_switch={np.interp(sw[n]['t_switch'], alphas, D[f'd_{n}']) if sw[n]['t_switch'] else None}")

print("saved:", sorted(os.listdir(PLOTS)))
