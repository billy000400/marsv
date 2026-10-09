"""S4: figures from saved results (no model needed).
plots/fig1_baseline_plateau.png, plots/fig2_intervention.png, plots/fig3_path_comparison.png"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CVD, OUT, PLOTS

GRAY = "#7A7A7A"
LAYER = 26  # layer with the sharpest baseline boundary at the country position
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "lines.markersize": 5,
                     "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=CVD)})


def fig1():
    z = np.load(os.path.join(OUT, "s1_path.npz"))
    ts, step, d = z["ts"], z["step"], z["d"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    rel = step / step.mean(1, keepdims=True)
    im = ax[0].imshow(rel, aspect="auto", origin="lower", cmap="cividis", extent=[0, 1, -0.5, 27.5])
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label="adjacent-step L2 / layer mean")
    ax[0].set(xlabel="interpolation position t (0 = Japan, 1 = Germany)", ylabel="layer (block output, country token)",
              title="(a) Where the country-token state moves fastest")
    for L, ls, mk, c in [(0, ":", "o", CVD[0]), (16, "--", "s", CVD[1]), (26, "-", "^", CVD[2])]:
        ax[1].plot(ts, d[L], ls=ls, marker=mk, markevery=4, color=c, label=f"layer {L}")
    ax[1].set(xlabel="interpolation position t", ylabel="relative distance d(t) from the Japan state",
              title="(b) Distance curves at the country token")
    ax[1].legend(loc="upper left")
    for key, lab, ls, mk, c in [("p_tokyo", 'P(" Tokyo") after " is"', "-", "o", CVD[0]),
                                ("p_berlin", 'P(" Berlin") after " is"', "--", "s", CVD[1]),
                                ("p_is", 'P(" is") at the country token', ":", "^", CVD[2])]:
        ax[2].plot(ts, z[key], ls=ls, marker=mk, markevery=4, color=c, label=lab)
    ax[2].set(xlabel="interpolation position t", ylabel="probability", ylim=(-0.02, 1), title="(c) Next-token probabilities")
    ax[2].legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig1_baseline_plateau.png"), dpi=150)
    plt.close(fig)


def fig2():
    scan = list(csv.DictReader(open(os.path.join(OUT, "s2_scan.csv"))))
    ev = list(csv.DictReader(open(os.path.join(OUT, "s2_frozen_eval.csv"))))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for a, (dname, title) in zip(ax[:2], [("to_Berlin", '(a) Japan prompt, pushed toward " Berlin"'),
                                          ("to_Tokyo", '(b) Germany prompt, pushed toward " Tokyo"')]):
        for site, lab, ls, mk, c in [("country", "edit at the country token", "-", "o", CVD[0]),
                                     ("is", 'edit at the " is" token (positive control)', "--", "s", CVD[1])]:
            rr = [r for r in scan if r["direction"] == dname and r["site"] == site and r["k"] == "10" and r["beta"] == "1.0"
                  and "+" not in r["layers"]]
            a.plot([int(r["layers"]) for r in rr], [float(r["signed_shift"]) for r in rr], ls=ls, marker=mk, color=c, label=lab)
        need = abs(float(next(r["base_margin"] for r in scan if r["direction"] == dname)))
        a.axhline(need, color=GRAY, ls=":", lw=1.2)
        a.annotate(f"shift needed to reverse the answer ({need:.1f})", (0, need), textcoords="offset points", xytext=(0, 4),
                   fontsize=8, color="#333333")
        a.set(xlabel="edited layer (top-10 features, beta = 1)", ylabel="margin shift toward the donor's capital (logits)",
              title=title, ylim=(-0.3, 11.5))
        a.legend(loc="center left")
    betas = [0.0, 0.5, 1.0, 2.0]
    for dname, lab, ls, mk, c in [("to_Berlin", "Berlin-directed edit (Japan prompt)", "-", "o", CVD[0]),
                                  ("to_Tokyo", "Tokyo-directed edit (Germany prompt)", "--", "s", CVD[1])]:
        rr = [r for r in ev if r["template"] == "original" and r["direction"] == dname]
        base = float(next(r["margin_toward_target"] for r in rr if r["arm"] == "none"))
        get = lambda arm, b: base if b == 0 else float(next(r["margin_toward_target"] for r in rr if r["arm"] == arm and float(r["beta"]) == b))
        ax[2].plot(betas, [get("edit", b) for b in betas], ls=ls, marker=mk, color=c, label=lab)
        ctrl = np.array([[get(f"control{s}", b) for b in betas] for s in (1, 2, 3)])
        ax[2].errorbar(betas, ctrl.mean(0), yerr=[ctrl.mean(0) - ctrl.min(0), ctrl.max(0) - ctrl.mean(0)], ls=ls, marker=mk,
                       mfc="white", color=GRAY, capsize=3, lw=1.2, label=f"random control, same size ({lab.split('-')[0]}-matched)")
    ax[2].axhline(0, color=GRAY, ls=":", lw=1.2)
    ax[2].annotate("above this line the other capital is preferred", (0, 0), textcoords="offset points", xytext=(0, 4), fontsize=8,
                   color="#333333")
    ax[2].set(xlabel="edit strength beta (1 = donor values, 2 = extrapolation)", xticks=betas,
              ylabel="margin toward the donor's capital (logits)",
              title='(c) Frozen layer-16 edit on the reserved prompt\n"The capital of <country> is"')
    ax[2].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig2_intervention.png"), dpi=150)
    plt.close(fig)


def fig3():
    z = np.load(os.path.join(OUT, "s3_path.npz"))
    cfg = json.load(open(os.path.join(OUT, "frozen_edit.json")))
    ts, mid = z["ts"], (z["ts"][1:] + z["ts"][:-1]) / 2
    main = [("baseline", "no edit", "-", "o", CVD[0]), ("to_Tokyo", "Tokyo-directed edit", "--", "s", CVD[1]),
            ("to_Berlin", "Berlin-directed edit", "-.", "^", CVD[2])]
    ctrl = [("to_Tokyo", "random controls matched to the Tokyo-directed edit (3 seeds)", ":"),
            ("to_Berlin", "random controls matched to the Berlin-directed edit (3 seeds)", (0, (5, 3)))]
    panels = [("d", ts, lambda v: v[LAYER], f"1a. Relative distance d(t), layer {LAYER}, country token", "d(t) (fixed unedited references)"),
              ("step", mid, lambda v: v[LAYER], f"1b. Adjacent-step change, layer {LAYER}, country token", "L2 change per step of 0.025"),
              ("probe", ts, lambda v: v[LAYER], f"2a. Country readout score, layer {LAYER}", "score (+1 Japan, -1 Germany)"),
              ("edit_l2", ts, lambda v: v[0], f"2b. Size of the layer-{cfg['layers'][0]} edit", "L2 norm of the added vector"),
              ("p_tokyo", ts, lambda v: v, '3a. P(" Tokyo") after the bridge " is"', "probability"),
              ("p_berlin", ts, lambda v: v, '3b. P(" Berlin") after the bridge " is"', "probability")]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10), sharex=True)
    for a, (key, x, pick, title, ylab) in zip(axes.flat, panels):
        for dname, lab, ls in ctrl:
            for s in (1, 2, 3):
                a.plot(x, pick(z[f"{dname}_control{s}/{key}"]), color=GRAY, ls=ls, lw=1.0, label=lab if s == 1 else None)
        for cname, lab, ls, mk, c in main:
            a.plot(x, pick(z[f"{cname}/{key}"]), color=c, ls=ls, marker=mk, markevery=4, label=lab)
        a.set_title(title, fontsize=9.5, loc="left")
        a.set_ylabel(ylab)
        if key == "probe":
            a.axhline(0, color=GRAY, lw=0.8)
    for a in axes[-1]:
        a.set_xlabel("interpolation position t (0 = Japan state, 1 = Germany state, at the layer-0 output)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[2:], l[2:], loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.legend(h[:2], l[:2], loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(PLOTS, "fig3_path_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
