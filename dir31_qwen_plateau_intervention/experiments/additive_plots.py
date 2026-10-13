"""Figures for stage S4 from the CSVs in results/additive/ (no model runs). Writes plots/additive/*.png.
Series are told apart by marker or hatch as well as colour (CLAUDE.md rule 13)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
from additive_common import MIN_EFFECT, OUT, PLOTS, TOL_PROB, TOL_READOUT_FRAC, read_csv

BETA_STYLE = {0.5: (C.CVD[0], "o"), 1.0: (C.CVD[1], "s"), 2.0: (C.CVD[2], "^")}
KIND_STYLE = {"single": (C.CVD[0], "o", "one feature"), "top3": (C.CVD[1], "s", "top-3 group"), "top5": (C.CVD[2], "^", "top-5 group")}
HATCH = ["", "//", "..", "\\\\"]


def num(rows, keys):
    for r in rows:
        for k in keys:
            r[k] = float(r[k])
        r["pass_all"] = r["pass_all"] == "True"
    return rows


def main():
    os.makedirs(PLOTS, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.prop_cycle": plt.cycler(color=C.CVD)})
    keys = ["layer", "beta", "answer_effect", "d_readout26_frac_of_gap", "control_max_fall", "is_fall", "edit_l2"]
    rows = num(read_csv("screen_single.csv") + read_csv("screen_group.csv"), keys)
    rows = [r for r in rows if r["beta"] > 0]
    cand = read_csv("candidates.csv")
    site = {int(r["layer"]): float(r["native_mlp_out_l2"]) for r in read_csv("site_norms.csv")}
    counts = json.load(open(os.path.join(OUT, "candidate_counts.json")))["candidates_per_layer"]

    # ---- candidates: how many per layer, and how large the added vector is next to the native MLP output ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    layers = sorted(int(k) for k in counts)
    axes[0].bar(layers, [counts[str(L)] for L in layers], color=C.CVD[0], edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("candidate features")
    axes[0].set_title(f"{len(cand)} Berlin candidates outside the Germany set")
    axes[0].set_xticks(layers[::2])
    axes[1].plot(layers, [site[L] for L in layers], color=C.CVD[1], linestyle="--", marker="s", markersize=4,
                 label="native MLP output at the Japan token (L2 norm)")
    axes[1].scatter([int(r["layer"]) for r in cand], [float(r["amplitude"]) for r in cand], color=C.CVD[0], marker="o", s=22,
                    label="one candidate's added vector at beta = 1 (= reference amplitude)", zorder=3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("L2 norm (log scale)")
    axes[1].set_title("Size of the additive edit")
    axes[1].set_xticks(layers[::2])
    axes[1].set_ylim(0.1, 2000)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "figA1_candidates.png"), dpi=150)
    plt.close(fig)

    # ---- screen: answer effect of every setting, by layer, one panel per strength ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    for ax, beta in zip(axes, (0.5, 1.0, 2.0)):
        for kind, (col, mk, lab) in KIND_STYLE.items():
            sel = [r for r in rows if r["beta"] == beta and r["kind"] == kind]
            off = {"single": 0.0, "top3": -0.25, "top5": 0.25}[kind]
            ax.scatter([r["layer"] + off for r in sel], [r["answer_effect"] for r in sel], color=col, marker=mk,
                       s=26 if kind == "single" else 48, edgecolor="black", linewidth=0.4, label=lab, zorder=3)
        bad = [r for r in rows if r["beta"] == beta and not r["pass_all"]]
        ax.scatter([r["layer"] for r in bad], [r["answer_effect"] for r in bad], color="black", marker="x", s=70, linewidth=1.3,
                   label="fails a preservation check", zorder=4)
        ax.axhline(MIN_EFFECT, color="black", linestyle="--", linewidth=1)
        ax.text(1.2, MIN_EFFECT - 0.07, "pilot threshold: +1 logit toward Berlin", fontsize=8, va="top")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"beta = {beta:g}")
        ax.set_xlabel("layer of the edit (country token)")
        ax.set_xlim(0.5, 26.5)
    axes[0].set_ylabel("answer effect on the tuning prompt:\nchange in logit(Berlin) - logit(Tokyo)")
    axes[0].set_ylim(-0.45, 1.15)
    axes[2].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "figA2_screen_answer_effect.png"), dpi=150)
    plt.close(fig)

    # ---- screen: answer effect against each preservation quantity, with the fixed tolerances ----
    panels = [("d_readout26_frac_of_gap", "layer-26 country readout change\n(fraction of the Japan-Germany gap)", TOL_READOUT_FRAC, True),
              ("control_max_fall", "largest fall of the correct answer's probability\non the four Japanese control prompts", TOL_PROB, False),
              ("is_fall", "fall of P(' is') at the country token", TOL_PROB, False)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for ax, (key, lab, tol, two_sided) in zip(axes, panels):
        for beta, (col, mk) in BETA_STYLE.items():
            sel = [r for r in rows if r["beta"] == beta]
            ax.scatter([r["answer_effect"] for r in sel], [r[key] for r in sel], color=col, marker=mk, s=26, edgecolor="black",
                       linewidth=0.3, label=f"beta = {beta:g}", zorder=3)
        ax.axhline(tol, color="black", linestyle="--", linewidth=1)
        if two_sided:
            ax.axhline(-tol, color="black", linestyle="--", linewidth=1)
        ax.axvline(MIN_EFFECT, color="black", linestyle="-.", linewidth=1)
        lo = -tol if two_sided else min(-0.03, min(r[key] for r in rows) - 0.01)
        ax.fill_between([MIN_EFFECT, 1.25], lo, tol, facecolor="none", edgecolor="gray", hatch="//", linewidth=0.5)
        ax.text(1.125, (lo + tol) / 2, "within\ntolerance\nand >= +1", ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_xlim(-0.45, 1.25)
        ax.set_xlabel("answer effect: change in logit(Berlin) - logit(Tokyo)")
        ax.set_ylabel(lab)
    axes[0].set_title("dashed: tolerance (+-10% of gap)", fontsize=9)
    axes[1].set_title("dashed: tolerance (0.05)", fontsize=9)
    axes[2].set_title("dashed: tolerance (0.05); dash-dot: +1 threshold", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(PLOTS, "figA3_screen_preservation.png"), dpi=150)
    plt.close(fig)

    # ---- frozen edit and its three matched random controls: tuning prompt (where it was selected) and both reserved prompts ----
    tune = read_csv("tuning_best_controls.csv")
    fz = json.load(open(os.path.join(OUT, "frozen_edit.json")))
    tune = [r for r in tune if int(r["layer"]) == fz["layer"] and r["features"] == "+".join(map(str, fz["features"])) and float(r["beta"]) == fz["beta"]]
    res = read_csv("reserved_eval.csv")
    panels = [("tuning prompt (edit selected here)\n" + "The capital city of Japan is", tune)]
    for t in dict.fromkeys(r["template"] for r in res):
        sel = [r for r in res if r["template"] == t and r["arm"] != "unedited"]
        panels.append(("reserved prompt\n" + sel[0]["prompt"], sel))
    failed = lambda r: [n for n, k in (("readout", "pass_readout"), ("control prompts", "pass_controls"), ("' is'", "pass_is")) if r[k] == "False"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    for ax, (title, sel) in zip(axes, panels):
        assert len(sel) == 4, (title, len(sel))
        vals = [float(r["answer_effect"]) for r in sel]
        bars = ax.bar(range(4), vals, color=C.CVD[:4], edgecolor="black", linewidth=0.5)
        for b, h in zip(bars, HATCH):
            b.set_hatch(h)
        for k, (r, v) in enumerate(zip(sel, vals)):
            ax.text(k, max(v, 0) + 0.03, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)
            if failed(r):
                ax.text(k, max(v, 0) + 0.13, "fails\n" + ", ".join(failed(r)), ha="center", va="bottom", fontsize=7.5)
        ax.axhline(MIN_EFFECT, color="black", linestyle="--", linewidth=1)
        ax.text(3.45, MIN_EFFECT - 0.04, "pilot threshold +1", fontsize=8, ha="right", va="top")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(range(4), ["frozen\nedit", "random 1", "random 2", "random 3"])
        ax.set_ylim(-0.4, 1.15)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("answer effect:\nchange in logit(Berlin) - logit(Tokyo)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "figA5_reserved_pilot.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
