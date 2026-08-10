"""Experiment 5 / S9d — the six figures of PLAN.md 5.4, CVD-safe (CLAUDE.md rule 13).

Reads results/allpairs_summary.json (+ analysis block) and results/allpairs_raw.npz.
Writes plots/allpairs_{width_matrix,width_by_char,curves_small_multiples,boundary_vs_logp,
readout_decision,controls}.png.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cvd_style import CVD, LINESTYLES, MARKERS, HATCHES, REF_DIAG, REF_RULE, use_cvd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
CLASSES = ["space", "punct", "upper", "lower"]
CLASS_LABEL = {"space": "space / newline", "punct": "punctuation & digits",
               "upper": "upper case", "lower": "lower case"}
LINEAR_W = 0.80          # w_{10->90} of a perfectly straight d(t) = t path
STRICT_W = 0.25          # frozen strict-plateau width threshold


def disp(c):
    return {"\n": "\\n", " ": "␣"}.get(c, c)


CTX_NOTE = ('context = "The house was " (14 characters), patch at the final position only; '
            'step-30000 character GPT')


def save(fig, name, note=""):
    """`note` (operator feedback #5, item 4) states the prompt context and the sample count behind
    each cell / point of the figure, directly on the figure."""
    if note:
        fig.text(0.5, -0.02, f"{CTX_NOTE}. {note}", ha="center", fontsize=8)
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def main():
    use_cvd()
    S = json.load(open(os.path.join(RES, "allpairs_summary.json")))
    A = S["analysis"]
    R = np.load(os.path.join(RES, "allpairs_raw.npz"))
    ts = R["ts"]
    chars, V = S["chars"], S["vocab_size"]
    pc = {p["idx"]: p for p in A["per_char"]}
    fin = S["final_block0"]

    # class-ordered permutation of the vocabulary
    order = [i for cl in CLASSES for i in range(V) if pc[i]["class"] == cl]
    pos = {c: k for k, c in enumerate(order)}

    W = np.full((V, V), np.nan)
    for p in fin:
        if p["w"] is not None:
            W[pos[p["i"]], pos[p["j"]]] = W[pos[p["j"]], pos[p["i"]]] = p["w"]

    # ---- Figure A: 65x65 width matrix --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    im = ax.imshow(W, cmap="viridis", vmin=np.nanmin(W), vmax=np.nanmax(W))
    ax.set_xticks(range(V)); ax.set_yticks(range(V))
    ax.set_xticklabels([disp(chars[i]) for i in order], fontsize=5.5, rotation=90)
    ax.set_yticklabels([disp(chars[i]) for i in order], fontsize=5.5)
    b = 0
    for cl in CLASSES[:-1]:
        b += sum(pc[i]["class"] == cl for i in range(V))
        ax.axhline(b - .5, color="w", lw=1.2); ax.axvline(b - .5, color="w", lw=1.2)
    start = 0
    for cl in CLASSES:
        n = sum(pc[i]["class"] == cl for i in range(V))
        ax.text(start + n / 2 - .5, -2.0, CLASS_LABEL[cl], ha="center", fontsize=8)
        start += n
    ax.set_xlabel("character B"); ax.set_ylabel("character A")
    ax.set_title("Transition width $w_{10\\to90}$ for all 2080 character pairs\n"
                 "(block-0 interpolation, step-30000 char GPT; diagonal masked)", pad=26)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("$w_{10\\to90}$  (0 = instant switch, 0.80 = straight line)")
    save(fig, "allpairs_width_matrix.png",
         "each cell = 1 character pair = 1 d(t) curve of 50 interpolation points; 2080 pairs")

    # ---- Figure B: per-character width distribution + flat_frac ------------------------------
    srt = sorted(range(V), key=lambda i: pc[i]["med_w"])
    data = []
    for i in srt:
        ws = [p["w"] for p in fin if (p["i"] == i or p["j"] == i) and p["w"] is not None]
        data.append(ws)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    bp = ax.boxplot(data, widths=.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=1.3))
    for k, i in enumerate(srt):
        bp["boxes"][k].set(facecolor=CVD[0], alpha=.45,
                           hatch=HATCHES[CLASSES.index(pc[i]["class"])], edgecolor="0.25")
    ax.axhline(LINEAR_W, **REF_DIAG); ax.axhline(STRICT_W, **REF_RULE)
    ax.text(V - 0.5, LINEAR_W + .012, "straight-line reference $w=0.80$", fontsize=8,
            color="0.35", ha="right")
    ax.text(V - 0.5, STRICT_W - .028, "strict plateau rule $w\\leq0.25$", fontsize=8, ha="right")
    ax.set_ylim(0.14, 0.90)
    ax.set_xticks(range(1, V + 1))
    ax.set_xticklabels([disp(chars[i]) for i in srt], fontsize=6.5)
    ax.set_ylabel("$w_{10\\to90}$ over the 64 partners")
    ax.set_xlabel("character $c$, sorted by median width")
    ax2 = ax.twinx()
    ax2.plot(range(1, V + 1), [pc[i]["flat_frac"] for i in srt], color=CVD[1],
             ls="none", marker="D", ms=3.4, label="flat_frac(c)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("flat_frac($c$)  (diamonds)", color=CVD[1])
    ax2.tick_params(axis="y", colors=CVD[1])
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=CVD[0], alpha=.45, hatch=HATCHES[k],
                        edgecolor="0.25", label=CLASS_LABEL[cl]) for k, cl in enumerate(CLASSES)]
    ax.legend(handles=hs, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.24),
              frameon=False, title="box hatch = character class")
    ax.set_title("Every character sits in a basin (flat_frac $\\approx$ 1), but how sharply it is "
                 "left varies by character")
    save(fig, "allpairs_width_by_char.png",
         "each box / diamond summarises the 64 pairs of one character, 1 d(t) curve of 50 points per pair")

    # ---- Figure C: raw d(t) small multiples --------------------------------------------------
    reps = [srt[0], srt[-1]]                                   # sharpest and widest
    for cl in CLASSES:                                         # one per character class
        cand = [i for i in srt if pc[i]["class"] == cl and i not in reps]
        if cand:
            reps.append(cand[len(cand) // 2])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, i in zip(axes.ravel(), reps):
        for p in fin:
            if p["i"] != i and p["j"] != i:
                continue
            d = R[f"final|L0|d|{p['i']}_{p['j']}"]
            if p["j"] == i:                                     # orient so c is always endpoint A
                d = 1 - d[::-1]
            ax.plot(ts, d, color=CVD[0], lw=.5, alpha=.32)
        ax.plot([0, 1], [0, 1], **REF_DIAG)
        ax.set_title(f"'{disp(chars[i])}'  ({CLASS_LABEL[pc[i]['class']]})\n"
                     f"median $w$={pc[i]['med_w']:.3f}, flat_frac={pc[i]['flat_frac']:.2f}",
                     fontsize=9)
        ax.set_ylim(-.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("interpolation position $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative distance $d(t)$")
    fig.suptitle("Raw $d(t)$ for six characters against all 64 partners (thin lines), "
                 "oriented with the named character at $t=0$; dashed = straight-line reference")
    save(fig, "allpairs_curves_small_multiples.png",
         "each panel = 64 curves (one character against all its partners), 50 interpolation points per curve")

    # ---- Figure D: boundary location vs relative plausibility --------------------------------
    lp = np.log10(np.array(S["p_next"]) + 1e-30)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for k, cl in enumerate(CLASSES):
        x = [lp[p["i"]] - lp[p["j"]] for p in fin
             if p["t_star"] is not None and pc[p["i"]]["class"] == cl]
        y = [p["t_star"] for p in fin
             if p["t_star"] is not None and pc[p["i"]]["class"] == cl]
        ax.plot(x, y, ls="none", marker=MARKERS[k], ms=3.2, alpha=.55,
                color=CVD[k], label=CLASS_LABEL[cl] + " (endpoint A)")
    ax.axhline(0.5, **REF_RULE); ax.axvline(0.0, **REF_DIAG)
    rho, p_ = A["plausibility"]["spearman_tstar_vs_dlogp_signed"]
    ax.set_xlabel("$\\log_{10} p(A\\mid\\text{context}) - \\log_{10} p(B\\mid\\text{context})$")
    ax.set_ylabel("midpoint crossing $t^{*}$  (isotonic $t$ where $d=0.5$)")
    ax.set_title(f"Where the switch happens vs which endpoint the model prefers\n"
                 f"Spearman $\\rho$ = {rho:.2f} (p = {p_:.1e}, n = {len(fin)})")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "allpairs_boundary_vs_logp.png",
         "each marker = 1 character pair = 1 d(t) curve of 50 points; 2080 pairs")

    # ---- Figure E: readout-decision test -----------------------------------------------------
    diff = [p["t_star"] - p["t_flip"] for p in fin
            if p["t_star"] is not None and p["t_flip"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))
    axes[0].hist(diff, bins=41, color=CVD[0], edgecolor="0.25", hatch=HATCHES[0])
    axes[0].axvline(0, **REF_RULE)
    axes[0].set_xlabel("$t^{*} - t_{\\text{flip}}$")
    axes[0].set_ylabel("number of pairs")
    axes[0].set_title(f"d(t) midpoint vs first prediction flip\nmedian $|t^*-t_{{flip}}|$ = "
                      f"{A['readout']['median_abs_diff']:.3f} ({A['readout']['median_abs_diff']*49:.1f} grid steps)",
                      fontsize=9)
    h = A["readout"]["n_argmax_hist"]
    ks = sorted(int(k) for k in h)
    axes[1].bar(ks, [h[str(k)] for k in ks], color=CVD[1], edgecolor="0.25", hatch=HATCHES[1])
    axes[1].set_xlabel("distinct next-character predictions visited along the path")
    axes[1].set_ylabel("number of pairs")
    axes[1].set_title(f"median = {A['readout']['median_n_argmax']:.0f} regions; "
                      f"{A['readout']['frac_exactly_2_argmax']*100:.0f}% visit exactly 2", fontsize=9)
    rw = A["readout_window"]
    bars = [rw["mean_frac_flips_in_window"], rw["frac_pairs_all_flips_in_window"],
            rw["frac_pairs_flat_arms_single_decision"]]
    lbl = ["mean fraction\nof flips inside\n$[t_{lo},t_{hi}]$", "pairs with\nALL flips\ninside",
           "pairs with\nsingle-prediction\nflat arms"]
    axes[2].bar(range(3), bars, color=[CVD[0], CVD[1], CVD[3]], edgecolor="0.25",
                hatch=[HATCHES[0], HATCHES[1], HATCHES[2]])
    for k, v in enumerate(bars):
        axes[2].text(k, v + .015, f"{v:.2f}", ha="center", fontsize=9)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(lbl, fontsize=7.5)
    axes[2].set_ylim(0, 1.08); axes[2].set_ylabel("fraction")
    axes[2].set_title("Prediction changes are confined to the transition", fontsize=9)
    save(fig, "allpairs_readout_decision.png",
         "all three panels pool the same 2080 character pairs, 1 d(t) curve of 50 points per pair")

    # ---- Figure F: controls ------------------------------------------------------------------
    wf = np.array([p["w"] for p in fin if p["w"] is not None])
    wi = np.array([p["w"] for p in S["init_block0"] if p["w"] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    bins = np.linspace(0, 1, 51)
    axes[0].hist(wi, bins=bins, color=CVD[1], alpha=.6, hatch=HATCHES[1], edgecolor="0.25",
                 label=f"step 0 (init), median {np.median(wi):.3f}")
    axes[0].hist(wf, bins=bins, color=CVD[0], alpha=.6, hatch=HATCHES[0], edgecolor="0.25",
                 label=f"step 30000 (final), median {np.median(wf):.3f}")
    axes[0].axvline(LINEAR_W, **REF_DIAG); axes[0].axvline(STRICT_W, **REF_RULE)
    axes[0].set_xlabel("$w_{10\\to90}$"); axes[0].set_ylabel("number of pairs")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("Learned-vs-init: sharpness is trained in, not architectural\n"
                      "(dashed = straight line 0.80, dotted = strict rule 0.25)", fontsize=9)
    blocks = sorted(A["depth"], key=int)
    med = [A["depth"][b]["median_w"] for b in blocks]
    q1 = [A["depth"][b]["iqr"][0] for b in blocks]
    q3 = [A["depth"][b]["iqr"][1] for b in blocks]
    x = [int(b) for b in blocks]
    axes[1].errorbar(x, med, yerr=[np.array(med) - q1, np.array(q3) - np.array(med)],
                     color=CVD[0], marker="o", ls="-", capsize=4, label="median $w$ (bars = IQR)")
    axes[1].axhline(LINEAR_W, **REF_DIAG)
    axes[1].text(11, LINEAR_W + .03, "straight-line reference $w=0.80$", fontsize=8,
                 color="0.35", ha="right")
    axes[1].set_xticks(x); axes[1].set_xlabel("interpolation block $L$ (patch site)")
    axes[1].set_ylabel("$w_{10\\to90}$"); axes[1].set_ylim(0, .95)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("Depth: all sharpness is generated by blocks 1-4\n"
                      "(200-pair subsample, final checkpoint)", fontsize=9)
    save(fig, "allpairs_controls.png",
         "left: 2080 pairs per histogram; right: the same 200-pair subsample at each interpolation block")


if __name__ == "__main__":
    main()
