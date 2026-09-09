"""S2 - sweep the same perturbation size along feature 1, feature 2, their equal mixture and
random directions, at 30 held-out ordinary-text anchors. Writes results/s2.json, plots/fig2_curves.png.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_assay import (load_model, contexts, base_resid, sweep_from_base, perturb, jsd,
                           feature_positions, BLOCK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
N_ANCHOR, N_RAND, SEED = 30, 10, 1
ALPHAS = np.linspace(0.0, 3.0, 31)
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def anchors(text, cand, rng, n):
    """Ordinary running-text positions: on a non-label line, ending in a lower-case letter."""
    cands = [i for i in cand["wordcont_pos"] + cand["wordcont_neg"] if text[i].islower()]
    return sorted(rng.choice(np.array(sorted(set(cands))), size=n, replace=False).tolist())


def orthonormal_pair(v1, v2):
    v1 = v1 / v1.norm()
    w = v2 - torch.dot(v2, v1) * v1
    return v1, w / w.norm()


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    model, stoi, step = load_model(device)
    ho_text, ho = feature_positions("heldout")
    idxs = anchors(ho_text, ho, rng, N_ANCHOR)
    ids = contexts(ho_text, idxs, stoi)

    D = torch.load(os.path.join(RES, "directions.pt"), weights_only=False)
    v1, v2 = orthonormal_pair(D["speaker_label"].float(), D["word_continuation"].float())
    mix = (v1 + v2) / (v1 + v2).norm()
    g = torch.Generator().manual_seed(SEED)
    rand = []
    for _ in range(N_RAND):
        u = torch.randn(v1.shape[0], generator=g)
        rand.append(u / u.norm())

    conds = {"speaker_label": v1, "word_continuation": v2, "equal_mixture": mix}
    J = {k: np.zeros((N_ANCHOR, len(ALPHAS))) for k in conds}
    Disp = {k: np.zeros((N_ANCHOR, len(ALPHAS))) for k in conds}
    JR = np.zeros((N_ANCHOR, N_RAND, len(ALPHAS)))
    for a in range(N_ANCHOR):
        base, p0 = base_resid(model, ids[a:a + 1], device)
        h0 = base[-1].float().cpu()[None, :]
        for k, v in conds.items():
            H = torch.cat([perturb(h0, v, float(al)) for al in ALPHAS])
            Disp[k][a] = ((H - h0).norm(dim=1) / h0.norm()).numpy()
            J[k][a] = jsd(sweep_from_base(model, base, H, device), p0[None, :])
        for r, u in enumerate(rand):
            H = torch.cat([perturb(h0, u, float(al)) for al in ALPHAS])
            JR[a, r] = jsd(sweep_from_base(model, base, H, device), p0[None, :])
        if a % 10 == 0:
            print(f"anchor {a}", flush=True)

    Jr_mean = JR.mean(1)
    out = {"ckpt_step": int(step), "block": BLOCK, "alphas": ALPHAS.tolist(),
           "anchor_indices": idxs, "n_random": N_RAND,
           "mean_jsd": {k: J[k].mean(0).tolist() for k in conds},
           "mean_jsd_random": Jr_mean.mean(0).tolist(),
           "per_anchor_jsd": {k: J[k].tolist() for k in conds},
           "per_anchor_jsd_random": Jr_mean.tolist(),
           "mean_relative_displacement": {k: Disp[k].mean(0).tolist() for k in conds}}
    # fraction of anchors whose single-feature curve is above the mixture curve, per alpha
    out["frac_single_above_mixture"] = {
        k: ((J[k] > J["equal_mixture"]).mean(0)).tolist() for k in ["speaker_label", "word_continuation"]}
    json.dump(out, open(os.path.join(RES, "s2.json"), "w"), indent=2)

    style = {"speaker_label": ("-", "o", CVD[0]), "word_continuation": ("--", "s", CVD[1]),
             "equal_mixture": ("-.", "^", CVD[2]), "random": (":", "D", "0.45")}
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
    for k in conds:
        ls, mk, col = style[k]
        ax[0].plot(ALPHAS, J[k].mean(0), ls=ls, marker=mk, ms=3, color=col,
                   lw=2.2, label=k.replace("_", " "))
    ax[0].plot(ALPHAS, Jr_mean.mean(0), ls=":", marker="D", ms=3, color="0.45", lw=2.0,
               label="random (mean of 10)")
    ax[0].set_title("mean over 30 held-out anchors")
    ax[0].legend(fontsize=7)
    for j, a in enumerate([0, 1, 2]):
        for k in conds:
            ls, mk, col = style[k]
            ax[j + 1].plot(ALPHAS, J[k][a], ls=ls, marker=mk, ms=3, color=col, lw=1.8)
        ax[j + 1].plot(ALPHAS, Jr_mean[a], ls=":", marker="D", ms=3, color="0.45", lw=1.6)
        ax[j + 1].set_title(f"anchor {a + 1}: ...{ho_text[idxs[a] - 12:idxs[a] + 1]!r}", fontsize=7)
    for a_ in ax:
        a_.set_xlabel(r"perturbation size $\alpha$")
        a_.grid(alpha=0.3)
    ax[0].set_ylabel("JSD from clean next-char distribution (bits)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "fig2_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/fig2_curves.png", flush=True)


if __name__ == "__main__":
    main()
