"""S3 - two-dimensional sweeps in the feature-feature plane and in a random-random plane.

Axes are rescaled per anchor so that each single-direction boundary (the perturbation size at which
JSD first reaches THETA) sits at +-1; the shape of the low-change region between the axes is then
directly comparable across anchors and between the two planes.
Writes results/s3.json and plots/fig3_planes.png.
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
from s2_compare import anchors, orthonormal_pair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
N_ANCHOR, SEED = 12, 2
THETA = 0.05           # JSD level (bits) that defines the single-direction boundary
SCAN = np.linspace(0.0, 6.0, 61)
GRID = np.linspace(-1.5, 1.5, 41)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def crossing(model, base, h0, v, sign, device):
    """Smallest perturbation size along `sign * v` whose JSD reaches THETA (linear interpolation)."""
    H = torch.cat([perturb(h0, v, float(sign * a)) for a in SCAN])
    P = sweep_from_base(model, base, H, device)
    p0 = P[0]
    d = jsd(P, p0[None, :])
    k = np.argmax(d >= THETA)
    if d[k] < THETA:
        return np.nan
    if k == 0:
        return float(SCAN[0])
    lo, hi = d[k - 1], d[k]
    return float(SCAN[k - 1] + (THETA - lo) / max(hi - lo, 1e-12) * (SCAN[k] - SCAN[k - 1]))


def plane(model, base, h0, va, vb, sa, sb, device):
    """JSD on the rescaled GRID x GRID plane spanned by va, vb."""
    X, Y = np.meshgrid(GRID, GRID, indexing="xy")
    ax = X * np.where(X >= 0, sa[0], sa[1])
    by = Y * np.where(Y >= 0, sb[0], sb[1])
    coef = torch.tensor(np.stack([ax.ravel(), by.ravel()], 1), dtype=torch.float32)
    nrm = h0.norm()
    u = h0 / nrm
    W = u + coef[:, :1] * va[None, :] + coef[:, 1:2] * vb[None, :]
    H = nrm * W / W.norm(dim=1, keepdim=True)
    P = sweep_from_base(model, base, H, device)
    p0 = sweep_from_base(model, base, h0, device)[0]
    return jsd(P, p0[None, :]).reshape(X.shape)


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
    g = torch.Generator().manual_seed(SEED)

    maps = {"feature": [], "random": []}
    scales = {"feature": [], "random": []}
    used = []
    for a in range(N_ANCHOR):
        base, _ = base_resid(model, ids[a:a + 1], device)
        h0 = base[-1].float().cpu()[None, :]
        r1 = torch.randn(v1.shape[0], generator=g)
        r1 = r1 / r1.norm()
        r2 = torch.randn(v1.shape[0], generator=g)
        r2 = r2 - torch.dot(r2, r1) * r1
        r2 = r2 / r2.norm()
        ok = True
        sc = {}
        for tag, (va, vb) in [("feature", (v1, v2)), ("random", (r1, r2))]:
            s = [[crossing(model, base, h0, va, +1, device), crossing(model, base, h0, va, -1, device)],
                 [crossing(model, base, h0, vb, +1, device), crossing(model, base, h0, vb, -1, device)]]
            if np.any(np.isnan(s)):
                ok = False
            sc[tag] = s
        if not ok:
            print(f"anchor {a}: no THETA crossing within scan, skipped", flush=True)
            continue
        for tag, (va, vb) in [("feature", (v1, v2)), ("random", (r1, r2))]:
            maps[tag].append(plane(model, base, h0, va, vb, sc[tag][0], sc[tag][1], device))
            scales[tag].append(sc[tag])
        used.append(idxs[a])
        print(f"anchor {a} done ({len(used)} kept)", flush=True)

    M = {k: np.mean(np.stack(v), 0) for k, v in maps.items()}
    json.dump({"ckpt_step": int(step), "block": BLOCK, "theta": THETA, "grid": GRID.tolist(),
               "anchor_indices": used, "scales": {k: np.array(v).tolist() for k, v in scales.items()},
               "mean_map": {k: v.tolist() for k, v in M.items()}},
              open(os.path.join(RES, "s3.json"), "w"), indent=2)

    vmax = float(max(M["feature"].max(), M["random"].max()))
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for j, (tag, title) in enumerate([("feature", "feature-feature plane"),
                                      ("random", "random-random plane")]):
        im = ax[j].pcolormesh(GRID, GRID, M[tag], cmap="viridis", vmin=0, vmax=vmax, shading="gouraud")
        cs = ax[j].contour(GRID, GRID, M[tag], levels=[THETA], colors="white", linewidths=2.0)
        ax[j].clabel(cs, fmt={THETA: f"JSD={THETA}"}, fontsize=7)
        ax[j].plot([-1, 1, 0, 0], [0, 0, -1, 1], ls="none", marker="o", ms=5,
                   mfc="white", mec="black", mew=1.2)
        ax[j].set_title(title)
        ax[j].set_xlabel("rescaled coefficient, axis 1")
        ax[j].set_aspect("equal")
    ax[0].set_ylabel("rescaled coefficient, axis 2")
    fig.colorbar(im, ax=ax, label="mean JSD from clean (bits)", fraction=0.035)
    fig.savefig(os.path.join(PLOTS, "fig3_planes.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/fig3_planes.png", flush=True)


if __name__ == "__main__":
    main()
