"""S7 — operator-feedback plateau metric.

Addresses human_feedback_07140930.md point 4: use the normalized plateau
coordinate

    p(t) = |h(t) - hA| / (|h(t) - hA| + |h(t) - hB|)

where h(t) is the induced 8-bin behavior distribution at waypoint t in Hellinger
(sqrt-prob) coordinates, hA = h(start endpoint) and hB = h(end endpoint), both
per base prompt. p=0 at the start behavior, p=1 at the end behavior; a *plateau*
shows as p(t) staying flat near 0 (or 1) before a rapid switch.

Reuses the exact TailRunner + injection convention from s4_sweep on the saved
Tue->Wed sweep paths (no re-optimization). Model tail only; cheap.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import pathlib_opt as P
import s4_sweep as S4

DEVICE = "cuda"
PAIR = "Tuesday-Wednesday"


def plateau_curve(tr, W, bases, V32):
    """Return (p(t), dstart(t), dend(t)) averaged over base prompts."""
    n = W.shape[0]
    p_acc = np.zeros(n); ds_acc = np.zeros(n); de_acc = np.zeros(n)
    for b in bases:
        shift = (W - b["coords"]) @ V32
        injected = b["a_last"][None] + shift
        dist8 = tr.behavior(b["hs"], injected)
        H = torch.sqrt(dist8.clamp_min(0.0) + 1e-12)      # [n, 8]
        HA, HB = H[0], H[-1]
        ds = np.array([S4.hellinger(H[t], HA) for t in range(n)])
        de = np.array([S4.hellinger(H[t], HB) for t in range(n)])
        denom = ds + de
        denom[denom < 1e-9] = 1e-9
        p_acc += ds / denom
        ds_acc += ds; de_acc += de
    m = len(bases)
    return p_acc / m, ds_acc / m, de_acc / m


def main():
    out = np.load(C.RESULTS + f"/sweep_{PAIR}.npy", allow_pickle=True).item()
    V32, mean_t, centroids, M = S4.build_context()

    rows = C.build_prompts()
    idx = out["base_idx"]
    tr = P.TailRunner()
    bases = S4.prepare_bases(tr, idx, rows, V32, mean_t)
    tr.move_tail_to_gpu()

    def Wt(arr):
        return torch.tensor(np.asarray(arr), dtype=torch.float32, device=DEVICE)

    res = out["results"]
    # linear-init family (matches the d(t) figure)
    paths = [
        ("linear chord", out["linear"]["W"], "gray", "s--"),
        ("centroid spline", out["spline_ref"]["W"], "green", "*-."),
        ("output-only", res["linear_outonly"]["W"], "#b2182b", "v:"),
    ]
    lam_paths = [("linear_lam0.1", 0.1), ("linear_lam1.0", 1.0),
                 ("linear_lam10.0", 10.0), ("linear_lam100.0", 100.0)]
    cmap = plt.cm.viridis
    curves = {}
    for name, W, color, style in paths:
        p, ds, de = plateau_curve(tr, Wt(W), bases, V32)
        curves[name] = dict(p=p.tolist(), dstart=ds.tolist(), dend=de.tolist(),
                            color=color, style=style)
    for i, (key, lam) in enumerate(lam_paths):
        p, ds, de = plateau_curve(tr, Wt(res[key]["W"]), bases, V32)
        curves[f"λ={lam:g}"] = dict(p=p.tolist(), dstart=ds.tolist(),
                                    dend=de.tolist(),
                                    color=matplotlib.colors.to_hex(cmap(i / 4)),
                                    style="-")

    n = len(out["linear"]["W"])
    tt = np.linspace(0, 1, n)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name in ["linear chord", "centroid spline", "output-only"]:
        c = curves[name]
        ax.plot(tt, c["p"], c["style"], color=c["color"], label=name,
                linewidth=2.0, markersize=7)
    for key, lam in lam_paths:
        name = f"λ={lam:g}"
        c = curves[name]
        ax.plot(tt, c["p"], "-", color=c["color"], label=name, alpha=0.9)
    ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("path position $t$ (0=Tuesday, 1=Wednesday)")
    ax.set_ylabel(r"plateau coord $p(t)=\frac{|h-h_A|}{|h-h_A|+|h-h_B|}$")
    ax.set_title("Downstream plateau metric — Tuesday→Wednesday\n"
                 "(normalized Hellinger progress; flat near 0 = plateau)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(C.PLOTS + "/s7_plateau_metric.png", dpi=130)
    plt.close(fig)

    import json
    with open(C.RESULTS + "/s7_plateau.json", "w") as f:
        json.dump({k: {kk: v[kk] for kk in ("p", "dstart", "dend")}
                   for k, v in curves.items()}, f, indent=2)
    # quick numeric summary: plateau "flatness" = fraction of first half with p<0.25
    for name in ["linear chord", "centroid spline", "output-only", "λ=100"]:
        p = np.array(curves[name]["p"])
        half = n // 2
        print(f"{name:16s} p(mid)={p[half]:.3f}  frac(first-half p<0.25)="
              f"{np.mean(p[:half] < 0.25):.2f}")
    print("saved s7_plateau_metric.png + s7_plateau.json")


if __name__ == "__main__":
    main()
