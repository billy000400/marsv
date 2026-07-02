"""Direction #9 — epsilon scan for plateau-perturbation (operator feedback 2026-07-01 04:38).

The plateau-perturbation metric injects a random unit direction of magnitude `eps` at the
measurement point and scores the mean next-token KL(clean||perturbed). Iter-2..7 used a SINGLE
eps=6.0. The operator asked whether a *different* eps — or a scan over eps — changes the picture
(does plateau-perturbation ever become competitive at some magnitude?).

This scans a grid of eps at every measurement point {input,resid3,resid6,resid9} on the SAME
canonical split as plateau_v2.py (randperm(seed=7), fit=perm[:NFIT], test=perm[NFIT:NFIT+N]) so
the AUROCs are directly comparable to auroc_table.csv. For each eps we reuse the SAME random unit
directions across all magnitudes (d fixed, only the scalar eps varies) so curves are apples-to-apples.

Writes results/auroc_perturbation_eps.csv [ood_set, measurement_point, eps, auroc] and a plot.
Higher score == more OOD; AUROC via plateau_score.auroc (pure-numpy Mann-Whitney).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import plateau_score as ps
import plateau_v2 as pv2

torch.set_num_threads(2)
torch.manual_seed(0)
DEVICE = pv2.DEVICE
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.225)   # this run's shared-GPU budget
POINTS = ["input", "resid3", "resid6", "resid9"]
EPS_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]


@torch.no_grad()
def perturbation_multi_eps(ids, point, seq_len, eps_grid, n_dirs=16, bs=8):
    """Mean next-token KL(clean||perturbed) at `point`,last token, for EVERY eps in eps_grid,
    reusing the same random unit directions across magnitudes. Returns {eps: score[N]}."""
    model, _ = pv2.load_model()
    pos = seq_len - 1
    hdim = model.config.n_embd
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    inj = {"delta": None}

    def hook(mod, i, o):
        if inj["delta"] is None:
            return o
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[:, pos] = h[:, pos] + inj["delta"]
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = pv2._module(model, point).register_forward_hook(hook)
    per_eps = {e: [] for e in eps_grid}
    try:
        for b in pv2._batches(ids, bs):
            inj["delta"] = None
            clean = torch.log_softmax(model(b).logits[:, pos], -1)
            p_clean = clean.exp()
            B = b.shape[0]
            kl = {e: torch.zeros(B, device=DEVICE) for e in eps_grid}
            for _ in range(n_dirs):
                d = torch.randn(B, hdim, generator=g, device=DEVICE)
                d = d / d.norm(dim=-1, keepdim=True)
                for e in eps_grid:
                    inj["delta"] = e * d
                    pert = torch.log_softmax(model(b).logits[:, pos], -1)
                    kl[e] += (p_clean * (clean - pert)).sum(-1)
                inj["delta"] = None
            for e in eps_grid:
                per_eps[e].append((kl[e] / n_dirs).cpu())
    finally:
        handle.remove()
    return {e: torch.cat(v).numpy() for e, v in per_eps.items()}


if __name__ == "__main__":
    t0 = time.time()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    NDIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    NFIT = 1000
    HERE = os.path.dirname(__file__)
    pv2.load_model()
    print(f"DEVICE={DEVICE} N={N} SEQ={SEQ} NDIRS={NDIRS} eps={EPS_GRID}", flush=True)

    texts = [l.rstrip("\n") for l in open(os.path.join(HERE, "..", "data", "fineweb_sample.txt"))]
    all_ids = ps.encode_fixed(texts, seq_len=SEQ)
    g_split = torch.Generator().manual_seed(7)
    perm = torch.randperm(all_ids.shape[0], generator=g_split)
    test_idx = perm[NFIT:NFIT + N]
    id_test = all_ids[test_idx]
    OOD = {"random": ps.make_ood(id_test, "random", seed=0),
           "shuffled": ps.make_ood(id_test, "shuffled", seed=0),
           "code": ps.make_ood(id_test, "code", seed=0)}
    print(f"id_test {tuple(id_test.shape)} | ood " +
          " ".join(f"{k}{tuple(v.shape)}" for k, v in OOD.items()), flush=True)

    # id-test scores once per point (shared across OOD sets)
    id_scores = {p: perturbation_multi_eps(id_test, p, SEQ, EPS_GRID, n_dirs=NDIRS) for p in POINTS}
    print(f"id scored {time.time()-t0:.0f}s", flush=True)
    rows = []
    for oname, oids in OOD.items():
        for p in POINTS:
            osc = perturbation_multi_eps(oids, p, SEQ, EPS_GRID, n_dirs=NDIRS)
            for e in EPS_GRID:
                sid, sood = id_scores[p][e], osc[e]
                a = ps.auroc(np.concatenate([sid, sood]),
                             np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
                rows.append([oname, p, e, round(float(a), 4)])
        print(f"  {oname} scored {time.time()-t0:.0f}s", flush=True)

    import csv
    out = os.path.join(HERE, "..", "results", "auroc_perturbation_eps.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ood_set", "measurement_point", "eps", "auroc"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows) {time.time()-t0:.0f}s", flush=True)

    # ---- plot: AUROC vs eps, one panel per OOD set, one line per measurement point
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oods = list(OOD.keys())
    fig, axes = plt.subplots(1, len(oods), figsize=(5 * len(oods), 4.2), sharey=True)
    for ax, oname in zip(axes, oods):
        for p in POINTS:
            ys = [next(r[3] for r in rows if r[0] == oname and r[1] == p and r[2] == e) for e in EPS_GRID]
            ax.plot(EPS_GRID, ys, marker="o", label=p)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(f"OOD = {oname}")
        ax.set_xlabel("perturbation magnitude  $\\epsilon$")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC (higher=OOD detected)")
    axes[-1].legend(title="measurement point", fontsize=8)
    fig.suptitle("plateau-perturbation: AUROC vs perturbation magnitude $\\epsilon$ (canonical split, N=%d)" % N)
    fig.tight_layout()
    pp = os.path.join(HERE, "..", "results", "plots", "perturbation_eps_scan.png")
    fig.savefig(pp, dpi=110); plt.close(fig)
    print(f"wrote {pp}", flush=True)

    # print best eps per (ood,point) and overall best plateau-perturbation AUROC per OOD set
    for oname in oods:
        best = max((r for r in rows if r[0] == oname), key=lambda r: r[3])
        print(f"BEST {oname}: {best[1]} eps={best[2]} auroc={best[3]}", flush=True)
