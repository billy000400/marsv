"""S4 (fresh confirmatory plan): does linearizing the differential neurons remove the sharp switch?

Gated on S3 being "supported". For every pair in the locked S2 manifest (both members of all
101 contrasts) we re-run the identical block-0 interpolation twice more:

  * differential condition -- at every alpha, the post-GELU activations of the neurons in the
    SYMMETRIC DIFFERENCE of the two endpoints' top-64-per-block feature sets (blocks 1-35) are
    overwritten with the linear endpoint interpolation (1-a) a_j(A) + a a_j(B);
  * control condition -- the same thing on an equal-size, per-block set matched on mean
    contribution magnitude, endpoint activation difference and output-weight norm.

Both conditions reproduce the two endpoints exactly (at alpha=0 the forced values equal a(A)),
so w_TV still describes that condition's own switch. The causal prediction is
w_TV(differential) > w_TV(control).

Writes results/causal_metrics.json and plots/causal_linearization.png.
"""
import json
import os

import numpy as np
import torch

from common import PLOTS, RESULTS, blocks, load
from run_interp import clean_run, rel_dist
from s1_sanity import w_tv

N_ALPHA = 101
CHUNK = 16
TOPK = 64
SEED = 31
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def endpoint_acts(m, ids, n_blocks):
    """Final-token post-GELU activations in blocks 1..n_blocks-1, plus block-0 resid and logits."""
    acts = {}
    hs = []
    bl = blocks(m)

    def mk(i):
        def hook(mod, inp, out):
            acts[i] = out[0, -1, :].detach().float().clone()
        return hook
    for i in range(1, n_blocks):
        hs.append(bl[i].mlp.act.register_forward_hook(mk(i)))
    rec, lg = clean_run(m, ids)
    for h in hs:
        h.remove()
    return rec[0], lg, acts


def pick_sets(aA, aB, wnorm, n_blocks, dev):
    """Per block: the symmetric-difference set and a size-matched, feature-matched control.

    Matching runs on CPU in numpy: for each differential neuron we take the nearest unused
    neuron outside the union, in the standardized 3-D space of (mean contribution magnitude,
    endpoint activation difference, output-weight norm).
    """
    diff, ctrl = {}, {}
    for i in range(1, n_blocks):
        a, b, w = (aA[i].cpu().numpy(), aB[i].cpu().numpy(), wnorm[i].cpu().numpy())
        sA = np.argpartition(-np.abs(a) * w, TOPK)[:TOPK]
        sB = np.argpartition(-np.abs(b) * w, TOPK)[:TOPK]
        union = np.zeros(len(w), bool)
        union[sA] = True
        union[sB] = True
        inter = np.zeros(len(w), bool)
        inter[sA] = True
        inter &= np.isin(np.arange(len(w)), sB)
        d = np.nonzero(union & ~inter)[0]

        f = np.stack([0.5 * (np.abs(a) + np.abs(b)) * w, np.abs(a - b), w], axis=1)
        f = (f - f.mean(0)) / (f.std(0) + 1e-9)
        pool = np.nonzero(~union)[0]
        dist = ((f[d][:, None, :] - f[pool][None, :, :]) ** 2).sum(-1)
        taken = np.zeros(len(pool), bool)
        chosen = []
        for r in range(len(d)):
            row = np.where(taken, np.inf, dist[r])
            k = int(row.argmin())
            taken[k] = True
            chosen.append(int(pool[k]))
        diff[i] = torch.tensor(d, device=dev, dtype=torch.long)
        ctrl[i] = torch.tensor(chosen, device=dev, dtype=torch.long)
    return diff, ctrl


def sweep_forced(m, ids, h_vecs, alphas, sets, aA, aB, n_blocks):
    """Block-0 patch as usual, plus forced linear activations on `sets` in blocks 1..n_blocks-1."""
    dev = next(m.parameters()).device
    bl = blocks(m)
    state = {}
    hs = []

    def patch0(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] = state["v"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hs.append(bl[0].register_forward_hook(patch0))

    def mk(i):
        idx = sets[i]

        def hook(mod, inp, out):
            a = state["a"].to(out.dtype).unsqueeze(1)                # (B,1)
            out[:, -1, idx] = ((1 - a) * aA[i][idx].unsqueeze(0)
                               + a * aB[i][idx].unsqueeze(0)).to(out.dtype)
        return hook
    for i in range(1, n_blocks):
        hs.append(bl[i].mlp.act.register_forward_hook(mk(i)))

    rows = []
    for s in range(0, h_vecs.shape[0], CHUNK):
        state["v"] = h_vecs[s:s + CHUNK]
        state["a"] = torch.tensor(alphas[s:s + CHUNK], device=dev, dtype=torch.float32)
        inp = torch.tensor([ids], device=dev).repeat(state["v"].shape[0], 1)
        with torch.no_grad():
            rows.append(m(inp, use_cache=False).logits[:, -1, :].float().cpu())
    for h in hs:
        h.remove()
    return torch.cat(rows)


def main():
    from run_interp import slerp_lerp_norm
    alphas = np.linspace(0, 1, N_ALPHA)
    lock = json.load(open(os.path.join(RESULTS, "matched_pairs.json")))
    s3 = json.load(open(os.path.join(RESULTS, "matched_metrics.json")))
    assert s3["verdict"] == "supported", "S4 runs only if S3 is supported"
    w_un = {(r["prefix"], lab): r[f"w_tv_{lab}"] for r in s3["contrasts"] for lab in ("high", "low")}

    rng = np.random.default_rng(SEED)
    tok, m = load("gpt2-large")
    dev = next(m.parameters()).device
    n_blocks = len(blocks(m))
    wnorm = {i: blocks(m)[i].mlp.c_proj.weight.detach().float().norm(dim=1).to(dev)
             for i in range(1, n_blocks)}

    rows, worst = [], 0.0
    for k, c in enumerate(lock["contrasts"]):
        for lab in ("high", "low"):
            e = c[lab]
            ida, idb = c["ids"] + [e["tok_a"]], c["ids"] + [e["tok_b"]]
            hA, lgA, aA = endpoint_acts(m, ida, n_blocks)
            hB, lgB, aB = endpoint_acts(m, idb, n_blocks)
            vecs, _, _ = slerp_lerp_norm(hA, hB, alphas)
            diff, ctrl = pick_sets(aA, aB, wnorm, n_blocks, dev)
            r = dict(prefix=c["prefix"], member=lab, F=e["F"], jsd=e["jsd"],
                     n_forced=int(sum(len(v) for v in diff.values())),
                     w_tv_unablated=w_un[(c["prefix"], lab)])
            for cond, sets in (("diff", diff), ("ctrl", ctrl)):
                lg = sweep_forced(m, ida, vecs, alphas, sets, aA, aB, n_blocks)
                worst = max(worst,
                            float((lg[0] - lgA.cpu()).norm() / lgA.cpu().norm()),
                            float((lg[-1] - lgB.cpu()).norm() / lgB.cpu().norm()))
                d = rel_dist(lg, lgA.cpu().unsqueeze(0), lgB.cpu().unsqueeze(0))
                r[f"w_tv_{cond}"] = w_tv(alphas, d)
            r["delta_diff"] = r["w_tv_diff"] - r["w_tv_unablated"]
            r["delta_ctrl"] = r["w_tv_ctrl"] - r["w_tv_unablated"]
            r["gap"] = r["delta_diff"] - r["delta_ctrl"]
            rows.append(r)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(lock['contrasts'])} contrasts")
    del m
    torch.cuda.empty_cache()

    gap = np.array([r["gap"] for r in rows])
    n = len(gap)
    boot = np.array([np.median(gap[rng.integers(0, n, n)]) for _ in range(10000)])
    signs = rng.choice([-1.0, 1.0], size=(10000, n))
    perm = np.median(gap[None, :] * signs, axis=1)
    med = float(np.median(gap))
    out = dict(
        n_pairs=n, worst_endpoint_rel_err=worst,
        median_w_unablated=float(np.median([r["w_tv_unablated"] for r in rows])),
        median_w_diff=float(np.median([r["w_tv_diff"] for r in rows])),
        median_w_ctrl=float(np.median([r["w_tv_ctrl"] for r in rows])),
        median_gap=med,
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        frac_predicted_sign=float((gap > 0).mean()),
        perm_p=float((np.abs(perm) >= abs(med)).mean()),
        median_n_forced=float(np.median([r["n_forced"] for r in rows])),
        by_member={lab: dict(
            median_gap=float(np.median([r["gap"] for r in rows if r["member"] == lab])),
            median_w_unablated=float(np.median([r["w_tv_unablated"] for r in rows
                                                if r["member"] == lab])),
            median_w_diff=float(np.median([r["w_tv_diff"] for r in rows if r["member"] == lab])),
            median_w_ctrl=float(np.median([r["w_tv_ctrl"] for r in rows if r["member"] == lab])))
            for lab in ("high", "low")},
        pairs=rows,
    )
    out["verdict"] = ("supported" if med > 0 and out["ci95"][0] > 0 else "not supported")
    print(f"n={n} median gap={med:+.4f} CI={out['ci95']} "
          f"frac>0={out['frac_predicted_sign']:.3f} p={out['perm_p']:.4f} -> {out['verdict']}")
    with open(os.path.join(RESULTS, "causal_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot(rows, out)
    return out


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    med = [np.median([r[f"w_tv_{c}"] for r in rows]) for c in ("unablated", "ctrl", "diff")]
    for r in rows:
        ax[0].plot([0, 1, 2], [r["w_tv_unablated"], r["w_tv_ctrl"], r["w_tv_diff"]],
                   "-", color=CVD[3], lw=0.5, alpha=0.35)
    ax[0].plot([0, 1, 2], med, "-o", color="k", lw=2.6, ms=7, label="median")
    ax[0].axhline(0.5, color="0.45", ls=":", lw=1.4, label="linear response")
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["unablated", "control\nlinearized", "differential\nlinearized"])
    ax[0].set_ylabel(r"transition width $w_{TV}$")
    ax[0].set_title("Forcing the differential neurons to interpolate linearly")
    ax[0].legend(fontsize=8)

    gap = np.array([r["gap"] for r in rows])
    ax[1].hist(gap, bins=30, color=CVD[0], edgecolor="k", lw=0.4)
    ax[1].axvline(0, color="0.35", ls=":", lw=1.5)
    ax[1].axvline(out["median_gap"], color=CVD[1], ls="--", lw=2,
                  label=f"median = {out['median_gap']:+.3f}")
    ax[1].axvspan(*out["ci95"], color=CVD[1], alpha=0.18, label="95% CI")
    ax[1].set_xlabel(r"$\Delta w_{TV}(\mathrm{differential}) - \Delta w_{TV}(\mathrm{control})$")
    ax[1].set_ylabel("pairs")
    ax[1].set_title(f"n={out['n_pairs']}, {100 * out['frac_predicted_sign']:.0f}% predicted sign")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "causal_linearization.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
