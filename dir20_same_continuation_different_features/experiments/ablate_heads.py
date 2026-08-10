"""S10: make the circuit-difference result causal by ablating the differentially-engaged heads.

Experiment 6 is correlational: pairs whose attention heads write more differently have sharper
switches. This script intervenes. For each low-JSD pair we mean-ablate, at the final token only,
either

  DIFF - the k heads with the largest differential engagement
         delta_h = (|c_h^A| + |c_h^B|) * (1 - cos(c_h^A, c_h^B)), i.e. the heads that write most
         *differently* for the two prompts, or
  CTRL - k heads matched one-for-one on total engagement m_h = |c_h^A| + |c_h^B| but drawn from the
         low-delta half, so the same amount of head output is removed without removing the
         difference.

Everything (both endpoints and the 101-point sweep) is re-run with the ablation active, so d(0)=0
and d(1)=1 hold within each condition and w_TV measures the switch shape of the ablated model.

Prediction of the causal reading: w_TV(DIFF) > w_TV(CTRL) - removing the differential machinery
flattens the switch more than removing an equal amount of ordinary head output.

Writes results/ablate_heads.json and plots/ablation_causal.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from analyze import tv_width
from analyze_bank import cluster_boot
from circuit_features import probe
from common import PLOTS, RESULTS, load
from mine_lowjsd import N_PREFIX, SEED, get_prefixes
from run_interp import clean_run, rel_dist, slerp_lerp_norm, sweep

MODELS = ["gpt2-medium", "gpt2-large"]
DOSES = [0.03, 0.06, 0.10]   # heads ablated, as a fraction of all heads in the model
NEAR = 24              # engagement-matching pool: this many nearest heads by |c^A| + |c^B|
N_MEAN = 100           # prompts used to build the mean head output that replaces an ablated head
N_ALPHA = 101
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
COND = ["base", "ctrl", "diff"]
COND_LABEL = {"base": "no ablation", "ctrl": "matched control heads", "diff": "differential heads"}
COND_STYLE = {"base": dict(color="0.55", marker="o", ls="-"),
              "ctrl": dict(color=CVD[0], marker="s", ls="--"),
              "diff": dict(color=CVD[1], marker="^", ls=":")}


def head_means(m, prompts):
    """Mean attention-head output (the c_proj input) at the final token, per block."""
    bl = m.transformer.h
    acc, hs = {}, []

    def mk(i):
        def h(mod, inp):
            acc[i] = acc.get(i, 0) + inp[0][0, -1, :].detach().float()
        return h

    for i, b in enumerate(bl):
        hs.append(b.attn.c_proj.register_forward_pre_hook(mk(i)))
    with torch.no_grad():
        for ids in prompts:
            m(torch.tensor([ids], device=next(m.parameters()).device), use_cache=False)
    for h in hs:
        h.remove()
    return torch.stack([acc[i] / len(prompts) for i in sorted(acc)])   # [n_block, d_model]


class Ablation:
    """Mean-ablate a set of (block, head) pairs at the final token position."""

    def __init__(self, m, means, dh):
        self.bl, self.means, self.dh, self.hs = m.transformer.h, means, dh, []

    def __enter__(self):
        return self

    def set(self, heads):
        self.remove()
        per_block = {}
        for b, h in heads:
            per_block.setdefault(b, []).append(h)
        for b, hh in per_block.items():
            idx = torch.cat([torch.arange(h * self.dh, (h + 1) * self.dh) for h in sorted(hh)])
            self.hs.append(self.bl[b].attn.c_proj.register_forward_pre_hook(self._mk(b, idx)))

    def _mk(self, b, idx):
        mean = self.means[b][idx]

        def hook(mod, inp):
            x = inp[0].clone()
            x[:, -1, idx] = mean.to(x.dtype)
            return (x,) + tuple(inp[1:])
        return hook

    def remove(self):
        for h in self.hs:
            h.remove()
        self.hs = []

    def __exit__(self, *a):
        self.remove()


def select_heads(ca, cb, k):
    """(differential heads, engagement-matched control heads) as lists of (block, head)."""
    nb, nh, _ = ca.shape
    a = ca.reshape(-1, ca.shape[-1])
    b = cb.reshape(-1, cb.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    mag = a.norm(dim=-1) + b.norm(dim=-1)
    delta = mag * (1.0 - cos)

    diff = torch.topk(delta, k).indices.tolist()
    # Matched controls: for each differential head, take the NEAR heads whose total engagement is
    # closest to it and keep the one that writes most similarly for the two prompts. Magnitude is
    # matched tightly, delta is made as small as that matching allows.
    mag_np, delta_np = mag.cpu().numpy(), delta.cpu().numpy()
    used = set(diff)
    ctrl = []
    for i in diff:
        cand = [j for j in np.argsort(np.abs(mag_np - mag_np[i])) if j not in used][:NEAR]
        pick = int(min(cand, key=lambda j: delta_np[j]))
        used.add(pick)
        ctrl.append(pick)
    unflat = lambda idx: [(int(i) // nh, int(i) % nh) for i in idx]
    return unflat(diff), unflat(ctrl), delta, mag


def hcd_after(ca, cb, means, dh, heads):
    """HCD recomputed with the ablated heads' contributions replaced by the shared mean vector.

    An ablated head writes the same vector for both prompts, so it contributes 0 to the numerator
    while still occupying weight in the denominator. Manipulation check for the intervention.
    """
    nb, nh, _ = ca.shape
    a = ca.reshape(-1, ca.shape[-1])
    b = cb.reshape(-1, cb.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    w = a.norm(dim=-1) + b.norm(dim=-1)
    keep = torch.ones(a.shape[0], dtype=torch.bool)
    denom_abl = 0.0
    for blk, h in heads:
        keep[blk * nh + h] = False
        Wc = means[blk][h * dh:(h + 1) * dh]
        denom_abl += 2.0 * float(Wc.norm())
    num = float((w[keep] * (1.0 - cos[keep])).sum())
    den = float(w[keep].sum()) + denom_abl
    return num / den


def run_condition(m, ids_a, ids_b, alphas):
    """Endpoints + 101-point block-0 sweep under whatever hooks are currently installed."""
    reca, lga = clean_run(m, ids_a)
    recb, lgb = clean_run(m, ids_b)
    vecs, _, _ = slerp_lerp_norm(reca[0], recb[0], alphas)
    _, lgs = sweep(m, ids_a, vecs, layer=0)
    d = rel_dist(lgs, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
    return d


def one_dose(m, rows, prefixes, means, dh, nh, k, alphas, rng_boot):
    """Run both ablation conditions at one dose over the whole bank; return (info, recs)."""
    recs = []
    with Ablation(m, means, dh) as abl:
        for n, r in enumerate(rows):
            pre = prefixes[r["prefix_idx"]]
            ida, idb = pre + [r["id_a"]], pre + [r["id_b"]]
            pa, pb = probe(m, ida), probe(m, idb)
            diff, ctrl, delta, mag = select_heads(pa["contrib"], pb["contrib"], k)

            rec = dict(key=r["key"], prefix_idx=r["prefix_idx"], jsd=r["jsd"],
                       wtv_base=r["wtv"],
                       hcd_base=hcd_after(pa["contrib"], pb["contrib"], means, dh, []),
                       mag_diff=float(sum(mag[b * nh + h] for b, h in diff)),
                       mag_ctrl=float(sum(mag[b * nh + h] for b, h in ctrl)))
            for cond, heads in (("ctrl", ctrl), ("diff", diff)):
                abl.set(heads)
                d = run_condition(m, ida, idb, alphas)
                rec[f"wtv_{cond}"] = tv_width(alphas, d)
                rec[f"hcd_{cond}"] = hcd_after(pa["contrib"], pb["contrib"], means, dh, heads)
                rec[f"err_{cond}"] = float(max(abs(d[0]), abs(d[-1] - 1)))
            abl.remove()
            recs.append(rec)
            if n % 100 == 0:
                print(f"    pair {n}/{len(rows)}", flush=True)

    cl = {}
    for r in recs:
        cl.setdefault(r["prefix_idx"], []).append(r)
    wc = np.array([r["wtv_ctrl"] for r in recs])
    wd = np.array([r["wtv_diff"] for r in recs])
    lo, hi = cluster_boot(
        cl, lambda rr: float(np.median([q["wtv_diff"] - q["wtv_ctrl"] for q in rr])), rng_boot)
    info = dict(
        n=len(recs), n_prefix=len(cl), k_heads=k,
        median_wtv={c: float(np.median([r[f"wtv_{c}"] for r in recs])) for c in COND},
        median_hcd={c: float(np.median([r[f"hcd_{c}"] for r in recs])) for c in COND},
        ci_wtv={c: list(cluster_boot(cl, lambda rr, c=c: float(np.median([q[f"wtv_{c}"]
                                                                         for q in rr])), rng_boot))
                for c in COND},
        mag_removed=dict(diff=float(np.median([r["mag_diff"] for r in recs])),
                         ctrl=float(np.median([r["mag_ctrl"] for r in recs]))),
        median_delta=float(np.median(wd - wc)), ci_delta=[lo, hi],
        wilcoxon_p=float(wilcoxon(wd, wc)[1]), frac_diff_gt_ctrl=float(np.mean(wd > wc)),
        max_endpoint_err=float(max(max(r["err_ctrl"], r["err_diff"]) for r in recs)))
    return info, recs


def main(model_keys=MODELS):
    alphas = np.linspace(0, 1, N_ALPHA)
    out = {}
    rng_boot = np.random.default_rng(0)

    for mkey in model_keys:
        rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
        rows = rows[:int(os.environ.get("LIMIT", len(rows)))]
        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
        nh = m.config.n_head
        dh = m.config.n_embd // nh
        n_heads = nh * len(m.transformer.h)

        seen, mean_prompts = set(), []
        for r in rows:
            if r["prefix_idx"] not in seen and len(mean_prompts) < N_MEAN:
                seen.add(r["prefix_idx"])
                mean_prompts.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
        means = head_means(m, mean_prompts)

        out[mkey] = dict(n_heads=n_heads, doses={})
        for frac in DOSES:
            k = max(4, int(round(frac * n_heads)))
            print(f"{mkey}: dose {frac:.0%} -> k={k} of {n_heads} heads, {len(rows)} pairs",
                  flush=True)
            info, recs = one_dose(m, rows, prefixes, means, dh, nh, k, alphas, rng_boot)
            info["frac"] = frac
            info["n_heads"] = n_heads
            out[mkey]["doses"][f"{frac:.2f}"] = dict(info=info, rows=recs)
            print(mkey, json.dumps(info), flush=True)
            with open(os.path.join(RESULTS, "ablate_heads.json"), "w") as f:
                json.dump(out, f, indent=1)   # persist as we go: a long run stays usable if cut
        del m
        torch.cuda.empty_cache()

    figure(out, [k for k in model_keys if len(out[k]["doses"]) == len(DOSES)])
    return out


MSTYLE = {"gpt2-medium": dict(marker="s", ls="--"), "gpt2-large": dict(marker="^", ls=":")}


def figure(out, model_keys):
    """Dose-response: what each ablation does to the switch, the paired effect, and the check."""
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0))
    ax = axes.ravel()
    dose_x = lambda doses: np.array([doses[d]["info"]["frac"] for d in sorted(doses)]) * 100

    for j, mkey in enumerate(model_keys[:2]):
        doses = out[mkey]["doses"]
        x = dose_x(doses)
        for c in COND:
            y = [doses[d]["info"]["median_wtv"][c] for d in sorted(doses)]
            ax[j].plot(x, y, label=COND_LABEL[c], lw=2.2, color=COND_STYLE[c]["color"],
                       marker=COND_STYLE[c]["marker"], ls=COND_STYLE[c]["ls"], ms=8, mec="k",
                       mew=0.6)
        ax[j].axhline(0.5, color="0.35", ls=(0, (4, 3)), lw=1.4)
        ax[j].axhline(0.25, color="0.1", ls=":", lw=1.4)
        ax[j].set_xticks(x)
        ax[j].set_xlabel("heads mean-ablated (% of all heads)")
        ax[j].set_ylabel("median transition width $w_{TV}$")
        d0 = doses[sorted(doses)[0]]["info"]
        ax[j].set_title(f"{mkey} (n={d0['n']} pairs, {d0['n_prefix']} prefixes)", fontsize=10)
        ax[j].legend(fontsize=8.5, loc="center left")
        ax[j].grid(alpha=0.3)

    axd = ax[2]
    for mkey in model_keys:
        doses = out[mkey]["doses"]
        x = dose_x(doses)
        y = np.array([doses[d]["info"]["median_delta"] for d in sorted(doses)])
        ci = np.array([doses[d]["info"]["ci_delta"] for d in sorted(doses)]).T
        axd.errorbar(x, y, yerr=np.abs(ci - y), lw=2.2, capsize=4, ms=8, mec="k", mew=0.6,
                     color=CVD[1] if mkey == "gpt2-large" else CVD[0],
                     marker=MSTYLE[mkey]["marker"], ls=MSTYLE[mkey]["ls"], label=mkey)
    axd.axhline(0.0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    axd.set_yscale("symlog", linthresh=0.01)
    axd.set_xticks(x)
    axd.set_xlabel("heads mean-ablated (% of all heads)")
    axd.set_ylabel("paired median $w_{TV}$(differential) $-$ $w_{TV}$(control)")
    axd.set_title("Paired effect of removing the differential heads\n"
                  "(symlog y; bars = 95% cluster bootstrap over prefixes)", fontsize=10)
    axd.legend(fontsize=8.5)
    axd.grid(alpha=0.3)

    axm = ax[3]
    for mkey in model_keys:
        doses = out[mkey]["doses"]
        x = dose_x(doses)
        base = doses[sorted(doses)[0]]["info"]["median_hcd"]["base"]
        for c in ("ctrl", "diff"):
            y = np.array([doses[d]["info"]["median_hcd"][c] for d in sorted(doses)]) / base
            axm.plot(x, y, color=COND_STYLE[c]["color"], marker=MSTYLE[mkey]["marker"],
                     ls=MSTYLE[mkey]["ls"], lw=2.2, ms=8, mec="k", mew=0.6,
                     label=f"{mkey}, {COND_LABEL[c]}")
    axm.axhline(1.0, color="0.35", ls=(0, (4, 3)), lw=1.4)
    axm.set_xticks(x)
    axm.set_xlabel("heads mean-ablated (% of all heads)")
    axm.set_ylabel("median HCD, relative to no ablation")
    axm.set_title("Manipulation check: the intervention removes\nthe circuit difference it targets",
                  fontsize=10)
    axm.legend(fontsize=7.5)
    axm.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "ablation_causal.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/ablation_causal.png")


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
