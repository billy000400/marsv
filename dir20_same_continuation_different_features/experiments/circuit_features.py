"""S9: the advisor's hypothesis with an actual feature-level independent variable.

Experiment 6 used IRD (residual-stream cosine distance) as a proxy for "different circuits or
features". This script measures circuit/feature difference directly, on the low-JSD bank mined by
mine_lowjsd.py:

  SFD  - SAE feature-set disjointness (gpt2-small only): Jaccard distance between the sets of
         features that fire in a trained sparse autoencoder, over all 12 residual-stream hook
         points. This is the measurement the operator's feedback asked for.
  SFC  - the same SAE features compared as a weighted profile: 1 - cos of the concatenated
         activation vectors (no threshold).
  HCD  - head-contribution distance: how differently each attention head writes to the final token,
         averaged over heads and weighted by how much that head writes.
  HSD  - attention-head-set disjointness: which heads write most to the final token.
  NSD  - MLP neuron-set disjointness: top-K post-GELU neurons per block.
  IRD  - the previous residual-stream proxy, kept for comparison.

Dependent variables (from the swept curves): IPW, the intermediate-plateau width (do the two
prompts rest at a level between A and B?), and w_TV (does the curve snap between the two endpoint
plateaus at all?).

Writes results/circuit_features.json, plots/sae_features.png, plots/circuit_forest.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import pearsonr, rankdata, spearmanr

from analyze_bank import cluster_boot
from common import DEV, PLOTS, RESULTS, blocks, load
from mine_lowjsd import N_PREFIX, SEED, get_prefixes

MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large"]
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_MODEL = "gpt2-small"
TOP_HEAD_FRAC = 0.10     # head set = top 10% of all heads by contribution norm
TOP_NEURON = 64          # neuron set = top 64 post-GELU neurons per block
IPW_LINEAR, IPW_THR = 0.10, 0.20
WTV_LINEAR, WTV_SHARP = 0.5, 0.25
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
STYLE = {"gpt2-small": dict(color=CVD[0], marker="o"),
         "gpt2-medium": dict(color=CVD[1], marker="s"),
         "gpt2-large": dict(color=CVD[2], marker="^")}
IV_LABEL = {"sfd": "SAE feature-set disjointness (SFD)",
            "sfc": "SAE feature-profile distance (SFC)",
            "hcd": "head-contribution distance (HCD)",
            "hsd": "head-set disjointness (HSD)",
            "nsd": "neuron-set disjointness (NSD)",
            "ird": "internal repr. distance (IRD)"}
IVS_SAE = ["sfd", "sfc", "hcd", "hsd", "nsd", "ird"]
IVS = ["hcd", "hsd", "nsd", "ird"]
DV_LABEL = {"ipw": "intermediate-plateau width (IPW)", "wtv": "transition width $w_{TV}$"}


# ---------------------------------------------------------------- SAE (gpt2-small only)
def load_saes(n_blocks=12):
    """Encoders for the residual stream after each block.

    Hook-point correspondence: `blocks.{L+1}.hook_resid_pre` is the residual stream after block L,
    and `blocks.11.hook_resid_post` covers the last block. TransformerLens centres the residual
    stream over d_model (center_writing_weights), so we centre the HF activations to match.
    """
    names = [f"blocks.{i + 1}.hook_resid_pre" for i in range(n_blocks - 1)]
    names.append(f"blocks.{n_blocks - 1}.hook_resid_post")
    saes = []
    for nm in names:
        w = load_file(hf_hub_download(SAE_REPO, f"{nm}/sae_weights.safetensors"))
        saes.append({k: v.to(DEV) for k, v in w.items()})
    return saes


def sae_encode(sae, x):
    """Feature activations for one d_model vector (centred, as TransformerLens would give it)."""
    xc = x - x.mean()
    return torch.relu((xc - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"])


def sae_check(saes, m, ids_list):
    """Validation: mean L0 and fraction of variance explained, per hook point.

    `ids_list` must hold one prompt per distinct prefix — variance is measured around the mean over
    prompts, which is meaningless for prompts that share a prefix and differ in one token.
    """
    l0, fve = [], []
    feats = [[] for _ in saes]
    xs = [[] for _ in saes]
    for ids in ids_list:
        rec = probe(m, ids)["resid"]
        for i, sae in enumerate(saes):
            feats[i].append(sae_encode(sae, rec[i]))
            xs[i].append(rec[i] - rec[i].mean())
    for i, sae in enumerate(saes):
        F = torch.stack(feats[i])
        X = torch.stack(xs[i])
        Xh = F @ sae["W_dec"] + sae["b_dec"]
        l0.append(float((F > 0).float().sum(1).mean()))
        fve.append(float(1 - ((X - Xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()))
    return l0, fve


# ---------------------------------------------------------------- activations
def probe(m, ids):
    """Final-token resid_post, per-head attention contribution norms and post-GELU MLP neurons."""
    bl = blocks(m)
    nh = m.config.n_head
    rec, zs, neu = {}, {}, {}
    hs = []

    def mk_resid(i):
        def h(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            rec[i] = x[0, -1, :].detach().float().clone()
        return h

    def mk_z(i):
        def h(mod, inp):
            zs[i] = inp[0][0, -1, :].detach().float().clone()
        return h

    def mk_neu(i):
        def h(mod, inp, out):
            neu[i] = out[0, -1, :].detach().float().clone()
        return h

    for i, b in enumerate(bl):
        hs.append(b.register_forward_hook(mk_resid(i)))
        hs.append(b.attn.c_proj.register_forward_pre_hook(mk_z(i)))
        hs.append(b.mlp.act.register_forward_hook(mk_neu(i)))
    with torch.no_grad():
        m(torch.tensor([ids], device=DEV), use_cache=False)
    for h in hs:
        h.remove()

    # Per-head contribution to the final token's residual stream: z_h @ W_O[h].
    contrib = []
    for i, b in enumerate(bl):
        W = b.attn.c_proj.weight.detach().float()          # [d_model, d_model], rows = input dims
        dh = W.shape[0] // nh
        contrib.append(torch.stack([zs[i][h_ * dh:(h_ + 1) * dh] @ W[h_ * dh:(h_ + 1) * dh, :]
                                    for h_ in range(nh)]))
    C = torch.stack(contrib)                               # [n_block, n_head, d_model]
    return dict(resid=rec, contrib=C, head=C.norm(dim=-1).cpu(), neu=neu)


def top_set(v, k, tag):
    return {(tag, int(j)) for j in torch.topk(v, k).indices}


def measures(pa, pb, saes=None):
    """All circuit/feature difference measures between two probed prompts."""
    ird = float(np.mean([1.0 - float((pa["resid"][i] * pb["resid"][i]).sum()
                                     / (pa["resid"][i].norm() * pb["resid"][i].norm()))
                         for i in sorted(pa["resid"])]))
    ua, ub = pa["head"].flatten(), pb["head"].flatten()
    k = max(1, int(round(TOP_HEAD_FRAC * ua.numel())))
    sa, sb = top_set(ua, k, "h"), top_set(ub, k, "h")
    hsd = 1.0 - len(sa & sb) / k
    ca = pa["contrib"].reshape(-1, pa["contrib"].shape[-1])
    cb = pb["contrib"].reshape(-1, pb["contrib"].shape[-1])
    cos = torch.nn.functional.cosine_similarity(ca, cb, dim=-1)
    w = (ca.norm(dim=-1) + cb.norm(dim=-1))
    hcd = float(((1.0 - cos) * w).sum() / w.sum())
    na, nb = set(), set()
    for i in sorted(pa["neu"]):
        na |= top_set(pa["neu"][i], TOP_NEURON, i)
        nb |= top_set(pb["neu"][i], TOP_NEURON, i)
    nsd = 1.0 - len(na & nb) / len(na | nb)
    out = dict(ird=ird, hcd=hcd, hsd=float(hsd), nsd=float(nsd))
    if saes is not None:
        fa, fb = [], []
        for i, sae in enumerate(saes):
            fa.append(sae_encode(sae, pa["resid"][i]))
            fb.append(sae_encode(sae, pb["resid"][i]))
        Fa, Fb = torch.cat(fa), torch.cat(fb)
        aa, ab = Fa > 0, Fb > 0
        inter = float((aa & ab).sum())
        union = float((aa | ab).sum())
        out["sfd"] = 1.0 - inter / union
        out["sfc"] = 1.0 - float((Fa * Fb).sum() / (Fa.norm() * Fb.norm()))
        out["l0"] = float(aa.sum())
    return out


# ---------------------------------------------------------------- stats
def rho_min(n):
    """Smallest |rho| a two-sided test at n would detect at alpha=0.05."""
    return float(np.tanh(1.96 / np.sqrt(max(n - 3, 1))))


def partial_rho(x, y, ctrls):
    """Spearman partial correlation: rank everything, regress out the controls, correlate what is
    left. Controls here are the residual JSD inside the low band and the block-0 angle Omega."""
    X, Y = rankdata(x), rankdata(y)
    C = np.column_stack([rankdata(c) for c in ctrls] + [np.ones(len(x))])
    rx = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
    ry = Y - C @ np.linalg.lstsq(C, Y, rcond=None)[0]
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def holm(ps):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main(model_keys=MODELS):
    rng_boot = np.random.default_rng(0)
    out = {}
    saes_cache = None

    for mkey in model_keys:
        rows = json.load(open(os.path.join(RESULTS, f"lowjsd_{mkey}.json")))
        tok, m = load(mkey)
        prefixes = get_prefixes(tok, np.random.default_rng(SEED), N_PREFIX)
        saes = None
        if mkey == SAE_MODEL:
            saes = saes_cache = load_saes(len(blocks(m)))

        recs = []
        for r in rows:
            pre = prefixes[r["prefix_idx"]]
            pa = probe(m, pre + [r["id_a"]])
            pb = probe(m, pre + [r["id_b"]])
            recs.append({**r, **measures(pa, pb, saes)})
        info = dict(n=len(recs), n_prefix=len({r["prefix_idx"] for r in recs}),
                    jsd_max=max(r["jsd"] for r in recs),
                    jsd_median=float(np.median([r["jsd"] for r in recs])),
                    median_ipw=float(np.median([r["ipw"] for r in recs])),
                    frac_intermediate=float(np.mean([r["ipw"] > IPW_THR for r in recs])),
                    median_wtv=float(np.median([r["wtv"] for r in recs])),
                    frac_sharp=float(np.mean([r["wtv"] < WTV_SHARP for r in recs])),
                    rho_min=rho_min(len(recs)), tests={})
        if saes is not None:
            seen, ids_list = set(), []
            for r in rows:                       # one prompt per distinct prefix
                if r["prefix_idx"] not in seen:
                    seen.add(r["prefix_idx"])
                    ids_list.append(prefixes[r["prefix_idx"]] + [r["id_a"]])
            l0, fve = sae_check(saes, m, ids_list)
            info["sae_l0"] = [round(v, 1) for v in l0]
            info["sae_fve"] = [round(v, 3) for v in fve]

        cl = {}
        for r in recs:
            cl.setdefault(r["prefix_idx"], []).append(r)
        ivs = IVS_SAE if saes is not None else IVS
        for iv in ivs:
            for dv in ["ipw", "wtv"]:
                x = np.array([r[iv] for r in recs])
                y = np.array([r[dv] for r in recs])
                rho, p = spearmanr(x, y)
                lo, hi = cluster_boot(
                    cl, lambda rr, iv=iv, dv=dv: (
                        spearmanr([q[iv] for q in rr], [q[dv] for q in rr])[0]
                        if len({q[iv] for q in rr}) > 2 else None), rng_boot)
                pr, pp = partial_rho(x, y, [[r["jsd"] for r in recs],
                                            [r["omega"] for r in recs]])
                info["tests"][f"{iv}_{dv}"] = dict(rho=float(rho), p=float(p), ci=[lo, hi],
                                                   rho_partial=pr, p_partial=pp,
                                                   median_iv=float(np.median(x)))
        info["iv_spread"] = {iv: [float(np.percentile([r[iv] for r in recs], q))
                                  for q in (25, 50, 75)] for iv in ivs}
        out[mkey] = dict(info=info, rows=recs)
        print(mkey, json.dumps({k: v for k, v in info.items() if k != "tests"}), flush=True)
        for k, v in info["tests"].items():
            print(f"   {k:9s} rho={v['rho']:+.3f} p={v['p']:.3g} "
                  f"CI=[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] "
                  f"partial={v['rho_partial']:+.3f} (p={v['p_partial']:.3g})", flush=True)
        del m
        torch.cuda.empty_cache()

    # Holm correction over the three pre-specified primary tests (feature-level IV vs IPW).
    prim = [(mk, "sfd_ipw" if mk == SAE_MODEL else "hsd_ipw") for mk in model_keys]
    ps = [out[mk]["info"]["tests"][t]["p"] for mk, t in prim]
    for (mk, t), q in zip(prim, holm(ps)):
        out[mk]["info"]["primary"] = dict(test=t, p_holm=float(q))

    with open(os.path.join(RESULTS, "circuit_features.json"), "w") as f:
        json.dump(out, f, indent=1)
    figures(out, model_keys)
    return out


# ---------------------------------------------------------------- figures
def pfmt(p):
    return f"p={p:.0e}" if p < 0.01 else f"p={p:.2f}"


def figures(out, model_keys):
    if SAE_MODEL in out:
        info = out[SAE_MODEL]["info"]
        rows = out[SAE_MODEL]["rows"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        for j, dv in enumerate(["ipw", "wtv"]):
            t = info["tests"][f"sfd_{dv}"]
            ax[j].scatter([r["sfd"] for r in rows], [r[dv] for r in rows], s=26, alpha=0.7,
                          edgecolor="k", linewidth=0.4, **STYLE[SAE_MODEL])
            ref, thr = (IPW_LINEAR, IPW_THR) if dv == "ipw" else (WTV_LINEAR, WTV_SHARP)
            ax[j].axhline(ref, color="0.35", ls=(0, (4, 3)), lw=1.4)
            ax[j].axhline(thr, color="0.1", ls=":", lw=1.4)
            ax[j].set_xlabel(IV_LABEL["sfd"])
            ax[j].set_ylabel(DV_LABEL[dv])
            ax[j].set_title(f"{SAE_MODEL}: SAE features vs {DV_LABEL[dv].split('(')[0].strip()}\n"
                            f"$\\rho$={t['rho']:+.2f} ({pfmt(t['p'])}, n={info['n']}), "
                            f"detectable $|\\rho|\\geq${info['rho_min']:.2f}", fontsize=9.5)
            ax[j].text(0.99, 0.97, f"dashed = linear response ({ref})\ndotted = threshold ({thr})",
                       fontsize=7.5, color="0.15", ha="right", va="top",
                       transform=ax[j].transAxes,
                       bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.9, pad=2))
            ax[j].grid(alpha=0.3)
        L = np.arange(len(info["sae_l0"]))
        ax[2].plot(L, info["sae_fve"], color=CVD[0], marker="o", lw=2, label="variance explained")
        ax[2].set_ylim(0, 1.02)
        ax[2].set_xlabel("residual stream after block")
        ax[2].set_ylabel("fraction of variance explained")
        a2 = ax[2].twinx()
        a2.plot(L, info["sae_l0"], color=CVD[1], marker="s", ls="--", lw=2, label="active features")
        a2.set_ylabel("active features per token ($L_0$)")
        ax[2].set_title("SAE validation on this bank's prompts", fontsize=9.5)
        h1, l1 = ax[2].get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        ax[2].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "sae_features.png"), dpi=140)
        plt.close(fig)

    labels, rhos, los, his, mkeys, prim = [], [], [], [], [], []
    for mk in model_keys:
        info = out[mk]["info"]
        for name, t in info["tests"].items():
            iv, dv = name.rsplit("_", 1)
            labels.append(f"{mk}  {iv.upper()} → {dv.upper()}  (n={info['n']})")
            rhos.append(t["rho"])
            los.append(t["ci"][0] if t["ci"][0] is not None else t["rho"])
            his.append(t["ci"][1] if t["ci"][1] is not None else t["rho"])
            mkeys.append(mk)
            prim.append(name == info["primary"]["test"])
    y = np.arange(len(labels))[::-1]
    fig, axf = plt.subplots(figsize=(8.6, 0.42 * len(labels) + 1.5))
    rmin = np.mean([out[mk]["info"]["rho_min"] for mk in model_keys])
    axf.axvspan(-rmin, rmin, color="0.85", zorder=0)
    axf.axvline(0, color="0.2", ls=(0, (4, 3)), lw=1.3, zorder=1)
    for i, (r, lo, hi, mk, pr) in enumerate(zip(rhos, los, his, mkeys, prim)):
        axf.errorbar(r, y[i], xerr=[[r - lo], [hi - r]], fmt=STYLE[mk]["marker"],
                     color=STYLE[mk]["color"], ms=9 if pr else 6, capsize=3, lw=1.6,
                     mec="k", mew=1.4 if pr else 0.5, zorder=3)
    for i in range(1, len(labels)):
        if mkeys[i] != mkeys[i - 1]:
            axf.axhline(y[i] + 0.5, color="0.6", lw=0.8)
    axf.set_yticks(y)
    axf.set_yticklabels(labels, fontsize=8)
    axf.set_xlabel("Spearman $\\rho$ (circuit/feature difference vs plateau statistic)")
    axf.set_title("Circuit-difference measures against plateau statistics, output JSD held low\n"
                  f"gray band = undetectable $|\\rho|<{rmin:.2f}$ at these sample sizes\n"
                  "large markers, thick edges = pre-specified primary tests", fontsize=9.5)
    axf.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "circuit_forest.png"), dpi=140)
    plt.close(fig)
    print("wrote plots/sae_features.png + plots/circuit_forest.png")


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
