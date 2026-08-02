"""Which of blocks 1-4 carries the plateau sharpness, and what does the width change track?

mlp_gain_probe.py showed the block-1..4 MLPs as a GROUP causally set the sharpness: deleting all
four (g=0) returns the median width to the untrained value, while deleting blocks 8-11 does almost
nothing. PLAN's named successor asks two things of that result:

  (a) which of blocks 1,2,3,4 carries it -- delete each early block's MLP on its own (g=0);
  (b) does the resulting width change track the endpoint PLAUSIBILITY gap or the DECISION structure?

For (b) we record, under every condition, the two candidate mediators for each pair:

  plausibility  Experiment 5's confound is max_p_i = max(p(A|context), p(B|context)) under THAT
                model, which correlates with w at rho = -0.46 raw and -0.59 partialling out the
                endpoint logit separation sep_i. If plausibility mediates the ablation effect, the
                pairs whose max_p moves most should widen most. |log10 p(A) - log10 p(B)| is kept
                as the second measure PLAN 5.3 names (it correlates only weakly, rho = +0.18).
  decision      whether the two endpoints still predict different next characters (am0 != am1), how
                many distinct argmax regions the path visits, and |t* - t_flip|. If the decision
                structure survives intact while d(t) goes straight, the plateau is not the decision.

Same 150-pair subsample, block-0 interpolation, step-30000 checkpoint as mlp_gain_probe.py, so
every width here is directly comparable to the group numbers. The all-four condition is re-run as
an in-run reference for how much of the group effect a single block recovers.

Raw -> results/mlp_block_scan_raw.npz, stats -> results/mlp_block_scan_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, self_test
from allpairs_sweep import load_vocab, load_model, pair_stats, CONTEXT, N_T, FINAL_STEP
from mlp_gain_probe import set_gain, restore, BLOCK, N_PAIRS, SEED, EARLY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def sweep(model, seqs, pairs, ts, device, p_next, sep):
    """Run the frozen assay for every pair under the currently installed gain.

    Returns per-pair arrays: width, t*, strict flag, t_flip, n_argmax regions, endpoints-differ
    flag, and the two plausibility measures + the endpoint logit separation under THIS model.
    """
    w, tstar, strict, tflip, nam, differ, maxp, dlogp, seps = ([] for _ in range(9))
    lp = np.log10(p_next + 1e-30)
    for ia, ib in pairs:
        s = pair_stats(ts, run_pair(model, seqs[ia], seqs[ib], BLOCK, ts, device))
        w.append(np.nan if s["w"] is None else s["w"])
        tstar.append(np.nan if s["t_star"] is None else s["t_star"])
        strict.append(s["plateau"])
        tflip.append(np.nan if s["t_flip"] is None else s["t_flip"])
        nam.append(s["n_argmax"])
        differ.append(s["am0"] != s["am1"])
        maxp.append(max(p_next[ia], p_next[ib]))
        dlogp.append(abs(lp[ia] - lp[ib]))
        seps.append(sep[ia, ib])
    f = lambda v, dt=float: np.array(v, dtype=dt)
    return dict(w=f(w), tstar=f(tstar), strict=f(strict, bool), tflip=f(tflip),
                n_argmax=f(nam), differ=f(differ, bool),
                maxp=f(maxp), dlogp=f(dlogp), sep=f(seps))


def endpoint_stats(model, stoi, seqs, device):
    """p(next char | context) and the pairwise endpoint logit-vector separation, under this model."""
    with torch.no_grad():
        ctx = torch.tensor([[stoi[c] for c in CONTEXT]], device=device)
        p_next = torch.softmax(model(ctx)[0][0, -1], dim=-1).float().cpu().numpy()
        lg = model(torch.from_numpy(np.stack(seqs)).to(device))[0][:, -1, :]
        sep = torch.cdist(lg.float(), lg.float()).cpu().numpy()
    return p_next, sep


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return None
    return round(float(spearmanr(x[m], y[m]).statistic), 4)


def partial_rho(x, y, z):
    """Spearman of x,y controlling for z -- the frozen definition from analyze_allpairs.py."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return None
    rxy, rxz, ryz = (spearmanr(a[m], b[m]).statistic for a, b in ((x, y), (x, z), (y, z)))
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return None if den == 0 else round(float((rxy - rxz * ryz) / den), 4)


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    model = load_model(FINAL_STEP, device)
    saved = {}
    t0 = time.time()

    conds = [("baseline", [])] + [(f"block{l}_g0", [l]) for l in EARLY] + [("early_all_g0", EARLY)]

    out, res = {}, {}
    for name, blocks in conds:
        restore(model, saved)
        if blocks:
            set_gain(model, blocks, 0.0, saved)
        r = sweep(model, seqs, pairs, ts, device, *endpoint_stats(model, stoi, seqs, device))
        res[name] = r
        out[name] = {"blocks": blocks,
                     "median_w": round(float(np.nanmedian(r["w"])), 4),
                     "iqr_w": [round(float(np.nanpercentile(r["w"], 25)), 4),
                               round(float(np.nanpercentile(r["w"], 75)), 4)],
                     "strict_frac": round(float(r["strict"].mean()), 4),
                     "median_t_star": round(float(np.nanmedian(r["tstar"])), 4),
                     # decision structure under this condition
                     "frac_endpoints_differ": round(float(r["differ"].mean()), 4),
                     "median_n_argmax": float(np.median(r["n_argmax"])),
                     "median_abs_tstar_minus_tflip":
                         round(float(np.nanmedian(np.abs(r["tstar"] - r["tflip"]))), 4),
                     # plausibility under this condition (Experiment 5's frozen definitions)
                     "median_max_p": round(float(np.median(r["maxp"])), 4),
                     "rho_w_vs_max_p": rho(r["w"], r["maxp"]),
                     "partial_rho_w_vs_max_p_given_sep": partial_rho(r["w"], r["maxp"], r["sep"]),
                     "rho_w_vs_abs_dlogp": rho(r["w"], r["dlogp"])}
        print(f"{name:14s} median_w={out[name]['median_w']:.4f} "
              f"strict={out[name]['strict_frac']:.3f} max_p={out[name]['median_max_p']:.3f} "
              f"rho(w,max_p)={out[name]['rho_w_vs_max_p']} "
              f"partial={out[name]['partial_rho_w_vs_max_p_given_sep']} ({time.time()-t0:.0f}s)",
              flush=True)
    restore(model, saved)

    base = res["baseline"]
    group_dw = float(np.nanmedian(res["early_all_g0"]["w"] - base["w"]))
    for name, r in res.items():
        if name == "baseline":
            continue
        dw = r["w"] - base["w"]
        dmaxp = r["maxp"] - base["maxp"]
        out[name].update({
            "median_paired_dw": round(float(np.nanmedian(dw)), 4),
            "frac_w_increased": round(float(np.nanmean(dw > 0)), 4),
            "median_abs_paired_dtstar": round(float(np.nanmedian(np.abs(r["tstar"] - base["tstar"]))), 4),
            "frac_of_group_effect": round(float(np.nanmedian(dw) / group_dw), 4),
            # mediation test: does the pair-by-pair widening track the pair-by-pair plausibility change?
            "median_paired_dmax_p": round(float(np.median(dmaxp)), 4),
            "rho_dw_vs_dmax_p": rho(dw, dmaxp),
            "rho_dw_vs_base_max_p": rho(dw, base["maxp"])})

    summary = {"context": CONTEXT, "step": FINAL_STEP, "block": BLOCK, "n_t": N_T,
               "n_pairs": N_PAIRS, "seed": SEED, "early_blocks": EARLY,
               "conditions": out}
    print(json.dumps(summary, indent=2), flush=True)
    with open(os.path.join(RES, "mlp_block_scan_summary.json"), "w") as f:
        json.dump({"summary": summary,
                   "pairs": [[chars[a], chars[b]] for a, b in pairs]}, f, indent=1)
    np.savez_compressed(os.path.join(RES, "mlp_block_scan_raw.npz"), ts=ts,
                        **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items()})
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
