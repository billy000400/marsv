"""Is the probe's embedding direction a handle on transition width, or only a correlate?

Everything else in this direction is correlational. Here we EDIT the model: for 16 tokens we add a
step to the token's static embedding row along the probe's direction, sized so the probe predicts a
width change of a given number of width units, re-measure that token's anchor width, and compare
against a random direction of the same step norm.

Writes results/intervene.json.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from anchor_width import N_ANCHOR, run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint, jsd_bits
from common import D18, RESULTS
from embed_probe import ALPHAS, ridge_fit

TARGETS = [-0.05, -0.025, 0.025, 0.05]     # width units the probe is asked to move w_hat by
N_TOKENS = 16

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def probe_direction(F, y, rng):
    """Ridge probe on standardised embeddings; returns the raw-space gradient g of the prediction."""
    mu, sd = F.mean(0), F.std(0) + 1e-8
    X = np.hstack([np.ones((len(y), 1)), (F - mu) / sd])
    ym = y.mean()
    perm = rng.permutation(len(y))
    best, best_lam = -np.inf, ALPHAS[0]
    for lam in ALPHAS:
        pred = np.empty(len(y))
        for f in range(5):
            te = perm[f::5]
            tr = np.setdiff1d(perm, te)
            pred[te] = X[te] @ ridge_fit(X[tr], y[tr] - ym, lam) + ym
        s = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if s > best:
            best, best_lam = s, lam
    b = ridge_fit(X, y - ym, best_lam)
    return b[1:] / sd, float(best_lam)          # d(prediction)/d(raw embedding)


@torch.inference_mode()
def measure(model, patcher, tok, anchors, token_id):
    """Median anchor width of one token over frames and anchors, plus its output-distribution shift."""
    w, logps = [], []
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        ids = torch.cat([pre, torch.tensor([[token_id]], device=pre.device)], 1)
        x, z = endpoint(model, patcher, ids)
        logps.append(z.log_softmax(-1))
        for a in anchors:
            aids = torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1)
            xb, zb = endpoint(model, patcher, aids)
            w.append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
    return float(np.nanmedian(w)), torch.stack(logps)


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    used = set(ids_by_str.values())
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    anchors = pool[:: max(1, len(pool) // N_ANCHOR)][:N_ANCHOR]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    W = model.gpt_neox.embed_in.weight
    E = W.detach().float().cpu().numpy()

    w0 = {s: float(np.nanmedian(v["w"]))
          for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    bank = sorted(w0)
    g, lam = probe_direction(np.array([E[ids_by_str[s]] for s in bank]),
                             np.array([w0[s] for s in bank]), np.random.default_rng(3))
    gg = float(g @ g)
    print(f"probe direction: alpha={lam:g}, ||g||={np.linalg.norm(g):.4g}, "
          f"a {TARGETS[-1]:.3f}-unit step has norm {abs(TARGETS[-1]) / gg * np.linalg.norm(g):.4g} "
          f"(median embedding row norm {np.median(np.linalg.norm(E, axis=1)):.4g})", flush=True)

    # 16 tokens spread over the measured-width range, so the test is not run only on narrow tokens
    order = sorted(bank, key=lambda s: w0[s])
    sel = [order[j] for j in np.linspace(0, len(order) - 1, N_TOKENS).round().astype(int)]
    rng = np.random.default_rng(7)

    patcher = Patcher(model)
    rows = {}
    for k, s in enumerate(sel):
        i = ids_by_str[s]
        base_row = W.data[i].clone()
        base_w, base_logp = measure(model, patcher, tok, anchors, i)
        r = dict(token_id=i, base_w=base_w, w0_ref=w0[s], probe=[], random=[])
        rdir = rng.normal(size=E.shape[1])
        rdir = rdir / np.linalg.norm(rdir)
        for target in TARGETS:
            step = (target / gg) * g                       # probe predicts +target width units
            for tag, delta in (("probe", step),
                               ("random", np.linalg.norm(step) * rdir)):
                W.data[i] = base_row + torch.tensor(delta, dtype=W.dtype, device=W.device)
                w_new, logp_new = measure(model, patcher, tok, anchors, i)
                out_shift = float(jsd_bits(base_logp, logp_new).mean())
                r[tag].append(dict(target=target, dw=w_new - base_w, w=w_new,
                                   step_norm=float(np.linalg.norm(delta)), out_jsd=out_shift))
                W.data[i] = base_row
        rows[s] = r
        print(f"[{k + 1}/{N_TOKENS}] {s!r} base w {base_w:.3f} | "
              + " ".join(f"{d['target']:+.3f}->{d['dw']:+.3f}" for d in r["probe"]), flush=True)
    patcher.close()

    tg = np.array([d["target"] for s in rows for d in rows[s]["probe"]])
    dp = np.array([d["dw"] for s in rows for d in rows[s]["probe"]])
    dr = np.array([d["dw"] for s in rows for d in rows[s]["random"]])
    oj_p = np.array([d["out_jsd"] for s in rows for d in rows[s]["probe"]])
    oj_r = np.array([d["out_jsd"] for s in rows for d in rows[s]["random"]])
    slope = float(np.polyfit(tg, dp, 1)[0])
    res = dict(n_tokens=len(rows), targets=TARGETS, alpha=lam, tokens=rows,
               rho_probe=[float(x) for x in spearmanr(tg, dp)[:2]],
               rho_random=[float(x) for x in spearmanr(tg, dr)[:2]],
               slope_probe=slope, slope_random=float(np.polyfit(tg, dr, 1)[0]),
               mean_abs_dw_probe=float(np.abs(dp).mean()),
               mean_abs_dw_random=float(np.abs(dr).mean()),
               frac_sign_agree=float(np.mean(np.sign(dp) == np.sign(tg))),
               frac_sign_agree_random=float(np.mean(np.sign(dr) == np.sign(tg))),
               out_jsd_probe=float(oj_p.mean()), out_jsd_random=float(oj_r.mean()))
    json.dump(res, open(os.path.join(RESULTS, "intervene.json"), "w"), indent=1)
    print(f"probe direction:  rho(target, dw) {res['rho_probe'][0]:+.3f} "
          f"(p={res['rho_probe'][1]:.1e}), slope {slope:+.3f}, "
          f"sign agreement {res['frac_sign_agree']:.2f}, mean |dw| {res['mean_abs_dw_probe']:.4f}")
    print(f"random direction: rho(target, dw) {res['rho_random'][0]:+.3f} "
          f"(p={res['rho_random'][1]:.1e}), slope {res['slope_random']:+.3f}, "
          f"sign agreement {res['frac_sign_agree_random']:.2f}, "
          f"mean |dw| {res['mean_abs_dw_random']:.4f}")
    print(f"output shift in bits: probe {res['out_jsd_probe']:.4f}, random {res['out_jsd_random']:.4f}")


if __name__ == "__main__":
    main()
