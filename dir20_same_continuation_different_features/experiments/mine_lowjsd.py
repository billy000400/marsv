"""S9: mine a bank of pairs that are specifically LOW output JSD, then sweep them.

Experiment 6 previously filtered a general bank down to JSD < 0.1, which left n = 38 in
gpt2-large. Here candidate partner tokens are screened for low JSD before the sweep, so nearly
every swept pair is usable and n rises to a few hundred per model.

For each wikitext prefix: token A = the model's own top-1 next token; candidate partners are taken
at near-top ranks plus log-uniform ranks up to MAX_RANK; one batched forward gives every candidate's
next-token distribution, and only candidates with JSD(A, B) < JSD_LOW are kept (spread over rank so
the bank is not all rank-2 tokens). Kept pairs get the same block-0 SLERP sweep as everywhere else.

Writes results/lowjsd_<model>.npz (d-curves) and results/lowjsd_<model>.json (per-pair metrics).
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset

from analyze import plateau_fraction, tv_width
from common import RESULTS, load
from feature_plateau import intermediate_plateau_width
from run_interp import clean_run, jsd, rel_dist, slerp_lerp_norm, sweep, width_10_90

N_PREFIX = 200         # wikitext prefixes (vs 40 in the general bank)
N_KEEP = 6             # low-JSD partners kept per prefix; ~half the prefixes yield any
NEAR_RANKS = list(range(1, 21))   # candidate partners at ranks 1..20
N_FAR = 10             # plus this many log-uniform ranks in [21, MAX_RANK]
MAX_RANK = 2000
JSD_LOW = 0.10
N_ALPHA = 101
SEED = 7               # not mine_pairs.SEED: this is an independent bank


def get_prefixes(tok, rng, n_prefix=N_PREFIX):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    out = []
    for t in ds["text"]:
        s = t.strip()
        if len(s) < 400 or s.startswith("="):
            continue
        ids = tok(s)["input_ids"]
        n = int(rng.integers(10, 41))
        if len(ids) < n + 5:
            continue
        out.append(ids[:n])
        if len(out) == n_prefix:
            break
    return out


def screen(m, pre, order, cand_ranks, dev):
    """Next-token JSD between prefix+[top-1] and prefix+[rank-r token], for every candidate r."""
    ta = int(order[0])
    ids = [pre + [ta]] + [pre + [int(order[r])] for r in cand_ranks]
    with torch.no_grad():
        lg = m(torch.tensor(ids, device=dev), use_cache=False).logits[:, -1, :].float()
    p = torch.softmax(lg, -1)
    return [jsd(p[0], p[i + 1]) for i in range(len(cand_ranks))]


def main(model_keys):
    alphas = np.linspace(0, 1, N_ALPHA)
    for mkey in model_keys:
        tok, m = load(mkey)
        dev = next(m.parameters()).device
        rng = np.random.default_rng(SEED)
        prefixes = get_prefixes(tok, rng)
        print(f"{mkey}: {len(prefixes)} prefixes", flush=True)

        rows, curves = [], {}
        for pi, pre in enumerate(prefixes):
            with torch.no_grad():
                lg = m(torch.tensor([pre], device=dev), use_cache=False).logits[0, -1, :].float()
            order = torch.argsort(lg, descending=True)
            far = np.unique(np.round(np.exp(rng.uniform(
                np.log(21), np.log(MAX_RANK), size=N_FAR))).astype(int))
            cand = NEAR_RANKS + [int(r) for r in far]
            js = screen(m, pre, order, cand, dev)
            ok = [(r, j) for r, j in zip(cand, js) if j < JSD_LOW]
            if not ok:
                continue
            take = np.unique(np.linspace(0, len(ok) - 1, min(N_KEEP, len(ok))).round().astype(int))
            ta = int(order[0])

            for idx in take:
                r, j = ok[int(idx)]
                tb = int(order[r])
                ida, idb = pre + [ta], pre + [tb]
                reca, lga = clean_run(m, ida)
                recb, lgb = clean_run(m, idb)
                vecs, omega, cos = slerp_lerp_norm(reca[0], recb[0], alphas)
                _, lgs = sweep(m, ida, vecs, layer=0)
                d = rel_dist(lgs, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
                w = width_10_90(alphas, d)
                ipw, lvl = intermediate_plateau_width(alphas, d)
                key = f"p{pi}_r{r}"
                curves[key] = d.astype(np.float32)
                rows.append(dict(
                    key=key, prefix_idx=pi, n_prefix=len(pre), rank=int(r),
                    tok_a=tok.convert_ids_to_tokens(ta), tok_b=tok.convert_ids_to_tokens(tb),
                    id_a=ta, id_b=tb, jsd=float(j),
                    w=None if w is None else float(w), wtv=tv_width(alphas, d),
                    pf=plateau_fraction(d), ipw=float(ipw), level=lvl,
                    mono=bool(np.all(np.diff(d) >= -1e-6)), cos_h0=cos, omega=omega,
                    endpoint_err=[float(abs(d[0])), float(abs(d[-1] - 1))]))
            if pi % 10 == 0:
                print(f"  prefix {pi}: {len(rows)} pairs so far", flush=True)

        np.savez(os.path.join(RESULTS, f"lowjsd_{mkey}.npz"), alphas=alphas, **curves)
        with open(os.path.join(RESULTS, f"lowjsd_{mkey}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        jj = np.array([r["jsd"] for r in rows])
        print(f"{mkey}: n={len(rows)} prefixes={len({r['prefix_idx'] for r in rows})} "
              f"JSD max {jj.max():.4f} median {np.median(jj):.4f} "
              f"max endpoint err {max(max(r['endpoint_err']) for r in rows):.2e}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ("gpt2-small", "gpt2-medium", "gpt2-large"))
