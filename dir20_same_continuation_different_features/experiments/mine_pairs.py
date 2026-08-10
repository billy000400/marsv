"""Mine a corpus-derived bank of single-final-token prompt pairs and run the same sweep.

Prefixes come from wikitext-103 (validation split, local cache). For each prefix we take the
model's own top-1 next token as final token A, and a token at rank r (log-uniform in [1, 5000])
as final token B, so the bank spans a wide range of endpoint JSD by construction. Building
input_ids as prefix_ids + [token_id] makes the tokenization validity condition exact.

Writes results/bank_<model>.npz (d-curves) and results/bank_<model>.json (per-pair metrics).
"""
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from common import RESULTS, blocks, load
from run_interp import clean_run, jsd, rel_dist, slerp_lerp_norm, sweep, width_10_90
from analyze import plateau_fraction, tv_width

N_PREFIX = 40          # wikitext prefixes
N_PARTNER = 5          # partner tokens per prefix -> 200 pairs per model
N_ALPHA = 101
MAX_RANK = 5000
SEED = 0


def get_prefixes(tok, rng):
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
        if len(out) == N_PREFIX:
            break
    return out


def main():
    rng_global = np.random.default_rng(SEED)
    alphas = np.linspace(0, 1, N_ALPHA)

    for mkey in ["gpt2-medium", "pythia-410m"]:
        tok, m = load(mkey)
        dev = next(m.parameters()).device
        rng = np.random.default_rng(SEED)                 # same prefixes/ranks for both models
        prefixes = get_prefixes(tok, rng)
        print(f"{mkey}: {len(prefixes)} prefixes")

        rows, curves = [], {}
        for pi, pre in enumerate(prefixes):
            with torch.no_grad():
                lg = m(torch.tensor([pre], device=dev), use_cache=False).logits[0, -1, :].float()
            order = torch.argsort(lg, descending=True)
            ranks = np.unique(np.round(
                np.exp(rng.uniform(np.log(1), np.log(MAX_RANK), size=N_PARTNER * 2))).astype(int))
            ranks = ranks[:N_PARTNER]
            ta = int(order[0])

            for r in ranks:
                tb = int(order[int(r)])
                ida, idb = pre + [ta], pre + [tb]
                reca, lga = clean_run(m, ida)
                recb, lgb = clean_run(m, idb)
                pa, pb = torch.softmax(lga, -1), torch.softmax(lgb, -1)
                vecs, omega, cos = slerp_lerp_norm(reca[0], recb[0], alphas)
                _, lgs = sweep(m, ida, vecs)
                d = rel_dist(lgs, lga.cpu().unsqueeze(0), lgb.cpu().unsqueeze(0))
                w = width_10_90(alphas, d)
                key = f"p{pi}_r{int(r)}"
                curves[key] = d.astype(np.float32)
                rows.append(dict(
                    key=key, prefix_idx=pi, n_prefix=len(pre), rank=int(r),
                    tok_a=tok.convert_ids_to_tokens(ta), tok_b=tok.convert_ids_to_tokens(tb),
                    jsd=jsd(pa, pb), w=None if w is None else float(w),
                    wtv=tv_width(alphas, d), pf=plateau_fraction(d),
                    mono=bool(np.all(np.diff(d) >= -1e-6)),
                    cos_h0=cos, omega=omega,
                    endpoint_err=[float(abs(d[0])), float(abs(d[-1] - 1))]))
            if pi % 10 == 0:
                print(f"  prefix {pi}: {len(rows)} pairs so far")

        np.savez(os.path.join(RESULTS, f"bank_{mkey}.npz"), alphas=alphas, **curves)
        with open(os.path.join(RESULTS, f"bank_{mkey}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        js = np.array([r["jsd"] for r in rows])
        print(f"{mkey}: n={len(rows)} JSD range {js.min():.3f}-{js.max():.3f} "
              f"median {np.median(js):.3f}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
