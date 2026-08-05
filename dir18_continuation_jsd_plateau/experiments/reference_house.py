"""Feedback #4: run the two named reference pairs through the plateau assay on Pythia.

Pairs, in the carrier the operator named plus the three project carriers:
    "My house is" + ` big` / ` large`      (near-synonyms, low corpus divergence)
    "My house is" + ` big` / ` in`         (different word class, high corpus divergence)

Same assay as the main experiment (post-block-0 patch, norm-rescaled SLERP over 50 positions,
final-position logits restricted to corpus-observed target IDs). In addition to the relative-logit
coordinate d(t) and its 10%-90% width w, we record the ABSOLUTE movement of the output distribution

    M(t) = JSD( p(t), p(0) )   [bits]

so "does the output stay put" can be answered without the normalisation that forces d to run 0 -> 1.

Usage: python3 reference_house.py
Writes results/reference_house.json (curves + summaries for every model/pair/context).
"""
import json
import os

import numpy as np
import torch

from assay import GRID, Patcher, endpoint_states, slerp_bank
from common import DATA, MODEL, RESULTS, load
from curve_metrics import metrics

CARRIERS = ["My house is", "The thing was", "They said it was", "I thought it was"]
PAIRS = [(" big", " large"), (" big", " in")]
RUNS = [("1.4B trained", MODEL, "step143000"),
        ("1.4B untrained", MODEL, "step0"),
        ("410M trained", "EleutherAI/pythia-410m-deduped", "step143000")]


def jsd_bits(p, q):
    """Base-2 Jensen-Shannon divergence between rows of p and a single row q."""
    m = 0.5 * (p + q)
    kl = lambda a: (a * (torch.log2(a.clamp_min(1e-30)) - torch.log2(m.clamp_min(1e-30)))).sum(-1)
    return 0.5 * (kl(p) + kl(q))


@torch.inference_mode()
def run_pair_abs(model, ids_a, ids_b, valid):
    """d(t), M(t) and the summaries for one pair in one carrier context."""
    xa, xb, za, zb = endpoint_states(model, ids_a, ids_b, 0)
    bank, fallback, _ = slerp_bank(xa, xb)
    p = Patcher(model, 0)
    p.bank = bank
    z = model(ids_a.repeat(len(GRID), 1)).logits[:, -1, :].float()[:, valid]
    p.close()
    za, zb = za[valid], zb[valid]

    da, db = (z - za).norm(dim=1), (z - zb).norm(dim=1)
    d = (da / (da + db)).cpu().numpy()
    probs = torch.softmax(z, -1)
    m_abs = jsd_bits(probs, probs[0]).cpu().numpy()

    mm = metrics(d, GRID)
    return dict(d=[float(x) for x in d], m_abs=[float(x) for x in m_abs],
                w=mm["w"], edge_drift=mm["edge_drift"], valid=bool(mm["valid"]),
                out_jsd=float(jsd_bits(probs[-1], probs[0])),
                err=max(float((z[0] - za).abs().max()), float((z[-1] - zb).abs().max()))
                    / float(za.abs().max()),
                fallback=bool(fallback))


if __name__ == "__main__":
    valid = torch.tensor(np.load(os.path.join(DATA, "reference_valid.npy")),
                         device="cuda", dtype=torch.long)
    out = {"grid": [float(t) for t in GRID], "carriers": CARRIERS,
           "pairs": ["|".join(p) for p in PAIRS], "runs": {}}
    for label, repo, rev in RUNS:
        tok, model = load(rev, repo)
        rec = {}
        for a, b in PAIRS:
            ta, tb = tok(a)["input_ids"], tok(b)["input_ids"]
            assert len(ta) == 1 and len(tb) == 1, "reference endpoints must be single tokens"
            for c in CARRIERS:
                cid = tok(c)["input_ids"]
                ia = torch.tensor([cid + ta], device="cuda")
                ib = torch.tensor([cid + tb], device="cuda")
                assert (ia[0, :-1] == ib[0, :-1]).all(), "pair must differ only in final token"
                r = run_pair_abs(model, ia, ib, valid)
                rec[f"{a}|{b}@@{c}"] = r
                print(f"{label:15s} {a}/{b:8s} {c:20s} w={r['w']:.3f} E={r['edge_drift']:.3f} "
                      f"M(1)={r['out_jsd']:.3f} err={r['err']:.1e}", flush=True)
        out["runs"][label] = dict(model=repo, revision=rev, curves=rec)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(RESULTS, "reference_house.json"), "w"), indent=2)
    print("saved results/reference_house.json")
