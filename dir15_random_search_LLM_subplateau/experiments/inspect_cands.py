"""S4 + S5 — continuations from the A / C / B regions, and nearest natural activations.

S4 (continuations). For a frozen inspection set (the 3 top-scoring candidates plus 3 drawn at
random with seed 7 from the remaining candidates) we patch three representative alphas — the centre
of the A run, the centre of the C run, the centre of the B run — and decode 20 tokens greedily
(primary view) and with temperature 0.8 under 3 fixed seeds (secondary view). Unpatched
continuations of both endpoint contexts are the control. Reproducibility across the C region is
checked by decoding greedily at the first, middle and last alpha of the C run.

S5 (nearest natural activations). For EVERY candidate path we take the C-region centre activation
and retrieve its nearest neighbours (cosine) among the 2000 held-out reference contexts' activations
at the same block, disjoint from both pair banks. Controls: the A-region and B-region centre
activations of the same path, and the reference contexts themselves used as queries (leave-one-out).
Reported per query: cosine distance to the nearest natural activation, and the fraction of the top-10
neighbours whose own unpatched top-1 next token equals the query's top-1 token.
"""
import json
import os
import pickle

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from common import ALPHAS, LAYERS, MODEL, RESULTS, REVISION, Runner, slerp_rescale

TOPK = 10
N_NEW = 20
TEMP = 0.8
SEEDS = (0, 1, 2)


def region_centres(r):
    """Representative alpha indices for the A, C and B regions of a candidate path."""
    kin, kout = r["k_in"], r["k_out"]
    return {"A": kin // 2, "C": (kin + kout) // 2, "B": (kout + len(ALPHAS)) // 2}


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    man = json.load(open(os.path.join(RESULTS, "manifest.json")))
    acts = np.load(os.path.join(RESULTS, "acts.npy"))
    ctx = np.load(os.path.join(RESULTS, "ctx.npz"))
    W, top1 = ctx["windows"], ctx["top1"]
    pri = pickle.load(open(os.path.join(RESULTS, "screen_primary.pkl"), "rb"))
    rows = pri["rows"]
    cand = [i for i, r in enumerate(rows) if r["is_candidate"]]
    assert cand, "no candidates to inspect"

    run = Runner()
    dev = run.device
    ts = torch.tensor(ALPHAS, dtype=torch.float32, device=dev)
    ref_idx = np.array(man["reference_idx"])
    out = {"n_candidates": len(cand), "topk": TOPK, "n_reference": len(ref_idx)}

    # ------------------------------------------------------------------ S5: neighbour retrieval
    R = {l: torch.from_numpy(acts[ref_idx, LAYERS.index(l)]).to(dev) for l in LAYERS}
    Rn = {l: v / v.norm(dim=-1, keepdim=True) for l, v in R.items()}
    ref_top1 = top1[ref_idx]

    def neighbours(h, l, exclude=-1):
        u = (h / h.norm()).to(dev)
        sim = Rn[l] @ u
        if exclude >= 0:
            sim[exclude] = -2
        v, i = torch.topk(sim, TOPK)
        return v.cpu().numpy(), i.cpu().numpy()

    nb = {k: {"cos_dist_nn": [], "frac_same_top1": []} for k in ("A", "C", "B", "natural")}
    nb_detail = []
    for ci in cand:
        r = rows[ci]
        li = LAYERS.index(r["layer"])
        hA = torch.from_numpy(acts[r["ia"], li]).to(dev)
        hB = torch.from_numpy(acts[r["ib"], li]).to(dev)
        H = slerp_rescale(hA, hB, ts)
        for reg, k in region_centres(r).items():
            tokid = {"A": r["A"], "C": r["C"], "B": r["B"]}[reg]
            v, i = neighbours(H[k], r["layer"])
            nb[reg]["cos_dist_nn"].append(float(1 - v[0]))
            nb[reg]["frac_same_top1"].append(float(np.mean(ref_top1[i] == tokid)))
            if len(nb_detail) < 6 * 3 and ci in cand[:6]:
                nb_detail.append({
                    "path": ci, "layer": r["layer"], "region": reg,
                    "query_token": tok.decode([tokid]),
                    "cos_dist_nn": float(1 - v[0]),
                    "neighbours": [{"cos_dist": float(1 - vv),
                                    "pred": tok.decode([int(ref_top1[ii])]),
                                    "text": tok.decode(W[ref_idx[ii]])[-70:]}
                                   for vv, ii in zip(v[:3], i[:3])]})
    rng = np.random.default_rng(11)
    for j in rng.choice(len(ref_idx), size=min(400, len(ref_idx)), replace=False):
        l = LAYERS[int(rng.integers(len(LAYERS)))]
        v, i = neighbours(R[l][j], l, exclude=int(j))
        nb["natural"]["cos_dist_nn"].append(float(1 - v[0]))
        nb["natural"]["frac_same_top1"].append(float(np.mean(ref_top1[i] == ref_top1[j])))
    out["neighbours"] = {k: {"n": len(v["cos_dist_nn"]),
                             "cos_dist_nn_mean": float(np.mean(v["cos_dist_nn"])),
                             "cos_dist_nn_sd": float(np.std(v["cos_dist_nn"])),
                             "frac_same_top1_mean": float(np.mean(v["frac_same_top1"])),
                             "frac_same_top1_sd": float(np.std(v["frac_same_top1"])),
                             "raw_cos": v["cos_dist_nn"], "raw_frac": v["frac_same_top1"]}
                         for k, v in nb.items()}
    out["neighbour_examples"] = nb_detail

    # ------------------------------------------------------------------- S4: continuations
    order = sorted(cand, key=lambda i: -rows[i]["score"])
    rest = [i for i in cand if i not in order[:3]]
    rng2 = np.random.default_rng(7)
    pick = order[:3] + list(rng2.choice(rest, size=min(3, len(rest)), replace=False))
    cont = []
    for ci in map(int, pick):
        r = rows[ci]
        li = LAYERS.index(r["layer"])
        hA = torch.from_numpy(acts[r["ia"], li]).to(dev)
        hB = torch.from_numpy(acts[r["ib"], li]).to(dev)
        H = slerp_rescale(hA, hB, ts)
        cond_idx = [r["ia"], r["ib"]][r["cond"]]
        ids = torch.from_numpy(W[cond_idx][None]).long()
        e = {"path": ci, "layer": r["layer"], "cond": "A" if r["cond"] == 0 else "B",
             "score": r["score"], "A": tok.decode([r["A"]]), "B": tok.decode([r["B"]]),
             "C": tok.decode([r["C"]]), "clean": bool(r["clean"]),
             "ctx_A": tok.decode(W[r["ia"]]), "ctx_B": tok.decode(W[r["ib"]]),
             "cond_ctx_tail": tok.decode(W[cond_idx])[-60:], "regions": {}}
        for reg, k in region_centres(r).items():
            g = tok.decode(run.generate(ids, patch=(r["layer"], H[k][None]), n_new=N_NEW))
            samp = [tok.decode(run.generate(ids, patch=(r["layer"], H[k][None]), n_new=N_NEW,
                                            temperature=TEMP, seed=s)) for s in SEEDS]
            e["regions"][reg] = {"alpha": float(ALPHAS[k]), "greedy": g, "sampled": samp}
        # reproducibility across the C run: greedy at first / middle / last C alpha
        ks = [r["k_in"], (r["k_in"] + r["k_out"]) // 2, r["k_out"]]
        gids = [run.generate(ids, patch=(r["layer"], H[k][None]), n_new=N_NEW) for k in ks]
        e["C_run_greedy"] = [{"alpha": float(ALPHAS[k]), "greedy": tok.decode(g), "ids": g}
                             for k, g in zip(ks, gids)]
        pref = 0
        while pref < N_NEW and len({g[pref] for g in gids}) == 1:
            pref += 1
        e["C_common_prefix_tokens"] = pref
        e["C_first_token_stable"] = pref >= 1
        # unpatched controls, both endpoint contexts
        e["control_unpatched"] = {
            "ctx_A": tok.decode(run.generate(torch.from_numpy(W[r["ia"]][None]).long(),
                                             n_new=N_NEW)),
            "ctx_B": tok.decode(run.generate(torch.from_numpy(W[r["ib"]][None]).long(),
                                             n_new=N_NEW))}
        # the same activation under the OTHER endpoint's context
        other = torch.from_numpy(W[[r["ia"], r["ib"]][1 - r["cond"]]][None]).long()
        kc = region_centres(r)["C"]
        e["C_under_other_context"] = tok.decode(
            run.generate(other, patch=(r["layer"], H[kc][None]), n_new=N_NEW))
        cont.append(e)
    out["continuations"] = cont

    n_stab = sum(c["C_first_token_stable"] for c in cont)
    out["C_first_token_stable_frac"] = n_stab / len(cont)
    with open(os.path.join(RESULTS, "inspection.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("n_candidates", "C_first_token_stable_frac")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("raw")}
                      for k, v in out["neighbours"].items()}, indent=1))
    for c in cont[:2]:
        print("--", c["C"], "|", {k: v["greedy"][:60] for k, v in c["regions"].items()})


if __name__ == "__main__":
    main()
