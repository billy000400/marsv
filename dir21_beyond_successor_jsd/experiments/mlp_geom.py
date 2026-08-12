"""How much of the post-block-0 state IS the MLP's contribution?

The transplant moves the recipient's width to the donor's almost completely. The deflationary reading
is that m_u simply dominates the post-block-0 state x_u = (embedding + attention) + m_u, so swapping
m_u swaps the whole state and the result is near-tautological. This measures the sizes involved and
where the hybrid state rest_r + m_d actually sits between the two tokens' own states.
Writes results/mlp_geom.json.
"""
import json

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION, FRAMES, Patcher
from common import D18, RESULTS
from mlp_read import MLPOut, states

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    toks12 = list(json.load(open(f"{RESULTS}/mode_split.json"))["tokens"].keys())

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)
    mlp = MLPOut(model)
    pre = tok(FRAMES[0], return_tensors="pt").input_ids.cuda()

    m, x = [], []
    for s in toks12:
        mu, xu, _ = states(model, mlp, patcher, pre, ids_by_str[s])
        m.append(mu.cpu().numpy())
        x.append(xu.cpu().numpy())
    m, x = np.stack(m), np.stack(x)
    rest = x - m

    nm, nx, nr = np.linalg.norm(m, axis=1), np.linalg.norm(x, axis=1), np.linalg.norm(rest, axis=1)
    # how much of the ACROSS-TOKEN variation of x comes from m rather than from the rest?
    vm = np.linalg.norm(m - m.mean(0), axis=1).mean()
    vr = np.linalg.norm(rest - rest.mean(0), axis=1).mean()
    vx = np.linalg.norm(x - x.mean(0), axis=1).mean()

    # where does the hybrid state sit between the donor's and the recipient's own states?
    fd, fr = [], []
    for i in range(len(toks12)):
        for j in range(len(toks12)):
            if i == j:
                continue
            h = rest[i] + m[j]                    # recipient i, donor j
            fd.append(np.linalg.norm(h - x[j]) / np.linalg.norm(x[i] - x[j]))
            fr.append(np.linalg.norm(h - x[i]) / np.linalg.norm(x[i] - x[j]))
    res = dict(tokens=toks12,
               norm_m=float(np.median(nm)), norm_x=float(np.median(nx)),
               norm_rest=float(np.median(nr)), ratio_m_over_x=float(np.median(nm / nx)),
               spread_m=float(vm), spread_rest=float(vr), spread_x=float(vx),
               spread_share_m=float(vm / (vm + vr)),
               hybrid_dist_to_donor=float(np.median(fd)),
               hybrid_dist_to_recipient=float(np.median(fr)))
    for k, v in res.items():
        if k != "tokens":
            print(f"{k:26s} {v:.4f}")
    json.dump(res, open(f"{RESULTS}/mlp_geom.json", "w"), indent=1)
    print("wrote results/mlp_geom.json")


if __name__ == "__main__":
    main()
