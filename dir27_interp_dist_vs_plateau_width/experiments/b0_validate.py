"""S2 of the block-0 rerun: validate the block-0 resid_post patch. Usage: DIR27_SITE=block0_resid_post b0_validate.py KEY

KEY = gpt2-large (common.Runner, the 20 s1_sanity tokens) or a gen_common model key (its 32 seed-0 tokens).
Checks, all on full-prompt (no prefix cache) runs:
 1. the input of block 1 at the last position equals r(t), and at prefix positions equals the unpatched A run;
 2. the raw mid/final readout vectors of the patched runs at t=0 and t=1 equal independent unpatched A and B runs.
For gen models the sweep's cached last-position path is also compared with the full-prompt patched path.
"""
import json
import os
import sys

import numpy as np
import torch

import common
from common import RESULTS, SITE, TS

assert SITE == "block0_resid_post"
key = sys.argv[1]
if key == "gpt2-large":
    from s1_sanity import WORDS
    tok, m, dev = common.load()
    r = common.Runner(tok, m, dev)
    blocks = m.transformer.h
    ids_b = [tok.encode(w)[0] for w in WORDS]
    emb = lambda ids: r.wte[torch.as_tensor(ids, device=dev)]
    unpatched = lambda e: r.run(e)
    patched_full = r.run_patched
    site = "GPT2Model.h[0] output"
else:
    import gen_common
    tok, m = gen_common.load(key)
    r = gen_common.Runner(tok, m)
    blocks = gen_common.blocks_of(r.base)
    ids_b = np.random.default_rng(0).choice(len(tok), 32, replace=False).tolist()
    emb = r.rows
    unpatched = r.run_full
    patched_full = lambda x: r.run_patched(x, full=True)
    site = r.site_module

seen = {}
blocks[1].register_forward_pre_hook(
    lambda mod, args, kwargs: seen.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).float().clone()),
    with_kwargs=True)

rel = lambda x, y: float(((x - y).norm(dim=-1) / y.norm(dim=-1)).max())
t = torch.tensor(TS, device="cuda", dtype=torch.float32).view(-1, 1)
out = dict(model=key, intervention_site=SITE, hook_module=site, n_tokens=len(ids_b), token_ids=ids_b)

# independent unpatched endpoint runs (clone: captures are overwritten by the next forward)
mid_a, fin_a = [x.clone() for x in unpatched(emb([r.id_a]))]
ra, pre_a = r.cap["resid0"][0].clone(), seen["x"][0, :-1].clone()
mid_b, fin_b = [x.clone() for x in unpatched(emb(ids_b))]
rb = r.cap["resid0"].clone()
out["resid0_dtype"] = r.cap.get("resid0_dtype", "torch.float32")

worst = dict(inj=0.0, prefix=0.0, mid0=0.0, fin0=0.0, mid1=0.0, fin1=0.0, mid0_abs=0.0, fin0_abs=0.0,
             mid1_abs=0.0, fin1_abs=0.0)
for j in range(len(ids_b)):
    rt = (1 - t) * ra + t * rb[j]
    mid, fin = patched_full(rt)
    x = seen["x"]
    worst["inj"] = max(worst["inj"], float((x[:, -1] - rt).abs().max()))
    worst["prefix"] = max(worst["prefix"], float((x[:, :-1] - pre_a).abs().max()))
    for name, got, ref in (("mid0", mid[0], mid_a[0]), ("fin0", fin[0], fin_a[0]),
                           ("mid1", mid[-1], mid_b[j]), ("fin1", fin[-1], fin_b[j])):
        worst[name] = max(worst[name], rel(got, ref))
        worst[name + "_abs"] = max(worst[name + "_abs"], float((got - ref).abs().max()))
out.update(injected_vs_r_t_max_abs=worst["inj"], prefix_vs_unpatched_A_max_abs=worst["prefix"],
           readout_t0_vs_unpatched_A_max_rel_l2=[worst["mid0"], worst["fin0"]],
           readout_t1_vs_unpatched_B_max_rel_l2=[worst["mid1"], worst["fin1"]],
           readout_t0_vs_unpatched_A_max_abs=[worst["mid0_abs"], worst["fin0_abs"]],
           readout_t1_vs_unpatched_B_max_abs=[worst["mid1_abs"], worst["fin1_abs"]],
           readout_norm_A=[float(mid_a.norm()), float(fin_a.norm())],
           D_min_max=[float((rb - ra).norm(dim=-1).min()), float((rb - ra).norm(dim=-1).max())])
os.makedirs(os.path.join(RESULTS, "validate"), exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "validate", f"{key}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
