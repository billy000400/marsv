"""Anchor-set swap: is the anchor width a token trait, or similarity to a typical content word?

Recomputes every endpoint token's anchor width twice — against six function words, and against six
rare content words — and asks whether the two rankings of tokens agree. Writes results/swap.json.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.append("/workspace/marsv_agent_haoyang/dir18_continuation_jsd_plateau/experiments")
import curve_metrics

from anchor_width import run_pair
from basin_probe import MODEL, REVISION, FRAMES, Patcher, endpoint
from common import D18, RESULTS, load
from explore1 import cv_r2

FUNCTION_WORDS = [" he", " it", " we", " they", " them", " those", " but", " between"]
GATE = 0.2

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)


def main():
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    ids_by_str = {}
    for p in man:
        ids_by_str[p["a_str"]] = p["a"]
        ids_by_str[p["b_str"]] = p["b_tok"]
    endpoints = sorted(ids_by_str.items())
    used = set(ids_by_str.values())

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    pool = [i for i in sorted(cand["pool"]) if i not in used]
    to_str = {i: tok.convert_ids_to_tokens(i).replace("Ġ", " ") for i in pool}
    fw = [i for i in pool if to_str[i] in FUNCTION_WORDS][:6]
    # highest BPE ids among alphabetic pool tokens: GPT-NeoX merges are frequency-ordered, so a high
    # id is a rare token
    cw = [i for i in sorted(pool, reverse=True)
          if to_str[i].strip().isalpha() and i not in fw][:6]

    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model)

    prev = (json.load(open(f"{RESULTS}/swap.json"))
            if os.path.exists(f"{RESULTS}/swap.json") else {})
    arms = {"function": fw, "rare_content": cw}
    todo = {n: ids for n, ids in arms.items()
            if len(ids) != 6 or len(prev.get(f"anchors_{n}", [])) != 6}
    print("recomputing arms:", list(todo), flush=True)
    out = {name: {s: [] for s, _ in endpoints} for name in todo}
    for frame in FRAMES:
        pre = tok(frame, return_tensors="pt").input_ids.cuda()
        sets = {}
        for name, ids_list in todo.items():
            sets[name] = [endpoint(model, patcher,
                                   torch.cat([pre, torch.tensor([[a]], device=pre.device)], 1))
                          for a in ids_list]
        for k, (s, i) in enumerate(endpoints):
            ids = torch.cat([pre, torch.tensor([[i]], device=pre.device)], 1)
            x, z = endpoint(model, patcher, ids)
            for name, anc in sets.items():
                for xb, zb in anc:
                    out[name][s].append(run_pair(model, patcher, ids, x, z, xb, zb)[0])
            if k % 40 == 0:
                print(f"frame {frame!r} token {k}/{len(endpoints)}", flush=True)

    wf = ({s: float(np.nanmedian(v)) for s, v in out["function"].items()}
          if "function" in out else prev["anchor_width_function"])
    wc = ({s: float(np.nanmedian(v)) for s, v in out["rare_content"].items()}
          if "rare_content" in out else prev["anchor_width_rare_content"])
    orig = {s: float(np.nanmedian(v["w"]))
            for s, v in json.load(open(f"{RESULTS}/anchor_width.json"))["tokens"].items()}
    tokfx = json.load(open(f"{RESULTS}/explore1.json"))["token_effects"]
    a = np.array(tokfx["effect"])
    names = tokfx["tokens"]
    F = np.array([wf[s] for s in names])
    C = np.array([wc[s] for s in names])
    O = np.array([orig[s] for s in names])

    t, _, _ = load()
    m = t["out_jsd_min"] >= GATE
    y = t["w"][m]
    one = np.ones((len(y), 1))
    col = lambda d: (np.array([d[x] for x in t["a_str"]])[m]
                     + np.array([d[x] for x in t["b_str"]])[m])[:, None]

    res = dict(
        anchors_function=([to_str[i] for i in fw] if "function" in todo
                          else prev["anchors_function"]),
        anchors_rare_content=([to_str[i] for i in cw] if "rare_content" in todo
                              else prev["anchors_rare_content"]),
        valid_rate={n: float(np.mean(~np.isnan(np.array([out[n][s] for s, _ in endpoints]))))
                    for n in out},
        rho_function_vs_rare=[float(x) for x in spearmanr(F, C)[:2]],
        rho_function_vs_original=[float(x) for x in spearmanr(F, O)[:2]],
        rho_rare_vs_original=[float(x) for x in spearmanr(C, O)[:2]],
        rho_with_token_effect=dict(
            function=[float(x) for x in spearmanr(F, a)[:2]],
            rare_content=[float(x) for x in spearmanr(C, a)[:2]]),
        median_width=dict(function=float(np.median(F)), rare_content=float(np.median(C)),
                          original=float(np.median(O))),
        cv_r2=dict(function=cv_r2(np.hstack([one, col(wf)]), y)[0],
                   rare_content=cv_r2(np.hstack([one, col(wc)]), y)[0],
                   both=cv_r2(np.hstack([one, col(wf), col(wc)]), y)[0]),
        anchor_width_function=wf, anchor_width_rare_content=wc)
    json.dump(res, open(os.path.join(RESULTS, "swap.json"), "w"), indent=1)
    print("function anchors:", res["anchors_function"])
    print("rare content anchors:", res["anchors_rare_content"])
    print("rho(function, rare_content) =", np.round(res["rho_function_vs_rare"][0], 3))
    print("rho with fitted a_u:", {k: round(v[0], 3) for k, v in res["rho_with_token_effect"].items()})
    print("CV-R2:", {k: round(v, 3) for k, v in res["cv_r2"].items()})


if __name__ == "__main__":
    main()
