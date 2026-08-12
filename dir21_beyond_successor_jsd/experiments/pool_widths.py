"""Measure crossing width and edge drift for 127 further endpoint tokens, to enlarge the pool to 250.

The standing negative -- no readout finds the part of the crossing width that curve shape does not
already explain -- was fitted on 123 tokens, and the learning curve (pattern 47) priced exactly what
that costs: the permutation null shrinks as 0.572/sqrt(n_test), so a true rho of +0.15 needs a test
half of about 58 tokens, which 123 tokens cannot supply at any training size. This run buys the
tokens.

The 123 came from dir18's endpoint candidate set, which is the top of a ranking of pool tokens by mean
next-token log-probability under the same three frames. The new tokens are simply the next 127 in that
same ranking, so the sample is extended the way it was drawn rather than by a new criterion. The six
anchors, three frames, block-0 site and the SLERP protocol are unchanged, so old and new measurements
are the same measurement.

Writes results/pool_widths.json (per token: the 18 curve statistics, one per frame x anchor).
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from basin_probe import MODEL, REVISION, FRAMES, Patcher
from common import D18, RESULTS
from envwidth import token_widths

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.225)

N_NEW = 127
CHUNK = 16
OUT = os.path.join(RESULTS, "pool_widths.json")


def selection():
    """The 6 anchors, the 123 measured tokens, and the next N_NEW tokens in dir18's own ranking."""
    cand = json.load(open(f"{D18}/endpoint_candidates.json"))
    man = json.load(open(f"{D18}/pair_manifest_large.json"))["pairs"]
    used = {}
    for p in man:
        used[p["a_str"]] = p["a"]
        used[p["b_str"]] = p["b_tok"]
    pool = cand["pool"]
    free = [i for i in sorted(pool) if i not in set(used.values())]
    anchors = free[:: max(1, len(free) // 6)][:6]

    lp = np.array([cand["pool_ctx_logprob"][f] for f in cand["pool_ctx_logprob"]]).mean(0)
    order = [pool[i] for i in np.argsort(-lp)]
    rank = {t: r for r, t in enumerate(order)}
    skip = set(used.values()) | set(anchors) | set(cand["candidates"])
    new = [t for t in order if t not in skip][:N_NEW]
    return anchors, used, new, {int(t): float(lp[pool.index(t)]) for t in order}, rank


def main():
    anchors, used, new_ids, logprob, rank = selection()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    new = {tok.decode([i]): i for i in new_ids}
    assert len(new) == len(new_ids), "decoded strings collide"
    for s, i in new.items():
        assert tok(s).input_ids == [i], (s, i)
    assert not (set(new) & set(used)), "new token already measured"

    res = json.load(open(OUT)) if os.path.exists(OUT) else {"tokens": {}}
    res.update(model=MODEL, revision=REVISION, frames=FRAMES, site="block0",
               anchor_ids=anchors, anchors=[tok.decode([a]) for a in anchors],
               n_requested=len(new),
               rank_used=[rank[i] for i in sorted(used.values())],
               rank_new={s: rank[i] for s, i in new.items()},
               logprob_used=[logprob[i] for i in sorted(used.values())],
               logprob_new={s: logprob[i] for s, i in new.items()})

    todo = {s: i for s, i in new.items() if s not in res["tokens"]}
    print(f"{len(res['tokens'])} already measured, {len(todo)} to go "
          f"(ranks {min(res['rank_new'].values())}-{max(res['rank_new'].values())})", flush=True)
    if not todo:
        json.dump(res, open(OUT, "w"), indent=1)
        return

    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                               torch_dtype=torch.float32).eval().cuda()
    patcher = Patcher(model, layer=0)
    items = sorted(todo.items())
    for s in range(0, len(items), CHUNK):
        chunk = dict(items[s:s + CHUNK])
        raw = token_widths(model, patcher, tok, chunk, anchors, FRAMES, log_every=0)
        for name, r in raw.items():
            res["tokens"][name] = {k: [float(x) for x in v] for k, v in r.items()}
            res["tokens"][name]["token_id"] = chunk[name]
        json.dump(res, open(OUT, "w"), indent=1)
        done = len(res["tokens"])
        E = np.array([np.median(res["tokens"][n]["edge"]) for n in res["tokens"]])
        W = np.array([np.median(res["tokens"][n]["w_env"]) for n in res["tokens"]])
        print(f"{done}/{len(new)} tokens; median edge {np.median(E):.3f}, "
              f"median w_env {np.median(W):.3f}", flush=True)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
