"""Confirmatory replication bank (bank 2), frozen before it is run.

Bank 1 (manifest.py) has 4 prefixes per class, which left `relevant` vs `unrelated` undecided
(exact rank-sum p = 0.49). Bank 2 adds 8 NEW prefixes per class — no string is reused from bank 1 —
so the class ordering can be re-tested with n = 8 vs 8. Everything else is unchanged: same endpoints,
same 3-token prefix length, same 50-step grid, same summary thresholds.

Same selection rule as bank 1: take the first eight candidates in the listed order that tokenize to
exactly 3 tokens. Random prefixes use numpy seed 1 (bank 1 used seed 0) so the two banks are
independent draws.

Writes results/manifest_bank2.json, then run with run_bank2.py.
"""
import json
import os

import numpy as np
from transformers import GPT2TokenizerFast

from manifest import ENDPOINTS, MODEL, PREFIX_LEN, REVISION

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEED = 1
N_PER_CLASS = 8

RELEVANT_CANDIDATES = [          # copular contexts where `big`/`large` is a natural continuation
    "The dog was", "The box was", "The city was", "The truck was", "The stone was",
    "The crowd was", "His car was", "The window was", "The garden was", "The bird was",
    "The screen was", "The wave was",
]
UNRELATED_CANDIDATES = [         # natural English that does not set up a size adjective
    "He kindly offered", "They finally agreed", "The concert ended", "She wrote back",
    "It rained yesterday", "The engine stalled", "He turned around", "They danced together",
    "We must decide", "Birds migrate south", "She apologized twice", "The alarm rang",
]


def build(tok):
    def ids(s):
        return tok(s)["input_ids"]

    ep = {name: {"string": s, "id": ids(s)[0]} for name, s in ENDPOINTS.items()}
    for name, s in ENDPOINTS.items():
        assert len(ids(s)) == 1, f"endpoint {s!r} is not one token"

    def pick(cands, label):
        out = []
        for c in cands:
            i = ids(c)
            if len(i) == PREFIX_LEN:
                out.append({"class": label, "prefix": c, "prefix_ids": i,
                            "prefix_tokens": [tok.decode([t]) for t in i]})
            if len(out) == N_PER_CLASS:
                break
        assert len(out) == N_PER_CLASS, f"{label}: only {len(out)} of length {PREFIX_LEN}"
        return out

    rng = np.random.default_rng(SEED)
    rand = []
    while len(rand) < N_PER_CLASS:
        toks = [int(t) for t in rng.integers(0, tok.vocab_size, size=PREFIX_LEN)]
        if tok.eos_token_id in toks:
            continue
        rand.append({"class": "random", "prefix": tok.decode(toks), "prefix_ids": toks,
                     "prefix_tokens": [tok.decode([t]) for t in toks]})

    contexts = rand + pick(UNRELATED_CANDIDATES, "unrelated") + pick(RELEVANT_CANDIDATES, "relevant")
    for k, c in enumerate(contexts):
        c["cid"] = f"b2{c['class']}{k}"
        for e in ("big", "large", "in"):
            c[f"ids_{e}"] = c["prefix_ids"] + [ep[e]["id"]]
    return {"model": MODEL, "revision": REVISION, "seed": SEED, "bank": 2,
            "endpoints": ep, "contexts": contexts}


if __name__ == "__main__":
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    man = build(tok)
    # no string may be reused from bank 1
    with open(os.path.join(RES, "manifest.json")) as f:
        b1 = {c["prefix"] for c in json.load(f)["contexts"]}
    assert not (b1 & {c["prefix"] for c in man["contexts"]}), "bank 2 reuses a bank-1 prefix"
    with open(os.path.join(RES, "manifest_bank2.json"), "w") as f:
        json.dump(man, f, indent=2)
    for c in man["contexts"]:
        print(f"  {c['cid']:<16} {c['class']:<10} {c['prefix']!r:<28} ids={c['prefix_ids']}")
    print("wrote", os.path.join(RES, "manifest_bank2.json"))
