"""Preregistered prompt manifest for both experiments (frozen BEFORE any curve was inspected).

Experiment 1 — fixed context `The house was`, endpoint pairs (big, large) and (big, in).
Experiment 2 — fixed endpoint pair (big, large) under four context classes:
    none      : the endpoint token alone (no prefix)
    random    : a length-matched sequence of GPT-2 tokens, seed 0
    unrelated : natural English prefix that does not set up a size adjective
    relevant  : natural English prefix that makes `big`/`large` a natural continuation

Selection rule (fixed in advance): for each non-random class take the first four candidates in the
listed order whose prefix tokenizes to exactly PREFIX_LEN = 3 tokens (matching `The house was`).
Random prefixes are drawn with numpy seed 0 from the non-special GPT-2 vocabulary, also 3 tokens.

Writes results/manifest.json with every string, token id, decoded token, and check outcome.
"""
import json
import os

import numpy as np
from transformers import GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "manifest.json")

MODEL = "gpt2-large"
REVISION = "main"
PREFIX_LEN = 3
SEED = 0
N_PER_CLASS = 4
N_STEPS = 50

REF_CONTEXT = "The house was"
ENDPOINTS = {"big": " big", "large": " large", "in": " in"}
EXP1_PAIRS = [("big", "large"), ("big", "in")]

RELEVANT_CANDIDATES = [
    "The house was",      # Matthew's reference context
    "The room was",
    "The rock was",
    "Her bag was",
    "The tree was",
]
UNRELATED_CANDIDATES = [
    "She quickly walked",
    "He apologized for",
    "Water boils at",
    "The meeting began",
    "They agreed to",
]


def build(tok):
    def ids(s):
        return tok(s)["input_ids"]

    ep = {}
    for name, s in ENDPOINTS.items():
        i = ids(s)
        assert len(i) == 1, f"endpoint {s!r} is not one token: {i}"
        ep[name] = {"string": s, "id": i[0], "decoded": tok.decode(i)}

    def pick(cands, label):
        out = []
        for c in cands:
            i = ids(c)
            if len(i) == PREFIX_LEN:
                out.append({"class": label, "prefix": c, "prefix_ids": i,
                            "prefix_tokens": [tok.decode([t]) for t in i]})
            if len(out) == N_PER_CLASS:
                break
        assert len(out) == N_PER_CLASS, f"{label}: only {len(out)} candidates of length {PREFIX_LEN}"
        return out

    rng = np.random.default_rng(SEED)
    special = {tok.eos_token_id}
    rand = []
    while len(rand) < N_PER_CLASS:
        toks = [int(t) for t in rng.integers(0, tok.vocab_size, size=PREFIX_LEN)]
        if any(t in special for t in toks):
            continue
        rand.append({"class": "random", "prefix": tok.decode(toks), "prefix_ids": toks,
                     "prefix_tokens": [tok.decode([t]) for t in toks]})

    contexts = ([{"class": "none", "prefix": "", "prefix_ids": [], "prefix_tokens": []}]
                + rand + pick(UNRELATED_CANDIDATES, "unrelated")
                + pick(RELEVANT_CANDIDATES, "relevant"))
    for k, c in enumerate(contexts):
        c["cid"] = f"{c['class']}{'' if c['class'] == 'none' else k}"
        # endpoint prompts: prefix ids + the single endpoint token id (endpoint id never changes)
        for e in ("big", "large"):
            c[f"ids_{e}"] = c["prefix_ids"] + [ep[e]["id"]]
        assert c["ids_big"][:-1] == c["ids_large"][:-1]
        assert c["ids_big"][-1] == ep["big"]["id"] and c["ids_large"][-1] == ep["large"]["id"]

    # Experiment 1 prompts, built the same way
    ref_ids = ids(REF_CONTEXT)
    exp1 = []
    for a, b in EXP1_PAIRS:
        exp1.append({"pair": f"{a}->{b}", "context": REF_CONTEXT,
                     "ids_A": ref_ids + [ep[a]["id"]], "ids_B": ref_ids + [ep[b]["id"]],
                     "prompt_A": REF_CONTEXT + ENDPOINTS[a], "prompt_B": REF_CONTEXT + ENDPOINTS[b]})
        assert exp1[-1]["ids_A"][:-1] == exp1[-1]["ids_B"][:-1]

    return {"model": MODEL, "revision": REVISION, "seed": SEED, "n_steps": N_STEPS,
            "prefix_len": PREFIX_LEN, "endpoints": ep, "exp1": exp1, "contexts": contexts,
            "ref_context_ids": ref_ids,
            "ref_context_tokens": [tok.decode([t]) for t in ref_ids]}


if __name__ == "__main__":
    tok = GPT2TokenizerFast.from_pretrained(MODEL, revision=REVISION)
    man = build(tok)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(man, f, indent=2)
    print("endpoints:", {k: (v["id"], repr(v["decoded"])) for k, v in man["endpoints"].items()})
    print("ref context tokens:", man["ref_context_tokens"])
    for c in man["contexts"]:
        print(f"  {c['cid']:<12} {c['class']:<10} {c['prefix']!r:<30} ids={c['prefix_ids']}")
    print("wrote", OUT)
