"""S2 step 1: in-distribution endpoint candidates.

Keep lowercase alphabetic word-start tokens that are in the trained 1.4B model's top-256 eligible
word continuations of ALL THREE carrier contexts. This is a model-prior filter only -- it never looks
at a plateau curve, so it is safe to run before the assay.
"""
import json
import os

import torch

from common import CONTEXTS, FINAL, RESULTS, eligible_word_tokens, load

TOPK = 256      # prespecified strict in-distribution filter
TOPK_BANK = 512  # relaxed filter used if TOPK cannot supply 75 unique-endpoint pairs
POOL_TOPK = 1024  # counting superset: counted once so no second pass over the corpus is needed

if __name__ == "__main__":
    tok, m = load(FINAL)
    V = m.get_output_embeddings().weight.shape[0]
    elig = eligible_word_tokens(tok, len(tok))
    elig_t = torch.tensor(elig, device="cuda")
    print(f"vocab_out={V} tokenizer={len(tok)} eligible_word_tokens={len(elig)}")

    lps = {}
    with torch.inference_mode():
        for c in CONTEXTS:
            ids = torch.tensor([tok(c)["input_ids"]], device="cuda")
            lps[c] = m(ids).logits[0, -1].float().log_softmax(-1)

    def inter(k):
        return sorted(set.intersection(*[set(elig_t[lps[c][elig_t].topk(k).indices].tolist())
                                         for c in CONTEXTS]))

    keep, bank_pool, pool = inter(TOPK), inter(TOPK_BANK), inter(POOL_TOPK)
    probs = lps
    for k, s in [(TOPK, keep), (TOPK_BANK, bank_pool), (POOL_TOPK, pool)]:
        print(f"intersection of top-{k} across {len(CONTEXTS)} contexts: {len(s)}")
    print("sample:", [tok.decode([i]) for i in keep[:40]])

    out = {
        "model": "EleutherAI/pythia-1.4b-deduped", "revision": FINAL,
        "vocab_out": V, "n_eligible_word_tokens": len(elig),
        "topk": TOPK, "topk_bank": TOPK_BANK, "pool_topk": POOL_TOPK,
        "contexts": CONTEXTS,
        "candidates": keep, "bank_pool": bank_pool, "pool": pool,
        "pool_strings": [tok.decode([i]) for i in pool],
        "pool_ctx_logprob": {c: [round(float(probs[c][i]), 4) for i in pool] for c in CONTEXTS},
        "eligible_word_tokens": elig,
    }
    with open(os.path.join(RESULTS, "endpoint_candidates.json"), "w") as f:
        json.dump(out, f)
    print("saved")
