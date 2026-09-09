"""S1: greedy continuations, top-1 next-token prediction, and tokenized lengths for the 8 prompts.

The top-1 next token ("what the model is actually predicting") was added in response to
human_feedback.txt. Run once per model: DIR26_MODEL=gpt2 | qwen.
"""
import json
import os

import torch

from common import DEV, MODEL, MODEL_KEY, PAIRS, RESULTS, load

MAX_NEW = 6
TOPK = 5


def screen(tok, m, p, intended):
    """Greedy continuation + the model's actual next-token distribution at the prompt's end."""
    ids = tok(p)["input_ids"]
    x = torch.tensor([ids], device=DEV)
    with torch.no_grad():
        logits = m(x, use_cache=False).logits[0, -1, :].float()
        gen = m.generate(x, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    probs = logits.softmax(-1)
    pv, pi = probs.topk(TOPK)
    return {"prompt": p, "n_tokens": len(ids),
            "continuation": tok.decode(gen[0, len(ids):]),
            "top1_token": tok.decode(int(pi[0])),
            "top1_prob": round(float(pv[0]), 4),
            "top5": [[tok.decode(int(i)), round(float(v), 4)] for v, i in zip(pv, pi)]}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tok, m = load()
    out = []
    for key, label, pa, pb, intended in PAIRS:
        rec = {"key": key, "label": label, "intended": intended, "prompts": {}}
        for side, p in (("A", pa), ("B", pb)):
            rec["prompts"][side] = screen(tok, m, p, intended)
        a, b = rec["prompts"]["A"], rec["prompts"]["B"]
        rec["completion_pass"] = (intended.lower() in a["continuation"].lower()
                                  and intended.lower() in b["continuation"].lower())
        rec["length_match"] = a["n_tokens"] == b["n_tokens"]
        out.append(rec)
        print(json.dumps(rec, indent=2))
    with open(os.path.join(RESULTS, f"s1_completions_{MODEL_KEY}.json"), "w") as f:
        json.dump({"model": MODEL, "pairs": out}, f, indent=2)


if __name__ == "__main__":
    main()
