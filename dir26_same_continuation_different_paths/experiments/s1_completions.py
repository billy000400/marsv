"""S1: greedy continuations + tokenized lengths for the 8 exact prompts."""
import json
import os

import torch

from common import DEV, PAIRS, RESULTS, load

MAX_NEW = 6


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tok, m = load()
    out = []
    for key, label, pa, pb, intended in PAIRS:
        rec = {"key": key, "label": label, "intended": intended, "prompts": {}}
        for side, p in (("A", pa), ("B", pb)):
            ids = tok(p)["input_ids"]
            with torch.no_grad():
                gen = m.generate(torch.tensor([ids], device=DEV),
                                 max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            cont = tok.decode(gen[0, len(ids):])
            rec["prompts"][side] = {"prompt": p, "n_tokens": len(ids),
                                    "continuation": cont}
        a, b = rec["prompts"]["A"], rec["prompts"]["B"]
        rec["completion_pass"] = (intended.lower() in a["continuation"].lower()
                                  and intended.lower() in b["continuation"].lower())
        rec["length_match"] = a["n_tokens"] == b["n_tokens"]
        out.append(rec)
        print(json.dumps(rec, indent=2))
    with open(os.path.join(RESULTS, "s1_completions.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
