"""S1b: the one minimal wording repair PLAN.md authorizes for pair 3, rechecked."""
import json
import os

import torch

from common import DEV, RESULTS, load

MAX_NEW = 6
REPAIR = ("p3_relation_repaired", "Different relation, same answer (repaired wording)",
          "The capital city of France is", "The largest city in France is", "Paris")


def main():
    tok, m = load()
    key, label, pa, pb, intended = REPAIR
    rec = {"key": key, "label": label, "intended": intended, "prompts": {}}
    for side, p in (("A", pa), ("B", pb)):
        ids = tok(p)["input_ids"]
        with torch.no_grad():
            gen = m.generate(torch.tensor([ids], device=DEV), max_new_tokens=MAX_NEW,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        rec["prompts"][side] = {"prompt": p, "n_tokens": len(ids),
                                "continuation": tok.decode(gen[0, len(ids):])}
    a, b = rec["prompts"]["A"], rec["prompts"]["B"]
    rec["completion_pass"] = (intended.lower() in a["continuation"].lower()
                              and intended.lower() in b["continuation"].lower())
    rec["length_match"] = a["n_tokens"] == b["n_tokens"]
    print(json.dumps(rec, indent=2))
    with open(os.path.join(RESULTS, "s1b_repair.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
