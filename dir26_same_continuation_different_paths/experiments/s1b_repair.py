"""S1b: the one minimal wording repair PLAN.md authorizes for pair 3, rechecked."""
import json
import os

from common import MODEL, MODEL_KEY, RESULTS, load
from s1_completions import screen
REPAIR = ("p3_relation_repaired", "Different relation, same answer (repaired wording)",
          "The capital city of France is", "The largest city in France is", "Paris")


def main():
    tok, m = load()
    key, label, pa, pb, intended = REPAIR
    rec = {"key": key, "label": label, "intended": intended, "prompts": {}}
    for side, p in (("A", pa), ("B", pb)):
        rec["prompts"][side] = screen(tok, m, p, intended)
    a, b = rec["prompts"]["A"], rec["prompts"]["B"]
    rec["completion_pass"] = (intended.lower() in a["continuation"].lower()
                              and intended.lower() in b["continuation"].lower())
    rec["length_match"] = a["n_tokens"] == b["n_tokens"]
    print(json.dumps(rec, indent=2))
    with open(os.path.join(RESULTS, f"s1b_repair_{MODEL_KEY}.json"), "w") as f:
        json.dump({"model": MODEL, "pair": rec}, f, indent=2)


if __name__ == "__main__":
    main()
