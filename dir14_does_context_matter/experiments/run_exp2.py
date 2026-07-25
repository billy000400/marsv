"""S4 — fixed endpoint transition, four preregistered context classes.

Primary (preregistered) arm: `big -> large` under
    none / random / unrelated / relevant  prefixes (4 prefixes per non-empty class).
Secondary arm added because the primary pair shows no plateau in the reference context:
`big -> in` under the same 13 contexts. A context effect on plateaus can only be measured where
a plateau exists, so the secondary arm is the positive control for the same question.

Endpoint token ids are asserted identical across all contexts (manifest.py already checks this).
"""
import json
import os

import numpy as np
import torch

from sweep import Assay, RES, save, summarize_all, sweep, worst_checks

N_STEPS = 50


def main():
    with open(os.path.join(RES, "manifest.json")) as f:
        man = json.load(f)
    ts = np.linspace(0.0, 1.0, N_STEPS)
    eps = man["endpoints"]

    conds = []
    for c in man["contexts"]:
        conds.append({"label": f"{c['cid']}|big->large", "ids_A": c["ids_big"],
                      "ids_B": c["ids_large"]})
        conds.append({"label": f"{c['cid']}|big->in", "ids_A": c["prefix_ids"] + [eps["big"]["id"]],
                      "ids_B": c["prefix_ids"] + [eps["in"]["id"]]})
    for c in conds:                      # endpoint ids identical across contexts
        assert c["ids_A"][-1] == eps["big"]["id"]
        assert c["ids_B"][-1] in (eps["large"]["id"], eps["in"]["id"])

    model = Assay(man["model"], device="cuda", mem_frac=0.225, threads=2,
                  revision=man["revision"])
    curves, checks = sweep(model, conds, ts)
    summaries = summarize_all(curves, ts)

    extra = {"ts": ts.tolist(), "n_layer": model.n_layer, "model": man["model"],
             "revision": man["revision"], "torch": torch.__version__,
             "dtype": "float32", "device": "cuda",
             "worst_checks": worst_checks(checks),
             "contexts": {c["cid"]: {"class": c["class"], "prefix": c["prefix"],
                                     "prefix_ids": c["prefix_ids"]} for c in man["contexts"]}}
    save("exp2", curves, checks, summaries, extra)

    worst = extra["worst_checks"]
    print("worst endpoint check over all conditions/layers:",
          {k: f"{max(w[k] for w in worst.values()):.2e}" for k in next(iter(worst.values()))})
    for c in man["contexts"]:
        for pair in ("big->large", "big->in"):
            s = summaries[f"{c['cid']}|{pair}|L0|logits"]
            print(f"{c['cid']:<12} {c['class']:<10} {pair:<11} L0 logits: "
                  f"w={s['width']:.3f} loc={s['location']:.3f} plateau={int(s['plateau'])}")


if __name__ == "__main__":
    main()
