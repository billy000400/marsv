"""Confirmatory replication: bank-2 prefixes (8 per class), same assay as run_exp2.py."""
import json
import os

import numpy as np
import torch

from sweep import Assay, RES, save, summarize_all, sweep, worst_checks

N_STEPS = 50


def main():
    with open(os.path.join(RES, "manifest_bank2.json")) as f:
        man = json.load(f)
    ts = np.linspace(0.0, 1.0, N_STEPS)
    eps = man["endpoints"]

    conds = []
    for c in man["contexts"]:
        conds.append({"label": f"{c['cid']}|big->large", "ids_A": c["ids_big"],
                      "ids_B": c["ids_large"]})
        conds.append({"label": f"{c['cid']}|big->in", "ids_A": c["ids_big"], "ids_B": c["ids_in"]})

    model = Assay(man["model"], device="cuda", mem_frac=0.225, threads=2, revision=man["revision"])
    curves, checks = sweep(model, conds, ts)
    summaries = summarize_all(curves, ts)
    extra = {"ts": ts.tolist(), "n_layer": model.n_layer, "model": man["model"],
             "revision": man["revision"], "torch": torch.__version__, "dtype": "float32",
             "device": "cuda", "worst_checks": worst_checks(checks),
             "contexts": {c["cid"]: {"class": c["class"], "prefix": c["prefix"],
                                     "prefix_ids": c["prefix_ids"]} for c in man["contexts"]}}
    save("bank2", curves, checks, summaries, extra)
    for c in man["contexts"]:
        s = summaries[f"{c['cid']}|big->in|L0|logits"]
        print(f"{c['cid']:<16} {c['class']:<10} big->in L0: w={s['width']:.3f} "
              f"loc={s['location']:.3f} plateau={int(s['plateau'])}")


if __name__ == "__main__":
    main()
