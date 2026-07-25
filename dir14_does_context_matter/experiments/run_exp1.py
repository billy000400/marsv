"""S2 + S3 — reference reproduction and fixed-context endpoint-pair comparison.

Context is fixed to Matthew's `The house was`; the endpoint pair is varied:
    big -> large   (the reference near-synonym pair)
    big -> in      (a different part of speech)
Everything else — model, grid, layers, patch position, hooks, dtype — is identical.

Also runs a deterministic re-run check on one configuration.
"""
import json
import os

import numpy as np
import torch

from sweep import Assay, RES, save, summarize_all, sweep, worst_checks

HERE = os.path.dirname(os.path.abspath(__file__))
N_STEPS = 50


def main():
    with open(os.path.join(RES, "manifest.json")) as f:
        man = json.load(f)
    ts = np.linspace(0.0, 1.0, N_STEPS)

    model = Assay(man["model"], device="cuda", mem_frac=0.225, threads=2,
                  revision=man["revision"])
    conds = [{"label": e["pair"], "ids_A": e["ids_A"], "ids_B": e["ids_B"]} for e in man["exp1"]]
    print("conditions:", [(c["label"], c["ids_A"], c["ids_B"]) for c in conds])

    curves, checks = sweep(model, conds, ts)
    summaries = summarize_all(curves, ts)

    # deterministic re-run of one configuration
    rerun_c, _ = sweep(model, conds[:1], ts, layers=[man.get("rerun_layer", 20)])
    rerun_maxabs = max(float(np.abs(curves[k] - v).max()) for k, v in rerun_c.items())

    extra = {"ts": ts.tolist(), "n_layer": model.n_layer, "model": man["model"],
             "revision": man["revision"], "torch": torch.__version__,
             "dtype": "float32", "device": "cuda",
             "rerun_layer": man.get("rerun_layer", 20), "rerun_maxabs_diff": rerun_maxabs,
             "worst_checks": worst_checks(checks),
             "prompts": {e["pair"]: [e["prompt_A"], e["prompt_B"]] for e in man["exp1"]}}
    save("exp1", curves, checks, summaries, extra)

    print("deterministic re-run max|diff| =", rerun_maxabs)
    for label, w in extra["worst_checks"].items():
        print(label, {k: f"{v:.2e}" for k, v in w.items()})
    for label in [c["label"] for c in conds]:
        s = summaries[f"{label}|L20|logits"]
        print(f"{label} logits @L20: width={s['width']:.3f} loc={s['location']:.3f} "
              f"plateau={s['plateau']}")


if __name__ == "__main__":
    main()
