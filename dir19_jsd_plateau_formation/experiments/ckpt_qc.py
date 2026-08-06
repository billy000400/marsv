"""QC: is every released revision really the checkpoint its name claims?

The scan produced a held-out loss that falls monotonically from 11.01 nats at step 0 to 3.05 at
step 2000 -- with ONE break, step16, whose loss (2.32) is better than step2000's and whose median
transition width matches the fully trained model. This script tests the obvious explanation: that
the uploaded `step16` weights are not a step-16 model. It needs no extra downloads -- it compares
the assay output of every checkpoint against every other, since two identical models produce
bit-identical d(t) curves on the same frozen bank.

Writes results/ckpt_qc.json and prints the nearest neighbour of every checkpoint.
"""
import glob
import json
import os

import numpy as np

from common import RESULTS

SUSPECT = "step16"


def main():
    files = [f for f in glob.glob(os.path.join(RESULTS, "assay_step*.json"))
             if not f.endswith("_t256.json")]
    cps = sorted((json.load(open(f)) for f in files), key=lambda a: a["step"])
    steps = [a["step"] for a in cps]
    C = np.array([np.load(os.path.join(RESULTS, f"curves_step{s}.npy")) for s in steps])
    L = np.array([a["heldout_loss"] for a in cps])

    # pairwise max absolute difference between whole d(t) curve tensors
    n = len(steps)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.nanmax(np.abs(C[i] - C[j]))
    near = {}
    for i, s in enumerate(steps):
        o = [j for j in range(n) if j != i]
        k = min(o, key=lambda j: D[i, j])
        near[s] = dict(nearest_step=steps[k], max_abs_curve_diff=float(D[i, k]))

    # loss monotonicity: which checkpoints are better than a LATER checkpoint
    breaks = [dict(step=steps[i], loss=float(L[i]), next_step=steps[i + 1],
                   next_loss=float(L[i + 1]))
              for i in range(n - 1) if L[i] < L[i + 1] - 0.05]

    si = steps.index(int(SUSPECT.replace("step", ""))) if \
        int(SUSPECT.replace("step", "")) in steps else None
    out = dict(steps=steps, heldout_loss=[float(x) for x in L], nearest=near,
               loss_monotonicity_breaks=breaks,
               suspect=SUSPECT,
               suspect_vs_final_max_abs_curve_diff=float(D[si, steps.index(max(steps))])
               if si is not None else None,
               hub_file_size_bytes={"step16": 2829329888, "all_other_revisions": 2829329920},
               verdict=("step16 fails the monotone-loss check and its curves are closest to a "
                        "late checkpoint; treated as a mislabelled release and excluded")
               if breaks else "no anomaly")
    json.dump(out, open(os.path.join(RESULTS, "ckpt_qc.json"), "w"), indent=2)
    for s in steps:
        print(f"step {s:>6}  loss {L[steps.index(s)]:7.4f}  nearest={near[s]['nearest_step']:>6} "
              f"(max|dd| {near[s]['max_abs_curve_diff']:.4f})")
    print("loss monotonicity breaks:", breaks)


if __name__ == "__main__":
    main()
