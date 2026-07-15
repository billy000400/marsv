"""Sanity: the GPU tail must reproduce the full-model behavior distribution when
we inject a prompt's own layer-28 last-token activation. Also checks the spline
matrix (endpoints exact, lambda=0 chord) and kinetic-energy finite differences."""
import json
import numpy as np
import torch

import common as C
import pathlib_opt as P


def main():
    setup = np.load(C.RESULTS + "/weekday_setup.npz")
    dist8_full = setup["dist8"]
    rows = C.build_prompts()

    tr = P.TailRunner()
    idxs = [0, 10, 24, 37, 48]          # a spread of prompts
    base_hs = {i: tr.base_hidden(rows[i]["prompt"]) for i in idxs}
    tr.move_tail_to_gpu()

    print("=== tail vs full-model behavior (L1 over 8 bins) ===")
    max_err = 0.0
    for i in idxs:
        inj = base_hs[i][-1:].clone()             # own last-token activation [1, hidden]
        d = tr.behavior(base_hs[i], inj)[0].detach().cpu().numpy()
        err = float(np.abs(d - dist8_full[i]).sum())
        max_err = max(max_err, err)
        print(f"prompt {i:2d}  L1={err:.4f}  argmax tail={int(d[:7].argmax())} "
              f"full={int(dist8_full[i][:7].argmax())}  gt={rows[i]['gt_idx']}")
    print(f"max L1 = {max_err:.4f}")

    # spline checks
    M = P.spline_matrix(n_ctrl=10, n_wp=20)
    start = np.random.RandomState(0).randn(32)
    end = np.random.RandomState(1).randn(32)
    C_ctrl = np.stack([start + (i / 9) * (end - start) for i in range(10)])  # linear
    W = M @ C_ctrl
    print("\n=== spline ===")
    print("endpoint err start:", np.abs(W[0] - start).max(),
          "end:", np.abs(W[-1] - end).max())
    # for a linear control polygon the natural spline is exactly linear -> waypoints on chord
    chord = np.stack([start + t * (end - start) for t in np.linspace(0, 1, 20)])
    print("linear-init waypoints vs chord max dev:", np.abs(W - chord).max())

    Wt = torch.tensor(W)
    print("kinetic energy (linear chord):", float(P.kinetic_energy(Wt)))


if __name__ == "__main__":
    main()
