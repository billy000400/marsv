"""S6 — extend the sweep to all seven adjacent weekday pairs (linear init).

Reuses the pilot machinery. Base prompts (context) are pair-independent, so the
16 base hidden states are computed once; only the endpoint centroids change per
pair. For each pair we run the coarse lambda grid + output-only from the linear
init, and record the activation-manifold recovery of every path.
"""
import json
import numpy as np
import torch

import common as C
import pathlib_opt as P
import s4_sweep as S4
from s5_analyze import dense_arc, recovery

PAIRS = [(i, (i + 1) % 7) for i in range(7)]   # Mon->Tue ... Sun->Mon
DEVICE = "cuda"


def main():
    setup = np.load(C.RESULTS + "/weekday_setup.npz")
    centroids = setup["centroids"]
    V32, mean_t, _, M = S4.build_context()
    idx, rows = S4.base_prompt_indices(seed=0)

    tr = P.TailRunner()
    bases = S4.prepare_bases(tr, idx, rows, V32, mean_t)
    tr.move_tail_to_gpu()

    out = {}
    for a_idx, b_idx in PAIRS:
        name = f"{C.WEEKDAYS[a_idx]}-{C.WEEKDAYS[b_idx]}"
        start = torch.tensor(centroids[a_idx, :32], dtype=torch.float32, device=DEVICE)
        end = torch.tensor(centroids[b_idx, :32], dtype=torch.float32, device=DEVICE)
        lin_interior = torch.stack([start + (i / (S4.N_CTRL - 1)) * (end - start)
                                    for i in range(1, S4.N_CTRL - 1)])
        arc = dense_arc(centroids, a_idx)
        W_spl = S4.spline_ref_path(centroids, a_idx, b_idx, S4.N_WP)
        row = dict(pair=name,
                   rec_spline=recovery(W_spl.cpu().numpy(), arc),
                   spline_E_act=float(P.kinetic_energy(W_spl)),
                   lam={})
        with torch.no_grad():
            W_lin = S4.waypoints(lin_interior, start, end, M)
            row["rec_linear"] = recovery(W_lin.cpu().numpy(), arc)
            row["linear_E_act"] = float(P.kinetic_energy(W_lin))
            row["spline_E_out"] = S4.eval_diag(tr, W_spl, bases, V32)["E_out"]
            row["linear_E_out"] = S4.eval_diag(tr, W_lin, bases, V32)["E_out"]
        for lam in S4.LAMBDAS:
            r = S4.optimize(tr, start, end, M, bases, V32, lam, lin_interior)
            row["lam"][str(lam)] = dict(rec=recovery(np.array(r["W"]), arc),
                                        E_act=r["E_act"], E_out=r["E_out"])
        r = S4.optimize(tr, start, end, M, bases, V32, 0.0, lin_interior, out_only=True)
        row["output_only"] = dict(rec=recovery(np.array(r["W"]), arc),
                                  E_act=r["E_act"], E_out=r["E_out"])
        out[name] = row
        best = min([row["lam"][str(l)]["rec"] for l in S4.LAMBDAS] +
                   [row["output_only"]["rec"]])
        print(f"{name:20s} rec_lin={row['rec_linear']:.3f} "
              f"best_opt={best:.3f} rec_spline={row['rec_spline']:.3f} "
              f"spline_dominated={row['spline_E_act']>row['linear_E_act'] and row['spline_E_out']>row['linear_E_out']}")

    with open(C.RESULTS + "/allpairs_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved allpairs")


if __name__ == "__main__":
    main()
