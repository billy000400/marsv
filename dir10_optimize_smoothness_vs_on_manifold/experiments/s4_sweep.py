"""S4 — combined-objective lambda sweep for one adjacent weekday pair.

loss(path) = E_act_norm + lambda * E_out_norm
  E_act = discrete kinetic energy of the path in the first-32 PCA subspace.
  E_out = mean over 16 base prompts of the kinetic energy of the induced 8-bin
          weekday behavior distribution in Hellinger (sqrt-prob) coordinates.
Path = natural cubic spline through 10 control points (endpoints fixed to the two
weekday centroids' first-32 PCA coords); interior 8 control points are optimized
with L-BFGS (strong-Wolfe). Normalization uses the linear-chord energies.

Saves every path + energies + convergence info to results/sweep_<pair>.npz.
"""
import argparse
import json
import os
import numpy as np
import torch

import common as C
import pathlib_opt as P

N_CTRL = 10
N_WP = 11          # evaluation waypoints
N_BASE = 16        # base prompts for E_out
LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0]
DEVICE = "cuda"


def build_context():
    setup = np.load(C.RESULTS + "/weekday_setup.npz")
    V = setup["pca_components"]            # [48, 4096]
    mean = setup["pca_mean"]              # [4096]
    centroids = setup["centroids"]        # [7, 48]
    V32 = torch.tensor(V[:32], dtype=torch.float32, device=DEVICE)   # [32,4096]
    mean_t = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
    M = torch.tensor(P.spline_matrix(N_CTRL, N_WP), dtype=torch.float32, device=DEVICE)
    return V32, mean_t, centroids, M


def base_prompt_indices(seed=0):
    rows = C.build_prompts()
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(rows), N_BASE, replace=False))
    return idx.tolist(), rows


def prepare_bases(tr, idx, rows, V32, mean_t):
    """Cache per-base-prompt: full layer-28 hidden states, last-token activation,
    and its first-32 PCA coords."""
    bases = []
    for i in idx:
        hs = tr.base_hidden(rows[i]["prompt"])       # [seq, hidden] gpu float
        a_last = hs[-1]                              # [hidden]
        coords = V32 @ (a_last - mean_t)             # [32]
        bases.append(dict(hs=hs, a_last=a_last, coords=coords))
    return bases


def waypoints(interior, start, end, M):
    Cfull = torch.cat([start[None], interior, end[None]], dim=0)   # [10,32]
    return M @ Cfull                                               # [N_WP,32]


def e_out(tr, W, bases, V32):
    """Mean over base prompts of Hellinger-space kinetic energy of the induced
    behavior curve. Differentiable w.r.t. W."""
    total = 0.0
    for b in bases:
        shift = (W - b["coords"]) @ V32               # [N_WP, hidden]
        injected = b["a_last"][None] + shift          # [N_WP, hidden]
        dist8 = tr.behavior(b["hs"], injected)        # [N_WP, 8]
        H = torch.sqrt(dist8.clamp_min(0.0) + 1e-12)  # Hellinger coords
        total = total + P.kinetic_energy(H)
    return total / len(bases)


def hellinger(H1, H2):
    """Hellinger distance between two Hellinger-coord vectors (sqrt-probs)."""
    return float((H1 - H2).norm() / np.sqrt(2.0))


@torch.no_grad()
def eval_diag(tr, W, bases, V32):
    """Energies + downstream d(t): mean over base prompts of the Hellinger
    distance of the behavior at each waypoint from the behavior at waypoint 0."""
    eact = float(P.kinetic_energy(W))
    dt_acc = np.zeros(W.shape[0])
    eout = 0.0
    for b in bases:
        shift = (W - b["coords"]) @ V32
        injected = b["a_last"][None] + shift
        dist8 = tr.behavior(b["hs"], injected)
        H = torch.sqrt(dist8.clamp_min(0.0) + 1e-12)      # [N_WP, 8]
        eout += float(P.kinetic_energy(H))
        H0 = H[0]
        dt_acc += np.array([hellinger(H[t], H0) for t in range(W.shape[0])])
    return dict(E_act=eact, E_out=eout / len(bases),
                dt=(dt_acc / len(bases)).tolist(),
                W=W.detach().cpu().numpy())


def spline_ref_path(centroids, a_idx, b_idx, n_wp):
    """Waypoints [n_wp, 32] along the shorter periodic-spline arc from weekday
    a_idx to b_idx (adjacent), in the first-32 PCA coords."""
    from scipy.interpolate import CubicSpline
    t = np.arange(7)
    cs = CubicSpline(np.append(t, 7), np.vstack([centroids, centroids[0]]),
                     bc_type="periodic", axis=0)
    # adjacent shorter arc: knot a_idx -> a_idx+1 (b_idx == (a_idx+1)%7)
    ts = np.linspace(a_idx, a_idx + 1, n_wp)
    arc = cs(ts)[:, :32]                                  # [n_wp, 32]
    return torch.tensor(arc, dtype=torch.float32, device=DEVICE)


def optimize(tr, start, end, M, bases, V32, lam, init_interior,
             outer=30, inner=4, tol=1e-4, out_only=False):
    interior = init_interior.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([interior], max_iter=inner, line_search_fn="strong_wolfe")
    # linear-chord normalizers (constants)
    with torch.no_grad():
        W0 = waypoints(init_interior, start, end, M)
        eact0 = float(P.kinetic_energy(W0))
        eout0 = float(e_out(tr, W0, bases, V32))
    eact0 = max(eact0, 1e-8); eout0 = max(eout0, 1e-8)

    coeff = (1.0 if out_only else lam) / eout0 / len(bases)
    hist = []
    prev = None
    steps = 0
    for step in range(outer):
        def closure():
            # Accumulate gradients per base prompt to bound peak GPU memory:
            # each base's autograd graph is built and freed independently.
            opt.zero_grad()
            loss_val = 0.0
            if not out_only:
                W = waypoints(interior, start, end, M)
                ea = P.kinetic_energy(W) / eact0
                ea.backward()
                loss_val += float(ea)
            for b in ([] if coeff == 0 else bases):
                W = waypoints(interior, start, end, M)     # fresh graph per base
                shift = (W - b["coords"]) @ V32
                injected = b["a_last"][None] + shift
                dist8 = tr.behavior(b["hs"], injected)
                H = torch.sqrt(dist8.clamp_min(0.0) + 1e-12)
                term = coeff * P.kinetic_energy(H)
                term.backward()
                loss_val += float(term)
            return torch.tensor(loss_val)
        loss = opt.step(closure)
        steps += 1
        lv = float(loss)
        hist.append(lv)
        if prev is not None and abs(prev - lv) < tol * max(1.0, abs(prev)):
            break
        prev = lv

    with torch.no_grad():
        W = waypoints(interior, start, end, M)
        diag = eval_diag(tr, W, bases, V32)
        endpoint_err = float(max((W[0] - start).abs().max(),
                                 (W[-1] - end).abs().max()))
    eact, eout = diag["E_act"], diag["E_out"]
    return dict(W=diag["W"], dt=diag["dt"],
                interior=interior.detach().cpu().numpy(),
                E_act=eact, E_act_norm=eact / eact0,
                E_out=eout, E_out_norm=eout / eout0,
                E_act_lin=eact0, E_out_lin=eout0,
                final_loss=hist[-1], steps=steps, endpoint_err=endpoint_err,
                loss_hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="Tuesday-Wednesday")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    a_name, b_name = args.pair.split("-")
    a_idx, b_idx = C.WEEKDAYS.index(a_name), C.WEEKDAYS.index(b_name)

    V32, mean_t, centroids, M = build_context()
    start = torch.tensor(centroids[a_idx, :32], dtype=torch.float32, device=DEVICE)
    end = torch.tensor(centroids[b_idx, :32], dtype=torch.float32, device=DEVICE)

    idx, rows = base_prompt_indices(seed=0)
    tr = P.TailRunner()
    bases = prepare_bases(tr, idx, rows, V32, mean_t)
    tr.move_tail_to_gpu()

    # linear-chord interior control points
    lin_interior = torch.stack([start + (i / (N_CTRL - 1)) * (end - start)
                                for i in range(1, N_CTRL - 1)])  # [8,32]

    results = {}
    for seed in range(args.seeds):
        if seed == 0:
            init = lin_interior.clone()
            itype = "linear"
        else:
            g = torch.Generator(device=DEVICE).manual_seed(seed)
            noise = 0.15 * (end - start).norm() / np.sqrt(32) * \
                torch.randn(lin_interior.shape, generator=g, device=DEVICE)
            init = lin_interior + noise
            itype = f"perturbed_s{seed}"

        for lam in LAMBDAS:
            r = optimize(tr, start, end, M, bases, V32, lam, init)
            r.update(lam=lam, seed=seed, init_type=itype, kind=f"lambda={lam}")
            results[f"{itype}_lam{lam}"] = r
            print(f"[{itype}] lam={lam:<6} E_act={r['E_act']:.3f} "
                  f"E_out={r['E_out']:.5f} loss={r['final_loss']:.4f} "
                  f"steps={r['steps']} epterr={r['endpoint_err']:.1e}")

        # output-only baseline
        r = optimize(tr, start, end, M, bases, V32, 0.0, init, out_only=True)
        r.update(lam=None, seed=seed, init_type=itype, kind="output_only")
        results[f"{itype}_outonly"] = r
        print(f"[{itype}] OUT-ONLY   E_act={r['E_act']:.3f} "
              f"E_out={r['E_out']:.5f} steps={r['steps']} epterr={r['endpoint_err']:.1e}")

    # linear chord + centroid-spline references (no optimization)
    with torch.no_grad():
        W_lin = waypoints(lin_interior, start, end, M)
        lin_ref = eval_diag(tr, W_lin, bases, V32)
        W_spl = spline_ref_path(centroids, a_idx, b_idx, N_WP)
        spline_ref = eval_diag(tr, W_spl, bases, V32)

    out = dict(pair=args.pair, a_idx=a_idx, b_idx=b_idx,
               base_idx=idx, n_wp=N_WP, n_ctrl=N_CTRL,
               lambdas=LAMBDAS, linear=lin_ref, spline_ref=spline_ref,
               results=results)
    np.save(C.RESULTS + f"/sweep_{args.pair}.npy", out, allow_pickle=True)
    with open(C.RESULTS + f"/sweep_{args.pair}_summary.json", "w") as f:
        summ = {k: {kk: v[kk] for kk in
                    ("lam", "seed", "init_type", "kind", "E_act", "E_act_norm",
                     "E_out", "E_out_norm", "final_loss", "steps", "endpoint_err")}
                for k, v in results.items()}
        json.dump(dict(pair=args.pair, linear_E_act=lin_ref["E_act"],
                       linear_E_out=lin_ref["E_out"], results=summ), f, indent=2)
    print("saved sweep for", args.pair)


if __name__ == "__main__":
    main()
