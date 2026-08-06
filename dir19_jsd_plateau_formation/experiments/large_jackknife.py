"""Do the two 1,000-pair clocks depend on which carrier sentence is used?

Result 12 ran this jackknife on the 60-pair bank: redo the whole scan once per carrier sentence,
with no median over the three, and check that the prespecified onset rules return the same brackets.
The bank where the two headline clocks actually live — the 1,000-pair bank of Results 14-16 — had
never been checked, so every large-bank timing claim still rested on a width that averages three
fixed contexts. If one frame carried the effect, the median over three would still show it.

Both large-bank clocks are re-run verbatim per context, using that single context's w only:

  graded ordering   rho_c(s) = Spearman(J, w_c(s)) on the 600 middle-divergence pairs, significance
                    from the endpoint-label (QAP) permutation with a SIMULTANEOUS envelope over the
                    ten checkpoints; bracket = after the last non-significant checkpoint, by the
                    first of two consecutive significant ones (Result 14).
  ranking lock-in   dpi_c(s) = pi_c(s) - pi_c(0), pi_c(s) = Spearman(w_c(s), w_c(143000)), dyadic
                    endpoint bootstrap with a simultaneous band; same bracket rule (Result 15).

A single context is a noisier measurement than the three-context median, so a bracket can honestly
only move LATER. The question is whether it moves at all and whether any frame reorders the events.

CPU only, ~2 min. Reads results/pair_manifest_large.json and results/assay_large_step*.json.
Writes results/large_jackknife.json.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from common import RESULTS
from large_persistence import wspear

B_PERM = 8000
N_BOOT = 1000
STEPS = [0, 8, 32, 64, 128, 256, 1000, 8000, 64000, 143000]
REF = 143000
RNG = np.random.default_rng(815)


def bracket(steps, sig):
    """After the last non-significant checkpoint, by the first of two consecutive significant."""
    for i in range(len(steps) - 1):
        if sig[i] and sig[i + 1]:
            j = i
            while j > 0 and sig[j - 1]:
                j -= 1
            return [steps[j - 1] if j > 0 else None, steps[j]]
    return None


def rho_rows(rw, rJ):
    a = rw - rw.mean()
    Bc = rJ - rJ.mean(axis=1, keepdims=True)
    return (Bc @ a) / (np.linalg.norm(Bc, axis=1) * np.linalg.norm(a) + 1e-300)


def main():
    man = json.load(open(os.path.join(RESULTS, "pair_manifest_large.json")))
    M = np.asarray(man["jsd_holdout_matrix"], dtype=float)
    pos = {e: i for i, e in enumerate(man["eligible_endpoints"])}
    ea = np.array([pos[p["ep_a"]] for p in man["pairs"]])
    eb = np.array([pos[p["ep_b"]] for p in man["pairs"]])
    J = M[ea, eb]
    n_ep = M.shape[0]

    steps = [s for s in STEPS
             if os.path.exists(os.path.join(RESULTS, f"assay_large_step{s}.json"))]
    Wc = {}                                        # step -> (3, n_pairs) per-context widths
    for s in steps:
        rows = json.load(open(os.path.join(RESULTS, f"assay_large_step{s}.json")))["rows"]
        Wc[s] = np.array([r["w_ctx"] for r in rows], dtype=float).T
    n_p = Wc[steps[0]].shape[1]
    nan_per_ctx = [int(np.isnan(np.stack([Wc[s][c] for s in steps])).sum()) for c in range(3)]
    print(f"{len(steps)} checkpoints x {n_p} pairs x 3 carrier sentences; "
          f"invalid per-context widths: {nan_per_ctx}")

    edges = np.quantile(J, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(J, edges)
    mid = (bins != 0) & (bins != 4)                # the 600 middle-divergence pairs

    perms = np.stack([RNG.permutation(n_ep) for _ in range(B_PERM)])
    Jnull = M[perms[:, ea], perms[:, eb]]
    rJ_null = rankdata(Jnull[:, mid], axis=1)
    rJ_obs = rankdata(J[mid])[None, :]
    mult = RNG.multinomial(n_ep, np.full(n_ep, 1 / n_ep), size=N_BOOT)
    wt = (mult[:, ea] * mult[:, eb]).astype(float)

    out = {"n_perm": B_PERM, "n_boot": N_BOOT, "n_pairs": n_p, "n_middle": int(mid.sum()),
           "n_endpoints": int(n_ep), "steps": steps, "reference_step": REF, "contexts": {}}

    for c in list(range(3)) + ["median3"]:
        W = ({s: np.nanmedian(Wc[s], axis=0) for s in steps} if c == "median3"
             else {s: Wc[s][c] for s in steps})

        # --- graded ordering on the 600 middle pairs, endpoint-label permutation ---
        rho, null_by_step = [], []
        for s in steps:
            rw = rankdata(W[s][mid])
            rho.append(float(rho_rows(rw, rJ_obs)[0]))
            null_by_step.append(np.abs(rho_rows(rw, rJ_null)))
        maxnull = np.stack(null_by_step, 1).max(axis=1)
        sim95 = float(np.quantile(maxnull, 0.95))
        p_fw = [float((1 + np.sum(maxnull >= abs(r) - 1e-12)) / (1 + B_PERM)) for r in rho]
        sig_o = [bool(r < 0 and abs(r) > sim95) for r in rho]
        br_o = bracket(steps, sig_o)

        # --- ranking lock-in, dyadic endpoint bootstrap with a simultaneous band ---
        wf = W[REF]
        pi = [float(np.corrcoef(rankdata(W[s]), rankdata(wf))[0, 1]) for s in steps]
        boot = np.stack([[wspear(W[s], wf, wt[b]) for b in range(N_BOOT)] for s in steps], axis=1)
        dboot = boot - boot[:, [0]]
        dobs = np.array(pi) - pi[0]
        sim_half = float(np.quantile(np.abs(dboot - dboot.mean(0, keepdims=True)).max(1), 0.95))
        sig_r = [bool(d - sim_half > 0) for d in dobs]
        br_r = bracket(steps, sig_r)

        key = "median3" if c == "median3" else f"ctx{c}"
        out["contexts"][key] = dict(
            rho=rho, p_fw=p_fw, sim95_rho=sim95, sig_ordering=sig_o, bracket_ordering=br_o,
            pi=pi, dpi=[float(x) for x in dobs], sim_halfwidth_dpi=sim_half,
            sig_ranking=sig_r, bracket_ranking=br_r)
        print(f"{key}: ordering bracket {br_o}  ranking bracket {br_r}  "
              f"(null95 {sim95:.3f}, dpi band +-{sim_half:.3f})")
        for i, s in enumerate(steps):
            print(f"   step {s:>6}  rho={rho[i]:+.3f} p_fw={p_fw[i]:.4f}"
                  f"{' SIG' if sig_o[i] else '    '}   dpi={dobs[i]:+.3f}"
                  f"{' SIG' if sig_r[i] else ''}")

    # how much the three contexts agree about per-pair width at each checkpoint
    out["ctx_agreement_rbar"] = [
        float(np.mean([np.corrcoef(rankdata(Wc[s][i]), rankdata(Wc[s][j]))[0, 1]
                       for i in range(3) for j in range(i + 1, 3)])) for s in steps]
    print("context agreement r_bar:", [round(x, 3) for x in out["ctx_agreement_rbar"]])

    json.dump(out, open(os.path.join(RESULTS, "large_jackknife.json"), "w"), indent=1)
    print("wrote results/large_jackknife.json")


if __name__ == "__main__":
    main()
