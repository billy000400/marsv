"""S1-S3 + first decomposition: what is left in w after corpus successor JSD?

Runs on dir18's stored artifacts only (no model, no GPU).
Writes results/explore1.json and results/contrasts.md.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load, token_index

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- reliability
def reliability(t):
    """How reproducible is a pair's w across the three sentence frames, and how
    does that depend on how far the model's output actually moves (out_jsd)?"""
    W = t["w_ctx"]
    rs = [np.corrcoef(W[:, i], W[:, j])[0, 1] for i, j in ((0, 1), (0, 2), (1, 2))]
    rbar = float(np.mean(rs))
    ceiling = 3 * rbar / (1 + 2 * rbar)          # Spearman-Brown, median-of-3
    edges = [0, 0.1, 0.2, 0.35, 0.6, 1.01]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t["out_jsd_med"] >= lo) & (t["out_jsd_med"] < hi)
        if m.sum() == 0:
            continue
        rb = np.mean([np.corrcoef(W[m][:, i], W[m][:, j])[0, 1]
                      for i, j in ((0, 1), (0, 2), (1, 2))])
        bins.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                         med_spread=float(np.median(t["w_spread"][m])),
                         frame_r=float(rb), med_w=float(np.median(t["w"][m]))))
    return dict(frame_r_pairwise=[float(x) for x in rs], frame_r_mean=rbar,
                ceiling_r2=float(ceiling), by_out_jsd=bins)


# ------------------------------------------------------------ matched contrasts
def contrasts(t, gate, dj=0.02, do=0.05, min_dw=0.15):
    """Pairs of pairs matched on corpus JSD and on model-output JSD whose widths
    differ, and whose ordering holds in every one of the three frames."""
    ok = np.flatnonzero(t["out_jsd_min"] >= gate)
    out = []
    for ii in range(len(ok)):
        i = ok[ii]
        for j in ok[ii + 1:]:
            if abs(t["jsd_B"][i] - t["jsd_B"][j]) > dj:
                continue
            if abs(t["out_jsd_med"][i] - t["out_jsd_med"][j]) > do:
                continue
            n, wd = (i, j) if t["w"][i] < t["w"][j] else (j, i)
            if t["w"][wd] - t["w"][n] < min_dw:
                continue
            if t["w_ctx"][n].max() >= t["w_ctx"][wd].min():
                continue                                  # not consistent in all frames
            shared = ({t["a_str"][n], t["b_str"][n]} & {t["a_str"][wd], t["b_str"][wd]})
            out.append(dict(
                narrow=int(n), wide=int(wd), dw=float(t["w"][wd] - t["w"][n]),
                narrow_pair=f'{str(t["a_str"][n])!r} / {str(t["b_str"][n])!r}',
                wide_pair=f'{str(t["a_str"][wd])!r} / {str(t["b_str"][wd])!r}',
                jsd=float((t["jsd_B"][i] + t["jsd_B"][j]) / 2),
                djsd=float(abs(t["jsd_B"][i] - t["jsd_B"][j])),
                out_jsd_n=float(t["out_jsd_med"][n]), out_jsd_w=float(t["out_jsd_med"][wd]),
                w_n=float(t["w"][n]), w_w=float(t["w"][wd]),
                ent_n=float((t["ent_a"][n] + t["ent_b"][n]) / 2),
                ent_w=float((t["ent_a"][wd] + t["ent_b"][wd]) / 2),
                shared=sorted(str(s) for s in shared)))
    out.sort(key=lambda r: -r["dw"])
    return out


# ------------------------------------------------- token-additive decomposition
def cv_r2(X, y, folds=5, lam=1e-3):
    """Held-out R^2 of ridge-stabilised least squares, 5-fold over pairs."""
    n = len(y)
    perm = np.random.default_rng(0).permutation(n)     # same folds for every model
    pred = np.empty(n)
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te)
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X[tr].T @ y[tr])
        pred[te] = X[te] @ beta
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), pred


def decompose(t, m):
    """Compare predictors of w on the reliable subset: corpus JSD, model-output
    JSD, and an additive per-token model w ~ mu + a_u + a_v."""
    y = t["w"][m]
    toks, ia, ib = token_index(t)
    ia, ib = ia[m], ib[m]
    n = len(y)
    T = np.zeros((n, len(toks)))
    T[np.arange(n), ia] += 1
    T[np.arange(n), ib] += 1
    one = np.ones((n, 1))
    J = t["jsd_B"][m][:, None]
    O = t["out_jsd_med"][m][:, None]
    models = {
        "corpus_jsd": np.hstack([one, J]),
        "corpus_jsd_quad": np.hstack([one, J, J ** 2]),
        "out_jsd": np.hstack([one, O]),
        "token_additive": np.hstack([one, T]),
        "token_additive_plus_jsd": np.hstack([one, T, J]),
        "token_additive_plus_jsd_outjsd": np.hstack([one, T, J, O]),
        "token_additive_plus_jsd_geometry": np.hstack([one, T, J, O, t["cos0"][m][:, None],
                                                       t["dist0"][m][:, None]]),
        "pair_covariates": np.hstack([one, J, O,
                                      t["ent_a"][m][:, None] + t["ent_b"][m][:, None],
                                      np.log10(t["count_a"][m] * t["count_b"][m])[:, None] / 2,
                                      t["surp_a"][m][:, None] + t["surp_b"][m][:, None],
                                      t["cos0"][m][:, None], t["dist0"][m][:, None]]),
    }
    res = {k: cv_r2(X, y)[0] for k, X in models.items()}

    # fitted token effects (full fit) and what they correlate with
    A = np.hstack([one, T])
    beta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y)
    a = beta[1:]
    seen = np.zeros(len(toks))
    ent = np.zeros(len(toks)); frq = np.zeros(len(toks)); srp = np.zeros(len(toks))
    for k in range(n):
        for tk, e, c, s in ((ia[k], t["ent_a"][m][k], t["count_a"][m][k], t["surp_a"][m][k]),
                            (ib[k], t["ent_b"][m][k], t["count_b"][m][k], t["surp_b"][m][k])):
            seen[tk] += 1; ent[tk] += e; frq[tk] += np.log10(c); srp[tk] += s
    u = seen >= 4
    ent, frq, srp = ent[u] / seen[u], frq[u] / seen[u], srp[u] / seen[u]
    tok_corr = {name: [float(x) for x in spearmanr(a[u], v)[:2]]
                for name, v in (("continuation_entropy", ent), ("log10_frequency", frq),
                                ("model_surprisal", srp))}
    return res, dict(tokens=[toks[i] for i in np.flatnonzero(u)],
                     effect=[float(x) for x in a[u]],
                     ent=[float(x) for x in ent], logf=[float(x) for x in frq],
                     surp=[float(x) for x in srp], n_uses=[int(x) for x in seen[u]],
                     corr=tok_corr)


def interaction(t, m):
    """Is anything left after `mu + a_u + a_v + b J`, or is the rest frame noise?
    Fit the additive model separately in each sentence frame and correlate the
    residuals across frames: a positive correlation is pair-specific structure
    that reproduces, a zero correlation means the leftover is measurement noise."""
    toks, ia, ib = token_index(t)
    ia, ib = ia[m], ib[m]
    n = int(m.sum())
    T = np.zeros((n, len(toks)))
    T[np.arange(n), ia] += 1
    T[np.arange(n), ib] += 1
    base = [np.ones((n, 1)), T, t["jsd_B"][m][:, None]]
    out = {}
    for name, extra in (("additive_jsd", []),
                        ("additive_jsd_outjsd_geometry",
                         [t["out_jsd_med"][m][:, None], t["cos0"][m][:, None],
                          t["dist0"][m][:, None]])):
        X = np.hstack(base + extra)
        P = X @ np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T)
        R = np.column_stack([t["w_ctx"][m][:, c] - P @ t["w_ctx"][m][:, c] for c in range(3)])
        r = [float(np.corrcoef(R[:, i], R[:, j])[0, 1]) for i, j in ((0, 1), (0, 2), (1, 2))]
        out[name] = dict(resid_frame_r=r, resid_frame_r_mean=float(np.mean(r)),
                         resid_ceiling_r2=3 * float(np.mean(r)) / (1 + 2 * float(np.mean(r))))
    rs = out["additive_jsd"]["resid_frame_r"]
    rbar = out["additive_jsd"]["resid_frame_r_mean"]
    # raw across-frame reliability of w for reference, same subset
    W = t["w_ctx"][m]
    raw = float(np.mean([np.corrcoef(W[:, i], W[:, j])[0, 1] for i, j in ((0, 1), (0, 2), (1, 2))]))
    return dict(resid_frame_r=rs, resid_frame_r_mean=rbar,
                resid_ceiling_r2=3 * rbar / (1 + 2 * rbar),
                raw_frame_r_mean=raw, models=out)


def path_length(t, m):
    """`w` is a fraction of the path, so a fixed-size transition zone on a longer
    path would look narrower. If that were the whole story, the transition width
    measured in residual-distance units would be more homogeneous than w."""
    w, D = t["w"][m], t["dist0"][m]
    wabs = w * D
    cv = lambda x: float(x.std() / x.mean())
    return dict(cv_w=cv(w), cv_w_absolute=cv(wabs),
                rho_dist0_w=[float(x) for x in spearmanr(D, w)[:2]],
                rho_cos0_w=[float(x) for x in spearmanr(t["cos0"][m], w)[:2]],
                rho_jsd_w_absolute=[float(x) for x in spearmanr(t["jsd_B"][m], wabs)[:2]],
                median_dist0=float(np.median(D)))


def main():
    t, curves, grid = load()
    rel = reliability(t)
    GATE = 0.2                                   # bits of endpoint output movement
    m = t["out_jsd_min"] >= GATE
    rho_all = spearmanr(t["jsd_B"], t["w"])
    rho_ok = spearmanr(t["jsd_B"][m], t["w"][m])
    con = contrasts(t, GATE)
    dec, tok = decompose(t, m)

    out = dict(
        n_pairs=int(len(t["w"])), gate_bits=GATE, n_reliable=int(m.sum()),
        reliability=rel,
        rho_jsd_w_all=[float(rho_all[0]), float(rho_all[1])],
        rho_jsd_w_reliable=[float(rho_ok[0]), float(rho_ok[1])],
        rho_outjsd_w_reliable=[float(x) for x in spearmanr(t["out_jsd_med"][m], t["w"][m])[:2]],
        n_contrasts=len(con), contrasts=con[:40],
        n_contrasts_shared_token=int(sum(1 for c in con if c["shared"])),
        cv_r2=dec, token_effects=tok, interaction=interaction(t, m),
        path_length=path_length(t, m),
    )
    json.dump(out, open(os.path.join(RESULTS, "explore1.json"), "w"), indent=1)

    print(f"reliable pairs {m.sum()}/{len(m)}   ceiling R2 {rel['ceiling_r2']:.3f}")
    print("rho(J,w) all", np.round(rho_all[0], 3), " reliable", np.round(rho_ok[0], 3))
    for k, v in dec.items():
        print(f"  CV-R2 {k:26s} {v:+.3f}")
    print("token-effect correlations:", {k: np.round(v[0], 3) for k, v in tok["corr"].items()})
    print(f"contrasts: {len(con)} (shared token: {out['n_contrasts_shared_token']})")
    for c in con[:8]:
        print(f"  dw={c['dw']:.3f} J={c['jsd']:.3f} "
              f"narrow {c['narrow_pair']} w={c['w_n']:.2f} | wide {c['wide_pair']} w={c['w_w']:.2f}")


if __name__ == "__main__":
    main()
