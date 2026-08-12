"""S24e -- can a LEARNED description of a unit's corpus response name what the missed recruits read?

Where the series stands. neuron_path.py: a few dozen block-1..4 MLP units per pair carry the bend in
d(t) (a pair's own top-32 of 3,840 recover 50.9% of the trained->untrained width gap).
neuron_feature.py + neuron_feature_causal.py: a hand-built profile over the CURRENT character alone
predicts which units a pair recruits (AUROC 0.847) and, used blind as a selection rule, removes 28.9%
of the gap. neuron_bigram.py: the recruits that rule misses are context-dependent (median 51% of
their corpus response explained by the current character, vs 96% for the ones it finds), and
sharpening the conditioning by hand -- restricting the profile to corpus positions whose previous
character is the assay's space -- ranked better but SELECTED worse (21.9% at k=32), because each
bigram cell is estimated from ~14x fewer positions and that noise bites at the top of the ranking.

This script replaces the hand-built conditioning with a fitted one, using forward passes only. For
every block-1..4 unit it fits a ridge regression predicting the unit's post-GeLU activation from the
characters in a short window ending at the position:

    y_j(pos) ~ intercept + sum_{l=0..L-1} beta_l[char(pos-l)] + gamma[char(pos), char(pos-1)]

i.e. additive one-hot effects for the current character (lag 0) and the seven before it, plus a full
lag0 x lag1 interaction table. Ridge shrinkage is what the raw bigram table lacked: rare cells are
pulled toward the additive fit instead of being estimated from a handful of positions. Windows are
split 80/10/10 into train / lambda-selection / test by window index, so every R^2 quoted is held out.

Three questions, one corpus pass:

  1. IS THE MISSED HALF DESCRIBABLE AT ALL? Held-out R^2 of the lag-0-only model against the full
     model, for the units the character rule finds vs the ones it misses. If the missed units are
     readable from a short character window, their R^2 rises steeply with window length; if they are
     diffuse, it stays low everywhere.

  2. WHERE DOES THE EXTRA STRUCTURE SIT? Incremental held-out R^2 per lag added, and for the
     interaction block, for both groups.

  3. DOES THE FITTED DESCRIPTION SELECT BETTER THAN THE HAND-BUILT ONE? A new blind selection rule:
     evaluate the fitted probe at the assay's own context ("The house was X"), score unit j for pair
     (a, b) by |yhat_j(ctx+a) - yhat_j(ctx+b)|, linearize the top k along the path
     (neuron_path.py's chord substitution) and measure the recovered fraction of the width gap.
     Nothing in the rule has seen d(t), the importance score I_j, or the pair's curve. This is the
     direct test of the noise diagnosis: same conditioning as the bigram rule, shrunk estimator.

Also prints, for the most-reused missed recruits, the probe's largest coefficients (lag, character)
next to the unit's top-activating corpus contexts -- the "is it nameable" evidence.

Raw -> results/neuron_probe_raw.npz, stats -> results/neuron_probe_summary.json.
"""
import os, sys, json, time, hashlib, itertools
import numpy as np
import torch
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, CORPUS, CORPUS_SHA, T, BATCH
from neuron_path import install, uninstall, record_pair, ablate_pair, width, CKPT_DIR, EARLY, N_PAIRS, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
L = 8                       # characters in the window: lag 0 (current) .. lag 7
MIN_POS = L                 # need L characters of history inside the window
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
ORDERS = [1, 2, 4, 8]       # additive window lengths; "full" adds the lag0 x lag1 interaction
TOP_DECILE = 384            # 3840 / 10, as in neuron_bigram.py: "the character rule finds this unit"
K_LIST = [8, 32, 128]
N_EXAMPLE = 6               # missed recruits whose fitted description we print
TOPC = 4                    # corpus contexts kept per example unit


def blocks_of(V):
    """(name, dim) of every one-hot feature group, in column order."""
    return [("intercept", 1)] + [(f"lag{l}", V) for l in range(L)] + [("lag0xlag1", V * V)]


def solve(G, Xty, ncol, lam):
    """ridge coefficients for the first ncol columns; the intercept (column 0) is not penalized."""
    A = G[:ncol, :ncol].clone()
    d = torch.full((ncol,), lam, dtype=A.dtype, device=A.device)
    d[0] = 0.0
    A += torch.diag(d)
    return torch.cholesky_solve(Xty[:ncol], torch.linalg.cholesky(A))


def r2(beta, G, Xty, yty, sy, n):
    """held-out R^2 per unit from sufficient statistics."""
    ncol = beta.shape[0]
    sse = ((beta * (G[:ncol, :ncol] @ beta)).sum(0) - 2 * (beta * Xty[:ncol]).sum(0) + yty)
    tss = yty - sy ** 2 / n
    return (1.0 - sse / tss.clamp(min=1e-12)).cpu().numpy()


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    text = raw.decode("utf-8")
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H

    prior = np.load(os.path.join(RES, "neuron_path_raw.npz"))
    pairs, top32 = prior["pairs"], prior["top32"]
    counts = prior["counts"]
    z_all = np.load(os.path.join(RES, "neuron_feature_raw.npz"))["z"]      # [V, U] character rule

    # which recruits the character rule finds / misses (same split as neuron_bigram.py)
    found_u, missed_u = [], []
    for i, (a, b) in enumerate(pairs):
        rank = np.argsort(np.argsort(-np.abs(z_all[a] - z_all[b])))
        f = rank[top32[i]] < TOP_DECILE
        found_u.append(top32[i][f])
        missed_u.append(top32[i][~f])
    found_u, missed_u = np.concatenate(found_u), np.concatenate(missed_u)
    # example units: the most-reused units that are missed more often than they are found
    miss_cnt = np.bincount(missed_u, minlength=n_units)
    find_cnt = np.bincount(found_u, minlength=n_units)
    ex_units = np.array([j for j in np.argsort(-miss_cnt) if miss_cnt[j] > find_cnt[j]][:N_EXAMPLE])
    ex_t = torch.as_tensor(ex_units, device=device)

    # ---- one corpus pass: sufficient statistics per split -------------------------------------
    grp = blocks_of(V)
    off = np.cumsum([0] + [d for _, d in grp])
    P = int(off[-1])
    print(f"{P} features, {n_units} units, {n_win} windows", flush=True)
    G = [torch.zeros(P, P, dtype=torch.float64, device=device) for _ in range(3)]
    Xty = [torch.zeros(P, n_units, dtype=torch.float64, device=device) for _ in range(3)]
    yty = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    sy = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    npos = [0, 0, 0]
    split_of = np.where(np.arange(n_win) % 10 == 8, 1, np.where(np.arange(n_win) % 10 == 9, 2, 0))
    ex_best = np.full((N_EXAMPLE, TOPC), -np.inf)
    ex_pos = np.zeros((N_EXAMPLE, TOPC), dtype=np.int64)

    for b0 in range(0, n_win, BATCH):
        wb = windows[b0:b0 + BATCH]
        idx = torch.as_tensor(wb, device=device)
        a = gelu_acts(model, idx)[:, MIN_POS:, :]                          # [B, T', U]
        B, Tp, _ = a.shape
        lag = [idx[:, MIN_POS - l:T - l].reshape(-1) for l in range(L)]    # lag l character
        feats = [torch.zeros(B * Tp, dtype=torch.long, device=device)] + lag + \
                [lag[0] * V + lag[1]]
        af = a.reshape(-1, n_units).double()
        sp = torch.as_tensor(np.repeat(split_of[b0:b0 + B], Tp), device=device)
        for s in range(3):
            msk = sp == s
            if not bool(msk.any()):
                continue
            ys = af[msk]
            npos[s] += int(msk.sum())
            yty[s] += (ys * ys).sum(0)
            sy[s] += ys.sum(0)
            fs = [f[msk] for f in feats]
            for gi, (_, dg) in enumerate(grp):
                Xty[s][off[gi]:off[gi + 1]].index_add_(0, fs[gi], ys)
                for hi in range(gi, len(grp)):
                    dh = grp[hi][1]
                    cnt = torch.bincount(fs[gi] * dh + fs[hi], minlength=dg * dh).reshape(dg, dh)
                    G[s][off[gi]:off[gi + 1], off[hi]:off[hi + 1]] += cnt.double()
                    if hi != gi:
                        G[s][off[hi]:off[hi + 1], off[gi]:off[gi + 1]] += cnt.T.double()
        # top-activating corpus contexts for the example units
        v = a[:, :, ex_t].permute(2, 0, 1).reshape(N_EXAMPLE, -1).float().cpu().numpy()
        gpos = ((b0 + np.arange(B))[:, None] * T + np.arange(MIN_POS, T)[None, :]).ravel()
        for u in range(N_EXAMPLE):
            k = np.argpartition(-v[u], TOPC)[:TOPC]
            cv = np.concatenate([ex_best[u], v[u][k]])
            cp = np.concatenate([ex_pos[u], gpos[k]])
            keep = np.argsort(-cv)[:TOPC]
            ex_best[u], ex_pos[u] = cv[keep], cp[keep]
        if (b0 // BATCH) % 20 == 0:
            print(f"  window {b0}/{n_win} ({time.time()-t0:.0f}s)", flush=True)
    del a, af
    torch.cuda.empty_cache()
    print(f"statistics done: {npos} positions per split ({time.time()-t0:.0f}s)", flush=True)

    # ---- fit: lambda on split 1, report R^2 on split 2 ----------------------------------------
    ncols = {str(o): int(off[1 + o]) for o in ORDERS}
    ncols["full"] = P
    R2 = {}
    lam_pick = {}
    beta_full = None
    for name, nc in ncols.items():
        best, best_lam, best_val = None, None, -np.inf
        for lam in LAMBDAS:
            bt = solve(G[0], Xty[0], nc, lam)
            val = float(np.median(r2(bt, G[1], Xty[1], yty[1], sy[1], npos[1])))
            if val > best_val:
                best, best_lam, best_val = bt, lam, val
        R2[name] = r2(best, G[2], Xty[2], yty[2], sy[2], npos[2])
        lam_pick[name] = best_lam
        if name == "full":
            beta_full = best
        print(f"  model {name}: lambda={best_lam}, median held-out R2="
              f"{np.median(R2[name]):.4f} ({time.time()-t0:.0f}s)", flush=True)

    def desc(u):
        return {"n": int(len(u)),
                **{f"r2_{k}_median": round(float(np.median(R2[k][u])), 4) for k in R2},
                "gain_context_median": round(float(np.median(R2["8"][u] - R2["1"][u])), 4),
                "gain_interaction_median": round(float(np.median(R2["full"][u] - R2["8"][u])), 4),
                "frac_explained_by_lag0_median":
                    round(float(np.median(R2["1"][u] / np.maximum(R2["full"][u], 1e-6))), 4)}

    describe = {"found": desc(found_u), "missed": desc(missed_u),
                "all_units": desc(np.arange(n_units)),
                "p_r2full_missed_lt_found":
                    float(mannwhitneyu(R2["full"][missed_u], R2["full"][found_u],
                                       alternative="less").pvalue),
                "p_gain_context_missed_gt_found":
                    float(mannwhitneyu((R2["8"] - R2["1"])[missed_u], (R2["8"] - R2["1"])[found_u],
                                       alternative="greater").pvalue)}

    # ---- fitted description of the example units ---------------------------------------------
    bfull = beta_full.cpu().numpy()
    examples = []
    for u, j in enumerate(ex_units):
        w = bfull[:, j]
        add = [(l, c, float(w[off[1 + l] + c])) for l in range(L) for c in range(V)]
        add.sort(key=lambda x: -abs(x[2]))
        big = [(c, float(w[off[1 + L] + c])) for c in range(V * V)]
        big.sort(key=lambda x: -abs(x[1]))
        examples.append({
            "unit": int(j), "block": EARLY[int(j) // H], "unit_in_block": int(j) % H,
            "n_pairs_missed": int(miss_cnt[j]), "n_pairs_found": int(find_cnt[j]),
            "n_pairs_recruiting": int(counts[j]),
            "r2_lag0": round(float(R2["1"][j]), 4), "r2_full": round(float(R2["full"][j]), 4),
            "top_additive": [{"lag": l, "char": chars[c], "beta": round(b, 3)} for l, c, b in add[:6]],
            "top_interaction": [{"prev": chars[c // V], "cur": chars[c % V], "beta": round(b, 3)}
                                for c, b in big[:4]],
            "contexts": [text[max(0, p - 15):p + 1] for p in ex_pos[u]],
            "act": [round(float(x), 2) for x in ex_best[u]]})

    # ---- probe-selected causal test ----------------------------------------------------------
    # evaluate the fitted probe at the assay's own context: only the last L characters matter.
    ctx = CONTEXT[-(L - 1):]                                # characters at lags 1..L-1
    assert len(ctx) == L - 1, CONTEXT
    yhat = np.zeros((V, n_units))
    base = bfull[0].copy()
    for l in range(1, L):
        base += bfull[off[1 + l] + stoi[ctx[-l]]]
    prev = stoi[ctx[-1]]
    for c in range(V):
        yhat[c] = base + bfull[off[1] + c] + bfull[off[1 + L] + c * V + prev]

    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[ch] for ch in CONTEXT + c], dtype=np.int64) for c in chars]
    all_pairs = list(itertools.combinations(range(V), 2))
    rng = np.random.default_rng(SEED)
    pairs_chk = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]
    assert (prior["pairs"] == np.array(pairs_chk)).all(), "pair subsample must match neuron_path.py"

    wraps = install(model)
    W = np.full((len(K_LIST), N_PAIRS), np.nan)
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r, _ = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r["d_logit"])
        order = np.argsort(-np.abs(yhat[a] - yhat[b]))       # fitted probe, blind to the curve
        for ki, k in enumerate(K_LIST):
            ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order[:k], H)
            W[ki, i], _ = width(ts, ra["d_logit"])
            ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 50 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(prior["w_init"]))
    gap = wi - wb
    prior_sum = json.load(open(os.path.join(RES, "neuron_path_summary.json")))
    prior_k = prior_sum["k_list"]
    caus = json.load(open(os.path.join(RES, "neuron_feature_causal.json")))
    big_sum = json.load(open(os.path.join(RES, "neuron_bigram_summary.json")))
    causal = {
        "reproduction_check": {"median_w_baseline_here": round(wb, 4),
                               "max_abs_diff_per_pair":
                                   round(float(np.abs(w_base - prior["w_base"]).max()), 6)},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "k_list": K_LIST, "n_pairs": N_PAIRS,
        "probe_selected": {"median_w": [round(float(np.median(W[ki])), 4) for ki in range(len(K_LIST))],
                           "recovered_frac": [round(float((np.median(W[ki]) - wb) / gap), 4)
                                              for ki in range(len(K_LIST))]},
        "character_rule": {"median_w": [caus["tuning_selected"]["median_w"][caus["k_list"].index(k)]
                                        for k in K_LIST],
                           "recovered_frac": [caus["tuning_selected"]["recovered_frac"][caus["k_list"].index(k)]
                                              for k in K_LIST]},
        "bigram_rule": {"median_w": [big_sum["causal"]["ctx_selected"]["median_w"][big_sum["causal"]["k_list"].index(k)]
                                     for k in K_LIST],
                        "recovered_frac": [big_sum["causal"]["ctx_selected"]["recovered_frac"][big_sum["causal"]["k_list"].index(k)]
                                           for k in K_LIST]},
        "reference_rules": {m: {"median_w": [prior_sum["curves"][m]["median_w"][prior_k.index(k)] for k in K_LIST],
                                "recovered_frac": [prior_sum["curves"][m]["recovered_frac"][prior_k.index(k)]
                                                   for k in K_LIST]}
                            for m in ["pair", "global", "random"]},
    }
    prior_caus = np.load(os.path.join(RES, "neuron_feature_causal_raw.npz"))["w_tuning"]
    w_ctx = np.load(os.path.join(RES, "neuron_bigram_raw.npz"))["w_ctx"]
    k32 = K_LIST.index(32)
    causal["paired_at_k32"] = {
        "p_vs_character_rule": float(wilcoxon(W[k32], prior_caus[caus["k_list"].index(32)]).pvalue),
        "p_vs_bigram_rule": float(wilcoxon(W[k32], w_ctx[big_sum["causal"]["k_list"].index(32)]).pvalue),
        "p_vs_pair_ranked": float(wilcoxon(W[k32], prior["w_pair"][prior_k.index(32)]).pvalue),
        "p_vs_random": float(wilcoxon(W[k32], prior["w_random"][prior_k.index(32)]).pvalue),
        "frac_pairs_wider_than_baseline": round(float((W[k32] > w_base).mean()), 4)}
    # how much the probe rule and the character rule overlap in what they pick
    ov = []
    for i, (a, b) in enumerate(pairs):
        s1 = set(np.argsort(-np.abs(yhat[a] - yhat[b]))[:32].tolist())
        s2 = set(np.argsort(-np.abs(z_all[a] - z_all[b]))[:32].tolist())
        ov.append(len(s1 & s2))
    causal["median_overlap_top32_with_character_rule"] = float(np.median(ov))

    summary = {
        "ckpt_dir": CKPT_DIR, "step": 30000, "context": CONTEXT, "early_blocks": EARLY,
        "n_units": n_units, "n_pairs": N_PAIRS, "window": L, "n_features": P,
        "lambda_grid": LAMBDAS, "lambda_selected": lam_pick,
        "positions_per_split": npos, "top_decile": TOP_DECILE,
        "describability": describe,
        "r2_by_order_median": {k: round(float(np.median(R2[k])), 4) for k in R2},
        "examples": examples, "causal": causal}
    with open(os.path.join(RES, "neuron_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_raw.npz"),
                        **{f"r2_{k}": R2[k] for k in R2},
                        found_u=found_u, missed_u=missed_u, miss_cnt=miss_cnt, find_cnt=find_cnt,
                        yhat=yhat.astype(np.float32), w_probe=W, w_base=w_base,
                        k_list=np.array(K_LIST), ex_units=ex_units,
                        overlap32=np.array(ov))
    print(json.dumps({k: summary[k] for k in
                      ["lambda_selected", "r2_by_order_median", "describability", "causal"]}, indent=2))
    for e in examples:
        print(f"unit {e['unit']} b{e['block']} missed {e['n_pairs_missed']}x R2 "
              f"{e['r2_lag0']:.2f}->{e['r2_full']:.2f} :: "
              f"{[(t['lag'], t['char'], t['beta']) for t in e['top_additive'][:3]]} :: "
              f"{e['contexts'][0]!r}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
