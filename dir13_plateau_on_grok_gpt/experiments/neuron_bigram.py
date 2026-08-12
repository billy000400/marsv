"""S24d -- is the residual half of the path-bending units responding to LONGER patterns?

Where the series stands. neuron_path.py: the bend in d(t) is carried by a few dozen MLP hidden units
of blocks 1-4 per pair (a pair's own top-32 of 3,840 recover 50.9% of the trained->untrained width
gap; one fixed global 32 recovers 19.0%). neuron_feature.py: those units are character detectors --
corpus tuning conditioned on the CURRENT character predicts recruitment at AUROC 0.847, and
neuron_feature_causal.py turned that into a held-out causal test (32 corpus-selected units recover
28.9%). The residual is the open part: 28.9% against the fitted ceiling's 50.9%. The stated suspect
is the conditioning itself -- a unit that responds to a two-character pattern is summarised crudely
by a profile over single characters.

This script tests that directly, from the SAME corpus pass, by conditioning on the (previous,
current) character pair. Three things fall out of one bigram table m[p, c, j] (mean post-GeLU
activation of unit j at corpus positions whose previous character is p and current character is c):

  1. CONTEXT-MATCHED TUNING. The assay always interpolates the final character of "The house was X",
     so the patched position's previous character is always a space. Restricting the tuning profile
     to corpus positions with previous character ' ' gives a profile measured in the assay's own
     context, with no interpolation involved. Does it predict recruitment better than the
     all-context profile, and -- the test that matters -- do units selected by it alone remove more
     of the width gap than the 28.9% the all-context rule removes?

  2. HOW MUCH OF A UNIT IS THE CURRENT CHARACTER. A weighted two-way decomposition of each unit's
     bigram table into a current-character main effect, a previous-character main effect and a
     residual (interaction) says how much of a unit's corpus response the single-character summary
     can capture at all.

  3. WHAT THE UNPREDICTED RECRUITS ARE. Each pair's fitted top-32 splits into units the all-context
     tuning ranking finds (top decile of |z_a - z_b|) and units it misses. If the miss is caused by
     crude conditioning, the missed units should carry a larger previous-character / interaction
     share than the found ones.

Raw -> results/neuron_bigram_raw.npz, stats -> results/neuron_bigram_summary.json.
"""
import os, sys, json, time, hashlib, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, auroc, CORPUS, CORPUS_SHA, T, MIN_POS, BATCH
from neuron_path import install, uninstall, record_pair, ablate_pair, width, CKPT_DIR, EARLY, N_PAIRS, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MIN_CELL_N = 20       # a (prev, cur) cell needs this many occurrences to enter the decomposition
MIN_CHAR_N = 100      # a character needs this many occurrences under the assay's context
TOP_DECILE = 384      # 3840 / 10: "the tuning ranking finds this unit"
K_LIST = [8, 32, 128]
K_EQ = 8              # matched set size for the found-vs-missed recruit comparison


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    sp = stoi[" "]                       # the assay's previous character
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    data = np.array([stoi[c] for c in raw.decode("utf-8")], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)

    model, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H

    prior = np.load(os.path.join(RES, "neuron_path_raw.npz"))
    pairs, top32 = prior["pairs"], prior["top32"]
    z_all = np.load(os.path.join(RES, "neuron_feature_raw.npz"))["z"]    # [V, U] all-context tuning

    # ---- one corpus pass: bigram table of mean activations -----------------------------------
    sums = torch.zeros(V * V, n_units, dtype=torch.float64, device=device)
    cnt = torch.zeros(V * V, dtype=torch.float64, device=device)
    for b0 in range(0, n_win, BATCH):
        idx = torch.as_tensor(windows[b0:b0 + BATCH], device=device)
        a = gelu_acts(model, idx)[:, MIN_POS:, :].reshape(-1, n_units).double()
        cur = idx[:, MIN_POS:].reshape(-1)
        prev = idx[:, MIN_POS - 1:-1].reshape(-1)
        cell = prev * V + cur
        sums.index_add_(0, cell, a)
        cnt.index_add_(0, cell, torch.ones(len(cell), dtype=torch.float64, device=device))
        if (b0 // BATCH) % 40 == 0:
            print(f"  window {b0}/{n_win} ({time.time()-t0:.0f}s)", flush=True)
    n_cell = cnt.cpu().numpy().reshape(V, V)                              # [prev, cur]
    m = (sums / cnt[:, None].clamp(min=1)).cpu().numpy().reshape(V, V, n_units)
    del sums
    torch.cuda.empty_cache()
    print(f"bigram table done ({time.time()-t0:.0f}s)", flush=True)

    # context-matched profile: previous character is a space, exactly as in "The house was X"
    sel_ctx = m[sp]                                                       # [V, U]
    ctx_n = n_cell[sp].astype(np.int64)
    ok = ctx_n >= MIN_CHAR_N
    z_ctx = np.full_like(sel_ctx, 0.0)
    mu, sd = sel_ctx[ok].mean(axis=0), sel_ctx[ok].std(axis=0)
    z_ctx[ok] = (sel_ctx[ok] - mu[None, :]) / np.maximum(sd[None, :], 1e-8)
    # free check: the all-context marginal recomputed here must match neuron_feature.py's profile
    marg = (m * n_cell[:, :, None]).sum(axis=0) / np.maximum(n_cell.sum(axis=0), 1)[:, None]
    mu2, sd2 = marg.mean(axis=0), marg.std(axis=0)
    z_marg = (marg - mu2[None, :]) / np.maximum(sd2[None, :], 1e-8)
    marg_err = float(np.abs(z_marg - z_all).max())

    # ---- two-way decomposition of each unit's bigram table ------------------------------------
    use = n_cell >= MIN_CELL_N
    w = np.where(use, n_cell, 0.0)                                        # [P, C] cell weights
    N = w.sum()
    g = (w[:, :, None] * m).sum(axis=(0, 1)) / N                          # grand mean per unit
    wc, wp = w.sum(axis=0), w.sum(axis=1)                                 # per-cur, per-prev totals
    mc = (w[:, :, None] * m).sum(axis=0) / np.maximum(wc, 1)[:, None]     # [C, U] current-char effect
    mp = (w[:, :, None] * m).sum(axis=1) / np.maximum(wp, 1)[:, None]     # [P, U] previous-char effect
    tot = (w[:, :, None] * (m - g[None, None, :]) ** 2).sum(axis=(0, 1)) / N
    v_cur = (wc[:, None] * (mc - g[None, :]) ** 2).sum(axis=0) / N
    v_prev = (wp[:, None] * (mp - g[None, :]) ** 2).sum(axis=0) / N
    resid = m - mc[None, :, :] - mp[:, None, :] + g[None, None, :]
    v_int = (w[:, :, None] * resid ** 2).sum(axis=(0, 1)) / N
    tot = np.maximum(tot, 1e-12)
    share_cur, share_prev, share_int = v_cur / tot, v_prev / tot, v_int / tot
    del m, resid
    print(f"decomposition done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 1. does context-matched tuning predict recruitment better? ---------------------------
    methods = ["ctx", "all"]
    rng = np.random.default_rng(0)
    keep = np.array([ok[a] and ok[b] for a, b in pairs])                  # pairs both of whose
    au = {mm: [] for mm in methods}                                       # characters are sampled
    p32 = {mm: [] for mm in methods}                                      # under the assay context
    for i, (a, b) in enumerate(pairs):
        if not keep[i]:
            continue
        lab = np.zeros(n_units, dtype=bool)
        lab[top32[i]] = True
        sc = {"ctx": np.abs(z_ctx[a] - z_ctx[b]), "all": np.abs(z_all[a] - z_all[b])}
        for mm in methods:
            au[mm].append(auroc(sc[mm], lab))
            p32[mm].append(lab[np.argsort(-sc[mm])[:32]].mean())
    au = {mm: np.array(v) for mm, v in au.items()}
    p32 = {mm: np.array(v) for mm, v in p32.items()}

    def ci(x):
        bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)])
        return [round(float(np.percentile(bs, 0.5)), 4), round(float(np.percentile(bs, 99.5)), 4)]

    pred = {mm: {"auroc_mean": round(float(au[mm].mean()), 4), "auroc_ci99": ci(au[mm]),
                 "prec32_mean": round(float(p32[mm].mean()), 4)} for mm in methods}
    pred["p_ctx_vs_all"] = float(wilcoxon(au["ctx"], au["all"]).pvalue)
    pred["n_pairs_scored"] = int(keep.sum())
    pred["min_char_n_under_context"] = MIN_CHAR_N

    # ---- 3. what the units the all-context ranking MISSES look like ---------------------------
    found_u, missed_u = [], []
    for i, (a, b) in enumerate(pairs):
        rank = np.argsort(np.argsort(-np.abs(z_all[a] - z_all[b])))
        f = rank[top32[i]] < TOP_DECILE
        found_u.append(top32[i][f])
        missed_u.append(top32[i][~f])
    found_u, missed_u = np.concatenate(found_u), np.concatenate(missed_u)
    sharp_all = np.abs(z_all).max(axis=0)

    def desc(u):
        return {"n": int(len(u)),
                "share_cur_median": round(float(np.median(share_cur[u])), 4),
                "share_prev_median": round(float(np.median(share_prev[u])), 4),
                "share_int_median": round(float(np.median(share_int[u])), 4),
                "sharpness_median": round(float(np.median(sharp_all[u])), 3)}

    split = {"found": desc(found_u), "missed": desc(missed_u),
             "all_units": desc(np.arange(n_units)),
             "frac_recruits_found": round(float(len(found_u) / (len(found_u) + len(missed_u))), 4),
             "p_share_cur_found_gt_missed":
                 float(mannwhitneyu(share_cur[found_u], share_cur[missed_u],
                                    alternative="greater").pvalue),
             "p_share_int_missed_gt_found":
                 float(mannwhitneyu(share_int[missed_u], share_int[found_u],
                                    alternative="greater").pvalue)}

    # ---- 2/4. causal: linearize the top-k picked by context-matched tuning alone ---------------
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(V), 2))
    rng2 = np.random.default_rng(SEED)
    pairs_chk = [all_pairs[i] for i in rng2.choice(len(all_pairs), N_PAIRS, replace=False)]
    assert (prior["pairs"] == np.array(pairs_chk)).all(), "pair subsample must match neuron_path.py"

    wraps = install(model)
    W = np.full((len(K_LIST), N_PAIRS), np.nan)
    W_fm = np.full((2, N_PAIRS), np.nan)      # matched-size found / missed recruits
    w_base = np.zeros(N_PAIRS)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r, _ = record_pair(model, wraps, seqs, a, b, ts, device)
        w_base[i], _ = width(ts, r["d_logit"])
        order = np.argsort(-np.abs(z_ctx[a] - z_ctx[b]))
        for ki, k in enumerate(K_LIST):
            ra = ablate_pair(model, wraps, seqs, a, b, ts, device, order[:k], H)
            W[ki, i], _ = width(ts, ra["d_logit"])
            ep_worst = max(ep_worst, abs(ra["d0"]), abs(1 - ra["d1"]))
        # top32 is in importance order, so the first K_EQ of each group are the strongest of that
        # group: a like-for-like test of whether the missed recruits carry as much of the bend.
        rank_all = np.argsort(np.argsort(-np.abs(z_all[a] - z_all[b])))
        f = rank_all[top32[i]] < TOP_DECILE
        grp = [top32[i][f], top32[i][~f]]
        if min(len(grp[0]), len(grp[1])) >= K_EQ:
            for gi in range(2):
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, grp[gi][:K_EQ], H)
                W_fm[gi, i], _ = width(ts, ra["d_logit"])
        if (i + 1) % 50 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)
    uninstall(model, wraps)

    wb, wi = float(np.median(w_base)), float(np.median(prior["w_init"]))
    gap = wi - wb
    prior_sum = json.load(open(os.path.join(RES, "neuron_path_summary.json")))
    prior_k = prior_sum["k_list"]
    caus = json.load(open(os.path.join(RES, "neuron_feature_causal.json")))
    causal = {
        "reproduction_check": {"median_w_baseline_here": round(wb, 4),
                               "max_abs_diff_per_pair": round(float(np.abs(w_base - prior["w_base"]).max()), 6)},
        "untrained_median_w": round(wi, 4), "worst_endpoint_error": round(ep_worst, 6),
        "k_list": K_LIST,
        "ctx_selected": {"median_w": [round(float(np.median(W[ki])), 4) for ki in range(len(K_LIST))],
                         "recovered_frac": [round(float((np.median(W[ki]) - wb) / gap), 4)
                                            for ki in range(len(K_LIST))]},
        "all_context_selected": {
            "median_w": [caus["tuning_selected"]["median_w"][caus["k_list"].index(k)] for k in K_LIST],
            "recovered_frac": [caus["tuning_selected"]["recovered_frac"][caus["k_list"].index(k)]
                               for k in K_LIST]},
        "reference_rules": {mm: {"median_w": [prior_sum["curves"][mm]["median_w"][prior_k.index(k)] for k in K_LIST],
                                 "recovered_frac": [prior_sum["curves"][mm]["recovered_frac"][prior_k.index(k)]
                                                    for k in K_LIST]}
                            for mm in ["pair", "global", "random"]},
    }
    prior_caus = np.load(os.path.join(RES, "neuron_feature_causal_raw.npz"))["w_tuning"]
    k32 = K_LIST.index(32)
    causal["paired_at_k32"] = {
        "p_vs_all_context": float(wilcoxon(W[k32], prior_caus[caus["k_list"].index(32)]).pvalue),
        "p_vs_pair_ranked": float(wilcoxon(W[k32], prior["w_pair"][prior_k.index(32)]).pvalue),
        "p_vs_global": float(wilcoxon(W[k32], prior["w_global"][prior_k.index(32)]).pvalue),
        "frac_pairs_wider_than_baseline": round(float((W[k32] > w_base).mean()), 4)}

    # like-for-like: the context-matched profile only exists for characters seen often enough after a
    # space, so score every k=32 rule again on just the pairs where both profiles are well estimated.
    gap_k = float(np.median(prior["w_init"][keep]) - np.median(w_base[keep]))

    def rf(v):
        return round(float((np.median(v[keep]) - np.median(w_base[keep])) / gap_k), 4)

    causal["restricted_to_well_sampled_pairs_k32"] = {
        "n_pairs": int(keep.sum()),
        "ctx": rf(W[k32]), "all_context": rf(prior_caus[caus["k_list"].index(32)]),
        "pair_ranked": rf(prior["w_pair"][prior_k.index(32)]),
        "global": rf(prior["w_global"][prior_k.index(32)]),
        "random": rf(prior["w_random"][prior_k.index(32)]),
        "p_ctx_vs_all_context": float(wilcoxon(W[k32][keep],
                                               prior_caus[caus["k_list"].index(32)][keep]).pvalue)}

    # found vs missed recruits, K_EQ units each, on the pairs where both groups are that large
    fm_ok = ~np.isnan(W_fm[0])
    gap_fm = float(np.median(prior["w_init"][fm_ok]) - np.median(w_base[fm_ok]))
    causal["found_vs_missed_recruits"] = {
        "k_each": K_EQ, "n_pairs": int(fm_ok.sum()),
        "found_median_w": round(float(np.median(W_fm[0][fm_ok])), 4),
        "missed_median_w": round(float(np.median(W_fm[1][fm_ok])), 4),
        "found_recovered_frac": round(float((np.median(W_fm[0][fm_ok]) - np.median(w_base[fm_ok])) / gap_fm), 4),
        "missed_recovered_frac": round(float((np.median(W_fm[1][fm_ok]) - np.median(w_base[fm_ok])) / gap_fm), 4),
        "p_paired": float(wilcoxon(W_fm[0][fm_ok], W_fm[1][fm_ok]).pvalue)}

    summary = {"ckpt_dir": CKPT_DIR, "step": 30000, "context": CONTEXT, "early_blocks": EARLY,
               "n_units": n_units, "n_pairs": N_PAIRS, "min_cell_n": MIN_CELL_N,
               "n_cells_used": int(use.sum()), "top_decile": TOP_DECILE,
               "marginal_check_max_abs_z_diff": round(marg_err, 4),
               "n_chars_sampled_under_context": int(ok.sum()),
               "prediction": pred, "recruit_split": split,
               "variance_shares_all_units": {
                   "cur_median": round(float(np.median(share_cur)), 4),
                   "prev_median": round(float(np.median(share_prev)), 4),
                   "int_median": round(float(np.median(share_int)), 4)},
               "causal": causal}
    with open(os.path.join(RES, "neuron_bigram_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_bigram_raw.npz"),
                        z_ctx=z_ctx.astype(np.float32), ctx_n=ctx_n, ok=ok, keep=keep,
                        share_cur=share_cur.astype(np.float32),
                        share_prev=share_prev.astype(np.float32),
                        share_int=share_int.astype(np.float32),
                        found_u=found_u, missed_u=missed_u, sharp_all=sharp_all.astype(np.float32),
                        w_ctx=W, w_base=w_base, w_fm=W_fm, k_list=np.array(K_LIST),
                        auroc_ctx=au["ctx"], auroc_all=au["all"])
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
