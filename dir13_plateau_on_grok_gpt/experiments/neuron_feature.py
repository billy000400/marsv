"""S24c -- WHAT do the path-bending units detect, and does it predict which paths recruit them?

Where the series stands: `neuron_path.py` showed that the bend in d(t) is carried by a few dozen
MLP hidden units of blocks 1-4 per pair (top-32 of 3,840 recover 50.9% of the trained->untrained
width gap), that the set is pair-dependent (one fixed global 32 recovers only 19.0%), and that 668
distinct units ever enter a per-pair top-32. It did NOT say what those units are for.

This probe answers that with forward passes only, and -- crucially -- the evidence is measured
OUTSIDE the interpolation experiment, so it is not circular:

  1. Character tuning from natural text. Tile the model's own 90% training split into
     non-overlapping 128-character windows, run them through the trained network, and accumulate
     each block-1..4 hidden unit's mean post-GeLU activation conditioned on the character sitting at
     that position. That gives a tuning profile sel[c, j] over the 65 characters for each of the
     3,840 units, from ordinary Shakespeare, with no interpolation and no shared context.

  2. Does tuning predict recruitment? For pair (a, b) the report's assay interpolates the final
     character from a to b under the shared context "The house was ". If the units that bend that
     path are the ones that detect its endpoint characters, then a unit's corpus tuning at a and b
     should predict whether it is in that pair's top-32. We rank all 3,840 units per pair by

        differential  |z_a(j) - z_b(j)|   (fires at one endpoint and not the other)
        max           max(z_a(j), z_b(j)) (fires at either endpoint)

     where z is sel standardized per unit across the 65 characters, and score the ranking by AUROC
     and precision@32 against the recorded top-32 labels. Baselines that use no pair information:
     the unit's overall mean activation (an "is it just an active unit" control), the global
     importance ranking from neuron_path.py (an in-domain reference that knows about the assay but
     not about the pair), and random.

  3. Are the 668 pool units different? Compare tuning sharpness (max |z| over characters) for pool
     vs non-pool units.

Raw -> results/neuron_feature_raw.npz, stats -> results/neuron_feature_summary.json.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon, mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab, CONTEXT
from frozen_assay import load_ckpt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = "/tmp/dir13_frozen/checkpoints_ref_pos"   # reference recipe, seed 1337 (see pos_assay.py)
CORPUS = "/tmp/tinyshakespeare.txt"
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
EARLY = [1, 2, 3, 4]      # the blocks neuron_path.py edited
T = 128                   # training block size
MIN_POS = 8               # skip window-initial positions: too little context to be representative
MIN_CHAR_N = 100          # a character needs this many corpus occurrences for the robustness variant
BATCH = 64
N_EXAMPLE_UNITS = 6       # units whose top-activating corpus contexts we print
TOPC = 6                  # contexts kept per example unit


def gelu_acts(model, idx):
    """post-GeLU hidden activations of blocks EARLY, [B,T,4*H], in EARLY order."""
    store = {}
    hs = [model.blocks[l].mlp.c_fc.register_forward_hook(
        lambda m, i, o, l=l: store.__setitem__(l, F.gelu(o))) for l in EARLY]
    with torch.no_grad():
        model(idx)
    for h in hs:
        h.remove()
    return torch.cat([store[l] for l in EARLY], dim=-1)


def auroc(scores, labels):
    """AUROC of a ranking; labels bool. Ties get average ranks."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    text = raw.decode("utf-8")
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    train_data = data[:n]                      # the split the model was trained on
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)

    model, ck = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H

    prior = np.load(os.path.join(RES, "neuron_path_raw.npz"))
    pairs, top32 = prior["pairs"], prior["top32"]
    counts, imp_mean = prior["counts"], prior["imp_mean"]
    n_pairs = len(pairs)
    pool = counts > 0

    # example units: the most-reused pool units (ties broken by mean importance)
    ex_units = np.lexsort((-imp_mean, -counts))[:N_EXAMPLE_UNITS]
    ex_t = torch.as_tensor(ex_units, device=device)

    # ---- pass over the training split: per-character activation sums ------------------------
    sums = torch.zeros(V, n_units, dtype=torch.float64, device=device)
    cnt = torch.zeros(V, dtype=torch.float64, device=device)
    tot = torch.zeros(n_units, dtype=torch.float64, device=device)
    ex_best = np.full((N_EXAMPLE_UNITS, TOPC), -np.inf)      # running top activations
    ex_pos = np.zeros((N_EXAMPLE_UNITS, TOPC), dtype=np.int64)
    for b0 in range(0, n_win, BATCH):
        wb = windows[b0:b0 + BATCH]
        idx = torch.as_tensor(wb, device=device)
        a = gelu_acts(model, idx)[:, MIN_POS:, :]                       # [B,T',U]
        ch = idx[:, MIN_POS:].reshape(-1)                               # current character
        af = a.reshape(-1, n_units).double()
        sums.index_add_(0, ch, af)
        cnt.index_add_(0, ch, torch.ones(len(ch), dtype=torch.float64, device=device))
        tot += af.sum(dim=0)
        # top-activating corpus positions for the example units
        v = a[:, :, ex_t].permute(2, 0, 1).reshape(N_EXAMPLE_UNITS, -1).float().cpu().numpy()
        gpos = ((b0 + np.arange(len(wb)))[:, None] * T + np.arange(MIN_POS, T)[None, :]).ravel()
        for u in range(N_EXAMPLE_UNITS):
            k = np.argpartition(-v[u], TOPC)[:TOPC]
            cand_v = np.concatenate([ex_best[u], v[u][k]])
            cand_p = np.concatenate([ex_pos[u], gpos[k]])
            keep = np.argsort(-cand_v)[:TOPC]
            ex_best[u], ex_pos[u] = cand_v[keep], cand_p[keep]
        if (b0 // BATCH) % 40 == 0:
            print(f"  window {b0}/{n_win} ({time.time()-t0:.0f}s)", flush=True)
    n_pos = float(cnt.sum().item())
    sel = (sums / cnt[:, None].clamp(min=1)).cpu().numpy()              # [V, U] mean activation
    char_n = cnt.cpu().numpy().astype(np.int64)
    mean_act = (tot / n_pos).cpu().numpy()                              # [U] overall mean
    del sums, tot
    torch.cuda.empty_cache()
    print(f"tuning done: {n_pos:.0f} positions, rarest char n={char_n.min()} "
          f"('{chars[int(char_n.argmin())]}') ({time.time()-t0:.0f}s)", flush=True)

    # standardize each unit's profile across characters
    mu, sd = sel.mean(axis=0), sel.std(axis=0)
    z = (sel - mu[None, :]) / np.maximum(sd[None, :], 1e-8)             # [V, U]

    # ---- does corpus tuning predict which pairs recruit a unit? -----------------------------
    methods = ["differential", "max", "mean_act", "global_imp", "random"]
    rng = np.random.default_rng(0)
    au = {m: np.zeros(n_pairs) for m in methods}
    p32 = {m: np.zeros(n_pairs) for m in methods}
    for i, (a, b) in enumerate(pairs):
        lab = np.zeros(n_units, dtype=bool)
        lab[top32[i]] = True
        sc = {"differential": np.abs(z[a] - z[b]),
              "max": np.maximum(z[a], z[b]),
              "mean_act": mean_act,
              "global_imp": imp_mean,
              "random": rng.random(n_units)}
        for m in methods:
            au[m][i] = auroc(sc[m], lab)
            p32[m][i] = lab[np.argsort(-sc[m])[:32]].mean()

    def ci(x):
        bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)])
        return [round(float(np.percentile(bs, 0.5)), 4), round(float(np.percentile(bs, 99.5)), 4)]

    pred = {m: {"auroc_mean": round(float(au[m].mean()), 4), "auroc_ci99": ci(au[m]),
                "auroc_median": round(float(np.median(au[m])), 4),
                "prec32_mean": round(float(p32[m].mean()), 4),
                "lift_over_chance": round(float(p32[m].mean() / (32 / n_units)), 2),
                "p_vs_random": float(wilcoxon(au[m], au["random"]).pvalue)}
            for m in methods}
    pred["differential"]["p_vs_mean_act"] = float(wilcoxon(au["differential"], au["mean_act"]).pvalue)
    pred["differential"]["p_vs_global_imp"] = float(wilcoxon(au["differential"], au["global_imp"]).pvalue)
    pred["differential"]["p_vs_max"] = float(wilcoxon(au["differential"], au["max"]).pvalue)

    # ---- robustness: three characters occur < 100 times, so their tuning means are noisy.
    # Re-standardize over the 62 well-sampled characters and re-score only the pairs made of them.
    ok = char_n >= MIN_CHAR_N
    z_ok = (sel - sel[ok].mean(axis=0)[None, :]) / np.maximum(sel[ok].std(axis=0)[None, :], 1e-8)
    keep = np.array([ok[a] and ok[b] for a, b in pairs])
    au_r = {m: [] for m in methods}
    p32_r = {m: [] for m in methods}
    for i, (a, b) in enumerate(pairs):
        if not keep[i]:
            continue
        lab = np.zeros(n_units, dtype=bool)
        lab[top32[i]] = True
        sc = {"differential": np.abs(z_ok[a] - z_ok[b]),
              "max": np.maximum(z_ok[a], z_ok[b]),
              "mean_act": mean_act, "global_imp": imp_mean,
              "random": rng.random(n_units)}
        for m in methods:
            au_r[m].append(auroc(sc[m], lab))
            p32_r[m].append(lab[np.argsort(-sc[m])[:32]].mean())
    restricted = {"min_char_n": MIN_CHAR_N, "n_chars_kept": int(ok.sum()),
                  "n_pairs_kept": int(keep.sum()),
                  **{m: {"auroc_mean": round(float(np.mean(au_r[m])), 4),
                         "prec32_mean": round(float(np.mean(p32_r[m])), 4)} for m in methods}}

    # ---- calibration: recruitment rate by differential-tuning decile ------------------------
    dec_rate = np.zeros(10)
    for i, (a, b) in enumerate(pairs):
        d = np.abs(z[a] - z[b])
        rank = np.argsort(np.argsort(-d))            # 0 = strongest
        lab = np.zeros(n_units, dtype=bool)
        lab[top32[i]] = True
        for q in range(10):
            m = (rank >= q * n_units // 10) & (rank < (q + 1) * n_units // 10)
            dec_rate[q] += lab[m].sum()
    dec_rate = dec_rate / (n_pairs * n_units / 10)

    # ---- do recruited units prefer one of the pair's own characters? ------------------------
    pref = z.argmax(axis=0)                          # each unit's most-preferred character
    hit, base = [], []
    for i, (a, b) in enumerate(pairs):
        hit.append(float(np.isin(pref[top32[i]], [a, b]).mean()))
        base.append(float(np.isin(pref, [a, b]).mean()))
    hit, base = np.array(hit), np.array(base)

    # ---- are pool units more sharply tuned than the rest? ----------------------------------
    sharp = np.abs(z).max(axis=0)
    u_sharp = mannwhitneyu(sharp[pool], sharp[~pool], alternative="greater")

    # ---- example units: top corpus contexts + preferred characters --------------------------
    examples = []
    for u, j in enumerate(ex_units):
        blk = EARLY[int(j) // H]
        prof = np.where(ok, z_ok[:, j], -np.inf)   # well-sampled characters only
        top_ch = [chars[c] for c in np.argsort(-prof)[:5]]
        ctxs = [text[max(0, p - 19):p + 1] for p in ex_pos[u]]
        examples.append({"unit": int(j), "block": blk, "unit_in_block": int(j) % H,
                         "n_pairs_recruiting": int(counts[j]),
                         "top_chars": top_ch, "top_z": [round(float(prof[c]), 2)
                                                        for c in np.argsort(-prof)[:5]],
                         "contexts": ctxs, "act": [round(float(v), 2) for v in ex_best[u]]})

    summary = {
        "ckpt_dir": CKPT_DIR, "step": 30000, "context": CONTEXT, "early_blocks": EARLY,
        "hidden_per_block": H, "n_units": n_units, "n_pairs": n_pairs,
        "corpus": {"n_positions": int(n_pos), "n_windows": int(n_win), "min_pos": MIN_POS,
                   "rarest_char": chars[int(char_n.argmin())], "rarest_char_n": int(char_n.min()),
                   "median_char_n": int(np.median(char_n))},
        "prediction": pred, "restricted": restricted,
        "decile_recruit_rate": [round(float(x), 5) for x in dec_rate],
        "chance_recruit_rate": round(32 / n_units, 5),
        "pref_char_is_endpoint": {"recruited_mean": round(float(hit.mean()), 4),
                                  "all_units_mean": round(float(base.mean()), 4),
                                  "lift": round(float(hit.mean() / base.mean()), 2),
                                  "p_paired": float(wilcoxon(hit, base).pvalue)},
        "tuning_sharpness": {"pool_median": round(float(np.median(sharp[pool])), 3),
                             "nonpool_median": round(float(np.median(sharp[~pool])), 3),
                             "n_pool": int(pool.sum()), "p_greater": float(u_sharp.pvalue)},
        "examples": examples,
    }
    with open(os.path.join(RES, "neuron_feature_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_feature_raw.npz"),
                        sel=sel.astype(np.float32), z=z.astype(np.float32), char_n=char_n,
                        mean_act=mean_act.astype(np.float32), sharp=sharp.astype(np.float32),
                        pool=pool, dec_rate=dec_rate, z_ok=z_ok.astype(np.float32), keep=keep, pref=pref, hit=hit, base=base,
                        ex_units=ex_units, chars=np.array(chars),
                        **{f"auroc_{m}": au[m] for m in methods},
                        **{f"prec32_{m}": p32[m] for m in methods})
    print(json.dumps({k: summary[k] for k in
                      ["prediction", "restricted", "pref_char_is_endpoint", "tuning_sharpness",
                       "decile_recruit_rate", "corpus"]}, indent=2))
    for e in examples:
        print(f"unit {e['unit']} (block {e['block']}, {e['n_pairs_recruiting']} pairs) "
              f"top chars {e['top_chars']} :: {e['contexts'][0]!r}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
