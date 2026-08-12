"""What is left of a demoted unit's response, once character identity is exhausted?

Where the series stands. neuron_probe_early.py fitted a ridge probe predicting each block-1..4 MLP
unit's post-GeLU activation from the characters in an 8-character window ending at the position, and
found that the units which lose head membership over training become harder to read (median held-out
R^2 falls). neuron_probe_struct's predecessor, neuron_probe_family.py, asked whether that decline was
an artefact of the 8-character window: widening the family to 32 characters of history plus 8 of
lookahead moved the demoted units' per-unit fall by 0.0002 (-0.0635 -> -0.0637), so the decline is not
a window-width artefact. But it left the *level* unexplained: at step 30,000 the demoted units still
sit at median R^2 0.78 under the widest character family (all40), i.e. about a fifth of their corpus
variance is not linear in the identity of any character in a 40-character neighbourhood.

PLAN's remaining successor named the two candidates for that residual: structure beyond the 40
characters the probe sees, and structure that is not character identity at all (word or line position,
lexical identity, "syntactic role"). This script fits both, on exactly the rows and split
neuron_probe_family.py used, so every number here is comparable to that run unit by unit:

    char1    intercept + the current character                     (published headline family)
    past8    char1 + lags 1..7 + the full lag0 x lag1 table        (published full family)
    all40    past8 + lags 8..31 + the 8 characters after the position
    struct   where the position sits in the line and the word      (NEW, no character identity)
    lex      which word / 3-character string the position is in    (NEW, conjunctions of characters)
    bag      which characters occur before lag 32, position-free   (NEW, reaches past all40's window)
    nonchar  struct + lex + bag                                    (everything new, together)
    union    all40 + nonchar                                       (everything)

char1 / past8 / all40 are refitted rather than copied: the union design's submatrices are identical, so
they must reproduce neuron_probe_family_summary.json to the last digit, and that is the pipeline check.

The two questions the report consumes:
  1. LEVEL. Does `union` raise the demoted units' step-30,000 R^2 above all40's 0.78, and is any such
     gain specific to them or shared with the units that stay readable (the `stable` role)?
  2. DECLINE. Does the fall from step 831 to step 30,000 shrink under `union`? If the demoted units had
     merely switched to reading word/line structure, the fall should largely disappear there.

Unit roles (stable / demoted / promoted / never_head) are loaded from
results/neuron_probe_family_raw.npz -- the same checkpoints on disk, so the roles are the same objects.

Raw -> results/neuron_probe_struct_raw.npz, stats -> results/neuron_probe_struct_summary.json,
figure -> plots/neuron_probe_struct.png.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from allpairs_sweep import load_vocab
from frozen_assay import load_ckpt
from neuron_feature import gelu_acts, CORPUS, CORPUS_SHA, T, BATCH
from neuron_path import CKPT_DIR, EARLY
from neuron_probe import solve, r2, LAMBDAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
STEPS = [831, 30000]
NLAG = 32            # characters of history in the widest character family (as in neuron_probe_family)
NFWD = 8             # characters of lookahead in that family
COLCAP = 40          # line-offset buckets 0..40, plus one "no newline in window" bucket
WCAP = 12            # word-offset and previous-word-length buckets 0..12, plus one censored bucket
RUNCAP = 4           # capital-run buckets 0..4
N_TRI = 511          # most frequent 3-character strings kept; the rest share an "other" bucket
N_WORD = 255         # most frequent word-so-far / previous-word strings kept, plus "other"
HASH_MOD = (1 << 61) - 1
HASH_BASE = 131
GROUPS = ["stable", "demoted", "promoted", "never_head"]
FAMILIES = ["char1", "past8", "all40", "struct", "lex", "bag", "nonchar", "union"]
NEW = ["struct", "lex", "bag"]
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
MK = ["o", "s", "^", "D", "v"]


# ----------------------------------------------------------------------------- feature layout
def groups_of(V):
    """(name, dim) of every ONE-HOT feature group in the union design, in column order.

    Every group here contributes exactly one 1 per row, which is what lets the batch design matrix be
    built with a single scatter. The bag block is dense and is appended after these.
    """
    return ([("intercept", 1)]
            + [(f"lag{l}", V) for l in range(NLAG)]
            + [(f"next{k}", V) for k in range(1, NFWD + 1)]
            + [("lag0xlag1", V * V)]
            + [("col", COLCAP + 2), ("wpos", WCAP + 2), ("prevwlen", WCAP + 2),
               ("nextsep", NFWD + 1), ("capsrun", RUNCAP + 1)]
            + [("tri", N_TRI + 1), ("curword", N_WORD + 2), ("prevword", N_WORD + 2)])


def offsets(V):
    off = np.cumsum([0] + [d for _, d in groups_of(V)])
    n_onehot = int(off[-1])
    return off, n_onehot, n_onehot + V + 1        # + bag frequencies + prefix length


def family_cols(V):
    """column indices of each fitted family inside the union design."""
    off, n_onehot, P = offsets(V)
    name_at = {n: i for i, (n, _) in enumerate(groups_of(V))}
    g = lambda n: np.arange(off[name_at[n]], off[name_at[n] + 1])
    inter = g("lag0xlag1")
    char1 = np.concatenate([[0], g("lag0")])
    past8 = np.concatenate([char1] + [g(f"lag{l}") for l in range(1, 8)] + [inter])
    all40 = np.concatenate([past8] + [g(f"lag{l}") for l in range(8, NLAG)]
                           + [g(f"next{k}") for k in range(1, NFWD + 1)])
    struct = np.concatenate([[0]] + [g(n) for n in ["col", "wpos", "prevwlen", "nextsep", "capsrun"]])
    lex = np.concatenate([[0]] + [g(n) for n in ["tri", "curword", "prevword"]])
    bag = np.concatenate([[0], np.arange(n_onehot, P)])
    nonchar = np.union1d(np.union1d(struct, lex), bag)
    return {"char1": char1, "past8": past8, "all40": all40, "struct": struct, "lex": lex,
            "bag": bag, "nonchar": nonchar, "union": np.union1d(all40, nonchar)}


# ----------------------------------------------------------------------------- string vocabularies
def poly_hash(s, itos_id):
    h = 0
    for ch in s:
        h = (h * HASH_BASE + itos_id[ch]) % HASH_MOD
    return h


def build_vocabs(text, stoi):
    """Frequency-ranked vocabularies for the lexical family, counted on the probe's own text.

    - trigrams: every 3-character substring.
    - word-so-far: the characters since the last separator (' ' or '\\n'), i.e. the partial word ending
      at the position; the empty string (the position IS a separator) gets its own bucket.
    - previous word: the last completed word before that separator.
    """
    tri, cur, prev = {}, {}, {}
    word, last = "", ""
    for i, ch in enumerate(text):
        if i >= 2:
            k = text[i - 2:i + 1]
            tri[k] = tri.get(k, 0) + 1
        if ch in (" ", "\n"):
            last, word = word, ""
        else:
            word += ch
        cur[word] = cur.get(word, 0) + 1
        prev[last] = prev.get(last, 0) + 1
    top = lambda d, n: [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
    V = len(stoi)
    tri_tab = np.full(V ** 3, N_TRI, dtype=np.int64)
    for r, k in enumerate(top(tri, N_TRI)):
        tri_tab[(stoi[k[0]] * V + stoi[k[1]]) * V + stoi[k[2]]] = r
    def wtab(d):
        ks = [k for k in top(d, N_WORD + 1) if k != ""]        # "" is bucket N_WORD by construction
        h = np.array([poly_hash(k, stoi) for k in ks[:N_WORD]], dtype=np.int64)
        o = np.argsort(h)
        return h[o], np.arange(len(ks[:N_WORD]))[o]
    return tri_tab, wtab(cur), wtab(prev)


# ----------------------------------------------------------------------------- batch features
def word_hashes(idx, sep):
    """[B,T] polynomial hash of the word-so-far ending at each position (0 at separators)."""
    B, Tw = idx.shape
    h = torch.zeros(B, Tw, dtype=torch.int64, device=idx.device)
    cur = torch.zeros(B, dtype=torch.int64, device=idx.device)
    for t in range(Tw):
        c = idx[:, t]
        cur = torch.where(sep[:, t], torch.zeros_like(cur),
                          (cur * HASH_BASE + c) % HASH_MOD)
        h[:, t] = cur
    return h


def lookup(h, table):
    """map hashes to their vocabulary rank, or the 'other' bucket, via sorted search."""
    keys, ranks = table
    j = torch.searchsorted(keys, h.reshape(-1).contiguous()).clamp(max=keys.numel() - 1)
    hit = keys[j] == h.reshape(-1)
    out = torch.where(hit, ranks[j], torch.full_like(j, N_WORD + 1))
    return out.reshape(h.shape)


def batch_codes(idx, V, tri_tab, cur_tab, prev_tab, nl, sp):
    """one-hot group codes [rows, n_groups] and the dense bag block [rows, V+1] for one batch.

    Rows are (window, position) with position in NLAG..T-NFWD-1, flattened in that order -- the same
    row set and order as neuron_probe_family.py.
    """
    B, Tw = idx.shape
    dev = idx.device
    pos = torch.arange(Tw, device=dev).expand(B, Tw)
    lo, hi = NLAG, Tw - NFWD
    sl = slice(lo, hi)

    sep = (idx == nl) | (idx == sp)
    # last index <= p of a newline / separator / non-capital, by running maximum
    last_of = lambda m: torch.cummax(torch.where(m, pos, torch.full_like(pos, -1)), dim=1).values
    last_nl, last_sep = last_of(idx == nl), last_of(sep)
    caps = (idx >= stoi_A) & (idx <= stoi_Z)
    last_low = last_of(~caps)

    col = torch.where(last_nl[:, sl] >= 0, (pos[:, sl] - last_nl[:, sl]).clamp(max=COLCAP),
                      torch.full_like(pos[:, sl], COLCAP + 1))
    wpos = torch.where(last_sep[:, sl] >= 0, (pos[:, sl] - last_sep[:, sl]).clamp(max=WCAP),
                       torch.full_like(pos[:, sl], WCAP + 1))
    # previous word length: separators at q1 = last_sep[p] and q2 = last_sep[q1 - 1]
    q1 = last_sep[:, sl]
    q2 = torch.gather(last_sep, 1, q1.clamp(min=1) - 1)
    ok = (q1 >= 1) & (q2 >= 0)
    prevwlen = torch.where(ok, (q1 - q2 - 1).clamp(min=0, max=WCAP),
                           torch.full_like(q1, WCAP + 1))
    nextsep = torch.zeros_like(col)
    for k in range(NFWD, 0, -1):
        nextsep = torch.where(sep[:, lo + k:hi + k], torch.full_like(nextsep, k), nextsep)
    capsrun = (pos[:, sl] - last_low[:, sl]).clamp(min=0, max=RUNCAP)
    capsrun = torch.where(last_low[:, sl] >= 0, capsrun, torch.full_like(capsrun, RUNCAP))

    lag = [idx[:, lo - l:hi - l] for l in range(NLAG)]
    fwd = [idx[:, lo + k:hi + k] for k in range(1, NFWD + 1)]
    tri = tri_tab[(lag[2] * V + lag[1]) * V + lag[0]]
    wh = word_hashes(idx, sep)
    curw = torch.where(sep[:, sl], torch.full_like(col, N_WORD),
                       lookup(wh[:, sl], cur_tab))
    prevh = torch.gather(wh, 1, q1.clamp(min=1) - 1)
    prevw = torch.where(ok & ~sep.gather(1, q1.clamp(min=1) - 1),
                        lookup(prevh, prev_tab), torch.full_like(col, N_WORD))

    codes = torch.stack([torch.zeros_like(col)] + lag + fwd + [lag[0] * V + lag[1]]
                        + [col, wpos, prevwlen, nextsep, capsrun]
                        + [tri, curw, prevw], dim=-1).reshape(-1, 1 + NLAG + NFWD + 1 + 5 + 3)

    # bag: character frequencies over window positions 0..p-NLAG (everything before lag NLAG-1),
    # plus that prefix's length -- a position-free summary that reaches past all40's window
    onehot = torch.zeros(B, Tw, V, dtype=torch.float64, device=dev)
    onehot.scatter_(2, idx.unsqueeze(-1), 1.0)
    csum = onehot.cumsum(1)[:, :hi - lo, :]                 # prefix 0..p-NLAG for p in lo..hi-1
    plen = (pos[:, sl] - NLAG + 1).double()
    bag = torch.cat([csum / plen.unsqueeze(-1), (plen / Tw).unsqueeze(-1)], dim=-1)
    return codes, bag.reshape(-1, V + 1)


# ----------------------------------------------------------------------------- fitting
def fit(step, windows, split_of, V, tabs, nl, sp, device, t0):
    model, _ = load_ckpt(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H
    off, n_onehot, P = offsets(V)
    G = [torch.zeros(P, P, dtype=torch.float64, device=device) for _ in range(3)]
    Xty = [torch.zeros(P, n_units, dtype=torch.float64, device=device) for _ in range(3)]
    yty = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    sy = [torch.zeros(n_units, dtype=torch.float64, device=device) for _ in range(3)]
    npos = [0, 0, 0]
    off_t = torch.as_tensor(off[:-1], device=device)

    for b0 in range(0, windows.shape[0], BATCH):
        idx = torch.as_tensor(windows[b0:b0 + BATCH], device=device)
        a = gelu_acts(model, idx)[:, NLAG:T - NFWD, :]
        B, Tp, _ = a.shape
        codes, bag = batch_codes(idx, V, *tabs, nl, sp)
        af = a.reshape(-1, n_units).double()
        X = torch.zeros(codes.shape[0], P, dtype=torch.float64, device=device)
        X.scatter_(1, codes + off_t, 1.0)
        X[:, n_onehot:] = bag
        spl = torch.as_tensor(np.repeat(split_of[b0:b0 + B], Tp), device=device)
        for s in range(3):
            msk = spl == s
            if not bool(msk.any()):
                continue
            ys, xs = af[msk], X[msk]
            npos[s] += int(msk.sum())
            yty[s] += (ys * ys).sum(0)
            sy[s] += ys.sum(0)
            G[s] += xs.T @ xs
            Xty[s] += xs.T @ ys
            del xs
        del a, af, X, codes, bag
    print(f"  step {step}: statistics done, {npos} positions ({time.time()-t0:.0f}s)", flush=True)

    cols = family_cols(V)
    out, lam_pick = {}, {}
    for fam in FAMILIES:
        c = torch.as_tensor(cols[fam], device=device)
        Gs = [g[c][:, c].contiguous() for g in G]
        Xs = [x[c].contiguous() for x in Xty]
        best, best_val, best_lam = None, -np.inf, None
        for lam in LAMBDAS:
            bt = solve(Gs[0], Xs[0], c.numel(), lam)
            val = float(np.median(r2(bt, Gs[1], Xs[1], yty[1], sy[1], npos[1])))
            if val > best_val:
                best, best_val, best_lam = bt, val, lam
        out[fam] = r2(best, Gs[2], Xs[2], yty[2], sy[2], npos[2])
        lam_pick[fam] = best_lam
        print(f"  step {step}: {fam} ({c.numel()} cols, lambda {best_lam:g}) median R2 "
              f"{np.median(out[fam]):.4f} ({time.time()-t0:.0f}s)", flush=True)
        del Gs, Xs
    del G, Xty, yty, sy, model
    torch.cuda.empty_cache()
    return out, lam_pick


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    t0 = time.time()

    stoi = load_vocab()
    V = len(stoi)
    itos = {i: c for c, i in stoi.items()}
    global stoi_A, stoi_Z, STOI
    stoi_A, stoi_Z, STOI = stoi["A"], stoi["Z"], stoi
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA, "corpus SHA mismatch"
    text = raw.decode("utf-8")
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    train_data = data[:int(0.9 * len(data))]
    n_win = len(train_data) // T
    windows = train_data[:n_win * T].reshape(n_win, T)
    split_of = np.where(np.arange(n_win) % 10 == 8, 1, np.where(np.arange(n_win) % 10 == 9, 2, 0))
    if "--smoke" in sys.argv:
        windows, split_of = windows[:640], split_of[:640]

    tri_tab, cur_tab, prev_tab = build_vocabs(text[:int(0.9 * len(text))], stoi)
    print(f"vocabularies built ({time.time()-t0:.0f}s)", flush=True)
    to_dev = lambda t: (torch.as_tensor(t[0], device=device), torch.as_tensor(t[1], device=device))
    tabs = (torch.as_tensor(tri_tab, device=device), to_dev(cur_tab), to_dev(prev_tab))
    nl, sp = stoi["\n"], stoi[" "]
    self_test_light(windows, V, tabs, nl, sp, itos, device)

    R2, lam = {}, {}
    for s in STEPS:
        R2[s], lam[s] = fit(s, windows, split_of, V, tabs, nl, sp, device, t0)

    z = np.load(os.path.join(RES, "neuron_probe_family_raw.npz"))
    role = {g: z[g] for g in GROUPS}
    n_units = R2[STEPS[0]]["char1"].size
    check = {}
    for s in STEPS:
        for fam in ["char1", "past8", "all40"]:
            p = z[f"r2_{fam}_{s}"]
            check[f"max_abs_diff_{fam}_{s}_vs_family_run"] = float(np.max(np.abs(R2[s][fam] - p)))
            check[f"median_{fam}_{s}"] = [round(float(np.median(R2[s][fam])), 4),
                                          round(float(np.median(p)), 4)]
    print(json.dumps(check, indent=1), flush=True)

    summary = {"steps": STEPS, "families": FAMILIES, "n_units": int(n_units),
               "n_cols": {f: int(family_cols(V)[f].size) for f in FAMILIES},
               "lambda_selected": {str(s): lam[s] for s in STEPS},
               "positions_per_window": int(T - NFWD - NLAG),
               "n_windows": int(windows.shape[0]),
               "vocab_sizes": {"trigrams_kept": N_TRI, "words_kept": N_WORD},
               "reproduction_check": check,
               "group_sizes": {g: int(role[g].size) for g in GROUPS},
               "medians": {}, "change": {}, "gain_over_all40_step30000": {},
               "gain_over_all40_step831": {}}
    for f in FAMILIES:
        summary["medians"][f] = {str(s): {**{g: round(float(np.median(R2[s][f][role[g]])), 4)
                                             for g in GROUPS},
                                          "all_units": round(float(np.median(R2[s][f])), 4)}
                                 for s in STEPS}
        summary["change"][f] = {}
        for g in GROUPS + ["all_units"]:
            u = np.arange(n_units) if g == "all_units" else role[g]
            d = R2[STEPS[-1]][f][u] - R2[STEPS[0]][f][u]
            summary["change"][f][g] = {"median": round(float(np.median(d)), 4),
                                       "frac_down": round(float((d < 0).mean()), 3),
                                       "n": int(u.size),
                                       "p_wilcoxon": float(wilcoxon(d).pvalue)}
    for s, key in [(30000, "gain_over_all40_step30000"), (831, "gain_over_all40_step831")]:
        for g in GROUPS + ["all_units"]:
            u = np.arange(n_units) if g == "all_units" else role[g]
            d = R2[s]["union"][u] - R2[s]["all40"][u]
            summary[key][g] = {"median_union_minus_all40": round(float(np.median(d)), 4),
                               "frac_up": round(float((d > 0).mean()), 3),
                               "median_residual_all40": round(float(np.median(1 - R2[s]["all40"][u])), 4),
                               "median_residual_union": round(float(np.median(1 - R2[s]["union"][u])), 4),
                               "frac_of_residual_closed": round(float(
                                   np.median(d) / max(np.median(1 - R2[s]["all40"][u]), 1e-9)), 4),
                               "p_wilcoxon": float(wilcoxon(d).pvalue)}
    with open(os.path.join(RES, "neuron_probe_struct_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_probe_struct_raw.npz"),
                        steps=np.array(STEPS),
                        **{f"r2_{f}_{s}": R2[s][f] for s in STEPS for f in FAMILIES},
                        **{g: role[g] for g in GROUPS})
    plot(R2, role, n_units)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def self_test_light(windows, V, tabs, nl, sp, itos, device):
    """structural + bag features rebuilt in plain Python for 24 random rows (lexical codes are
    checked by construction: their vocabularies are exact frequency ranks)."""
    idx = torch.as_tensor(windows[:4], device=device)
    codes, bag = batch_codes(idx, V, *tabs, nl, sp)
    off, _, _ = offsets(V)
    names = [n for n, _ in groups_of(V)]
    at = {n: names.index(n) for n in names}
    Tp = T - NFWD - NLAG
    rng = np.random.default_rng(0)
    for _ in range(24):
        b, p = int(rng.integers(4)), int(rng.integers(NLAG, T - NFWD))
        row = b * Tp + (p - NLAG)
        s = "".join(itos[c] for c in windows[b])
        pre = s[:p + 1]
        nlpos = pre.rfind("\n")
        want = {"col": COLCAP + 1 if nlpos < 0 else min(p - nlpos, COLCAP)}
        seps = [i for i, c in enumerate(pre) if c in " \n"]
        want["wpos"] = WCAP + 1 if not seps else min(p - seps[-1], WCAP)
        want["prevwlen"] = WCAP + 1 if len(seps) < 2 else min(max(seps[-1] - seps[-2] - 1, 0), WCAP)
        nxt = [k for k in range(1, NFWD + 1) if s[p + k] in " \n"]
        want["nextsep"] = nxt[0] if nxt else 0
        r = 0
        while p - r >= 0 and s[p - r].isupper() and r < RUNCAP:
            r += 1
        want["capsrun"] = r
        for n, w in want.items():
            got = int(codes[row, at[n]])
            assert got == w, (n, p, got, w, repr(pre[-10:]))
        bg = bag[row].cpu().numpy()
        prefix = s[:p - NLAG + 1]
        assert abs(bg[-1] - len(prefix) / T) < 1e-9, (p, bg[-1])
        for c in set(prefix):
            assert abs(bg[STOI[c]] - prefix.count(c) / len(prefix)) < 1e-9, (p, c)
    print("  self-test: structural and bag features reproduce a plain-Python rebuild", flush=True)


def plot(R2, role, n_units):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))
    show = ["char1", "past8", "all40", "nonchar", "union"]
    x = np.arange(len(show))

    a = ax[0]
    for u, c, dx, lab in [(role["demoted"], CVD[0], -0.13, "demoted units"),
                          (np.arange(n_units), CVD[1], 0.13, "all 3,840 units")]:
        m = {s: np.array([np.median(R2[s][f][u]) for f in show]) for s in STEPS}
        a.vlines(x + dx, m[STEPS[-1]], m[STEPS[0]], color=c, lw=1.4)
        for s, mk, fill in zip(STEPS, ["o", "s"], [True, False]):
            a.plot(x + dx, m[s], linestyle="none", marker=mk, color=c, ms=7,
                   mfc=c if fill else "white", label=f"{lab}, step {s:,}")
    a.set_xticks(x); a.set_xticklabels(show, rotation=20)
    a.set_ylim(0, 1.3); a.set_xlabel("feature family the probe is fitted with")
    a.set_ylabel("median held-out $R^2$")
    a.set_title("(a) adding non-character features\nto the widest character family")
    a.legend(fontsize=7, loc="upper left", ncol=2); a.grid(alpha=0.3)

    b = ax[1]
    for g, c, mk in zip(GROUPS, CVD, MK):
        d = np.sort(R2[30000]["union"][role[g]] - R2[30000]["all40"][role[g]])
        b.step(d, np.arange(d.size) / d.size, where="post", color=c, lw=1.8,
               marker=mk, markevery=max(1, d.size // 8), ms=5,
               label=f"{g.replace('_','-')} (n={d.size})")
    b.axvline(0.0, color="0.35", lw=1.0)
    b.set_xlim(-0.02, 0.30)
    b.set_xlabel("per-unit $R^2$ gain of union over all40, step 30,000 (x clipped at 0.30)")
    b.set_ylabel("fraction of units at or below")
    b.set_title("(b) who the new features help")
    b.legend(fontsize=8, loc="upper left"); b.grid(alpha=0.3)

    d = ax[2]
    fams = ["char1", "past8", "all40", "struct", "lex", "bag", "nonchar", "union"]
    xg = np.arange(len(fams))
    for g, c, mk in zip(["demoted", "stable"], [CVD[0], CVD[2]], [MK[0], MK[2]]):
        med = [np.median(R2[30000][f][role[g]]) for f in fams]
        d.plot(xg, med, linestyle="none", marker=mk, color=c, ms=8, label=f"{g} units")
        for xi, mv in zip(xg, med):
            d.vlines(xi, 0, mv, color=c, lw=1.0, alpha=0.5)
    d.set_xticks(xg); d.set_xticklabels(fams, rotation=30)
    d.set_ylim(0, 1.05); d.set_ylabel("median held-out $R^2$ at step 30,000")
    d.set_xlabel("feature family")
    d.set_title("(c) each family on its own")
    d.legend(fontsize=8, loc="lower left"); d.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "neuron_probe_struct.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        z = np.load(os.path.join(RES, "neuron_probe_struct_raw.npz"))
        plot({s: {f: z[f"r2_{f}_{s}"] for f in FAMILIES} for s in STEPS},
             {g: z[g] for g in GROUPS}, z[f"r2_char1_{STEPS[0]}"].size)
    else:
        main()
