"""S24 — interpolate a token that is NOT the position the readout reads.

Every plateau number in this study patches the *final* sequence position and reads the *final*
logits, exactly as Matthew's assay does. That leaves one control untested: is the sharp transition a
property of the readout position itself, or does an interpolated token still flip the network's
downstream state sharply when the readout sits k tokens later?

The two prompts here differ at position `POS = len(CONTEXT)` and then share a fixed filler suffix,
so the varied character sits k characters before the end:

    "The house was b" + suffix_k          suffix_k = the first k characters of " and then"
    "The house was i" + suffix_k

Hook. Patching resid_post of block 0 at a non-final position is NOT exact: the later positions'
block-0 residuals were computed from prompt A's token at POS and cannot be recomputed from the
patched vector, so t=1 would not reproduce prompt B. The one injection point that keeps both
endpoint identities exact for every k is the residual stream *entering* block 0 (token + position
embedding). Everything downstream — every block, every position >= POS — then recomputes from the
interpolated vector. We therefore inject there, at POS, for every k including k=0, so the k-axis is
internally consistent; k=0 with the block-0 hook is also run separately as the anchor that ties this
sweep to every existing number in the report.

Two readouts come free from the same forward pass:
  read_final — logits at the last position (k characters after the patch): the actual question;
  read_patch — logits at POS itself. Causal attention makes this independent of the suffix, so it
               must be identical for every k, and equals the k=0 condition. A built-in check.

Scoring is the frozen pipeline: slerp_rescale over 50 evenly spaced t, relative distance
d(t) = |x(t)-x_A| / (|x(t)-x_A| + |x(t)-x_B|) in logit space, w_10->90 on the isotonic copy, the
same 150 character pairs and seed as every frozen-condition row.

results/pos_assay.json (+ pos_assay_raw.npz). Figure: plot_pos.py.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import slerp_norm, rel_dist, run_pair, is_plateau, iso_crossing, pava_isotonic, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from mlp_gain_probe import N_PAIRS, SEED
from frozen_assay import load_ckpt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = "/tmp/dir13_frozen/checkpoints_ref_pos"
POS = len(CONTEXT)                 # index of the varied character (the context is 14 chars)
FILLER = " and then"               # suffix_k = FILLER[:k]
KS = [0, 1, 2, 4, 8]
BLOCK = 0                          # anchor run only (final position, resid_post of block 0)


@torch.no_grad()
def run_pair_emb(model, seqA, seqB, pos, ts, device):
    """Interpolate the residual stream entering block 0 at `pos`; read out at the last position and
    at `pos`. seqA/seqB must be identical except at index `pos`."""
    idx = torch.from_numpy(np.stack([seqA, seqB])).to(device)
    T = idx.shape[1]
    pos_ids = torch.arange(T, device=device)
    E = model.tok_emb(idx) + model.pos_emb(pos_ids)[None]          # [2,T,C]  (eval -> no dropout)
    _, clean = model.forward_from_block(E, 0)                      # [2,T,V]
    H = slerp_norm(E[0, pos].clone(), E[1, pos].clone(),
                   torch.tensor(ts, dtype=E.dtype, device=device))  # [K,C]
    X = E[0][None].repeat(len(ts), 1, 1)
    X[:, pos, :] = H
    _, lg = model.forward_from_block(X, 0)                          # [K,T,V]

    out = {"ep_err": max(float((lg[0, -1] - clean[0, -1]).abs().max()),
                         float((lg[-1, -1] - clean[1, -1]).abs().max()))}
    for name, rp in (("final", T - 1), ("patch", pos)):
        out[name] = {"d": rel_dist(lg[:, rp, :], clean[0, rp], clean[1, rp]),
                     "argmax": lg[:, rp, :].argmax(-1).cpu().numpy(),
                     "sep": float(torch.linalg.norm(clean[0, rp] - clean[1, rp])),
                     "differ": bool(clean[0, rp].argmax() != clean[1, rp].argmax())}
    return out


def rho_or_none(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return None
    return round(float(spearmanr(x[m], y[m]).statistic), 4)


def score(ts, d):
    ok, w, t_lo, t_hi, dev = is_plateau(ts, d)
    return (np.nan if not np.isfinite(w) else float(w), bool(ok),
            float(iso_crossing(ts, pava_isotonic(d), 0.5)))


def summarize(ts, per_pair, key):
    w = np.array([p[key]["w"] for p in per_pair])
    return {"median_w": round(float(np.nanmedian(w)), 4),
            "mean_w": round(float(np.nanmean(w)), 4),
            "iqr_w": [round(float(np.nanpercentile(w, 25)), 4),
                      round(float(np.nanpercentile(w, 75)), 4)],
            "strict_frac": round(float(np.mean([p[key]["strict"] for p in per_pair])), 4),
            "median_t_star": round(float(np.nanmedian([p[key]["tstar"] for p in per_pair])), 4),
            "median_sep": round(float(np.median([p[key]["sep"] for p in per_pair])), 4),
            "frac_endpoints_differ": round(float(np.mean([p[key]["differ"] for p in per_pair])), 4),
            "n": int(len(per_pair))}


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    ts = np.linspace(0.0, 1.0, N_T)
    all_pairs = list(itertools.combinations(range(len(chars)), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    out = {"context": CONTEXT, "pos": POS, "filler": FILLER, "ks": KS, "n_t": N_T,
           "n_pairs": N_PAIRS, "seed": SEED, "hook": "resid entering block 0 (tok+pos embedding)",
           "anchor_hook": f"resid_post block {BLOCK}, final position (the report's standard assay)",
           "checkpoints": {}}
    raw = {"ts": ts}

    for which, fn in (("init", "ckpt_000000.pt"), ("matched", "ckpt_matched.pt"),
                      ("last", "ckpt_last.pt")):
        path = os.path.join(CKPT_DIR, fn)
        if not os.path.exists(path):
            print(f"[skip] {path} missing", flush=True)
            continue
        model, ck = load_ckpt(path, device)
        cond = {"step": int(ck["step"]), "val_acc": ck.get("val_acc"), "conditions": {}}
        t0 = time.time()

        # anchor: the report's standard assay (resid_post block 0, final position, no suffix)
        seqs0 = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
        aw, astrict = [], []
        for ia, ib in pairs:
            r = run_pair(model, seqs0[ia], seqs0[ib], BLOCK, ts, device)
            w, ok, _ = score(ts, r["d_logit"])
            aw.append(w); astrict.append(ok)
        cond["anchor_block0_final"] = {"median_w": round(float(np.nanmedian(aw)), 4),
                                       "strict_frac": round(float(np.mean(astrict)), 4),
                                       "n": len(aw)}
        print(f"[{which}] anchor (block0, final position): median w = "
              f"{cond['anchor_block0_final']['median_w']}", flush=True)

        for k in KS:
            suf = FILLER[:k]
            seqs = [np.array([stoi[c] for c in CONTEXT + ch + suf], dtype=np.int64) for ch in chars]
            per_pair, ep = [], 0.0
            dcurves = {"final": [], "patch": []}
            for ia, ib in pairs:
                r = run_pair_emb(model, seqs[ia], seqs[ib], POS, ts, device)
                ep = max(ep, r["ep_err"])
                rec = {}
                for key in ("final", "patch"):
                    w, ok, tstar = score(ts, r[key]["d"])
                    rec[key] = {"w": w, "strict": ok, "tstar": tstar,
                                "sep": r[key]["sep"], "differ": r[key]["differ"]}
                    dcurves[key].append(r[key]["d"].astype(np.float32))
                per_pair.append(rec)
            wf = np.array([p["final"]["w"] for p in per_pair])
            sepf = np.array([p["final"]["sep"] for p in per_pair])
            row = {"k": k, "suffix": suf, "seq_len": len(CONTEXT) + 1 + k,
                   "max_endpoint_err": round(ep, 6),
                   "read_final": summarize(ts, per_pair, "final"),
                   "read_patch": summarize(ts, per_pair, "patch"),
                   "rho_w_vs_sep": rho_or_none(wf, sepf)}
            if k > 0:                       # paired against the same pairs read at their own position
                base = raw[f"{which}|k0|w_final"]
                d = wf - base
                m = np.isfinite(d)
                row["paired_vs_k0"] = {
                    "median_dw": round(float(np.nanmedian(d)), 4),
                    "frac_wider": round(float(np.mean(d[m] > 0)), 4),
                    "p_wilcoxon": float(f"{wilcoxon(wf[m], base[m]).pvalue:.3g}"), "n": int(m.sum())}
            cond["conditions"][f"k{k}"] = row
            raw[f"{which}|k{k}|final"] = np.stack(dcurves["final"])
            raw[f"{which}|k{k}|patch"] = np.stack(dcurves["patch"])
            raw[f"{which}|k{k}|w_final"] = wf
            raw[f"{which}|k{k}|w_patch"] = np.array([p["patch"]["w"] for p in per_pair])
            raw[f"{which}|k{k}|sep_final"] = sepf
            print(f"[{which}] k={k} (len {row['seq_len']}): read_final median w="
                  f"{row['read_final']['median_w']} strict={row['read_final']['strict_frac']} "
                  f"sep={row['read_final']['median_sep']} | read_patch median w="
                  f"{row['read_patch']['median_w']} | ep_err={ep:.2e} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        out["checkpoints"][which] = cond
        del model
        torch.cuda.empty_cache()

    # trained-vs-untrained at every k: is the transported sharpness a training effect, or just what
    # this prompt geometry gives any network of this shape?
    out["trained_vs_init"] = {}
    for which in ("matched", "last"):
        if which not in out["checkpoints"] or "init" not in out["checkpoints"]:
            continue
        rows = {}
        for k in KS:
            a, b = raw[f"{which}|k{k}|w_final"], raw[f"init|k{k}|w_final"]
            m = np.isfinite(a) & np.isfinite(b)
            rows[f"k{k}"] = {"median_dw": round(float(np.median(a[m] - b[m])), 4),
                             "frac_wider": round(float(np.mean(a[m] > b[m])), 4),
                             "p_wilcoxon": float(f"{wilcoxon(a[m], b[m]).pvalue:.3g}"),
                             "n": int(m.sum())}
        out["trained_vs_init"][which] = rows

    json.dump(out, open(os.path.join(RES, "pos_assay.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, "pos_assay_raw.npz"), **raw)
    print("DONE -> results/pos_assay.json", flush=True)


if __name__ == "__main__":
    main()
