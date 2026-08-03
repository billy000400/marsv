"""Experiment 5 / S9 — interpolate EVERY character to EVERY other character.

Generalises `comma_sweep.py` from "the comma vs the other 64" to all C(65,2)=2080 unordered
character pairs, using the identical frozen code path (`matthew_assay.run_pair`): shared context
"The house was ", 50 evenly spaced t including both endpoints, `slerp_rescale` on the final-position
resid_post of the interpolation block, patch only the final position, relative distance d(t) in
final-logit space, w_{10->90} on the isotonic copy.

Conditions (all on the fresh character GPT):
  (1) final checkpoint (step 30000), interpolation block 0, all 2080 pairs  -- the main sweep;
  (2) init checkpoint (step 0),      interpolation block 0, all 2080 pairs  -- learned-vs-init control;
  (3) 200-pair subsample at blocks 4, 8, 11 on the final ckpt                -- depth control;
  (4) 100 pairs re-run with endpoints swapped                                -- symmetry check.

Per pair we also record the argmax character at every t (readout-decision test) and the model's
next-character probabilities for both endpoints (plausibility confound).

Raw curves -> results/allpairs_raw.npz; per-pair/per-character stats -> results/allpairs_summary.json.
Analysis (variance decomposition, correlations) lives in analyze_allpairs.py; plots in plot_allpairs.py.
"""
import os, sys, json, time, itertools, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, transition_width, is_plateau, pava_isotonic, iso_crossing, self_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(RES, "checkpoints_grok_char")
CORPUS = "/tmp/tinyshakespeare.txt"
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CONTEXT = "The house was "
N_T = 50
FINAL_STEP = 30000
INIT_STEP = 0
DEPTH_BLOCKS = [4, 8, 11]
N_DEPTH = 200
N_SWAP = 100
SEED = 0


def load_vocab():
    """Character vocabulary of the tinyshakespeare corpus. Prefer a pilot checkpoint's saved stoi;
    if the scratch checkpoints have been wiped, rebuild it from the corpus, which every run built as
    sorted(set(text)) -- identical as long as the corpus SHA matches the one the runs recorded."""
    ck_path = os.path.join(RES, "checkpoints", "ckpt_00000.pt")
    if os.path.exists(ck_path):
        return torch.load(ck_path, map_location="cpu", weights_only=False)["stoi"]
    raw = open(CORPUS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == CORPUS_SHA, f"corpus SHA {sha} != training corpus {CORPUS_SHA}"
    stoi = {c: i for i, c in enumerate(sorted(set(raw.decode("utf-8"))))}
    assert len(stoi) == 65, f"vocab size {len(stoi)} != 65"
    return stoi


def load_model(step, device):
    ck = torch.load(os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pt"),
                    map_location=device, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def pair_stats(ts, r):
    """All per-pair scalars, from the one d(t) curve + its argmax trace."""
    ok, w, t_lo, t_hi, dev = is_plateau(ts, r["d_logit"])
    t_star = iso_crossing(ts, pava_isotonic(r["d_logit"]), 0.5)
    am = r["argmax"]
    flips = np.nonzero(am[1:] != am[:-1])[0]
    # first argmax flip, placed midway between the two grid points that bracket it
    t_flip = float(0.5 * (ts[flips[0]] + ts[flips[0] + 1])) if len(flips) else np.nan
    return {"w": None if not np.isfinite(w) else round(float(w), 4),
            "t_lo": round(float(t_lo), 4) if np.isfinite(t_lo) else None,
            "t_hi": round(float(t_hi), 4) if np.isfinite(t_hi) else None,
            "t_star": round(float(t_star), 4) if np.isfinite(t_star) else None,
            "plateau": bool(ok), "iso_dev": round(float(dev), 4),
            "n_argmax": int(len(np.unique(am))), "n_flips": int(len(flips)),
            "t_flip": round(t_flip, 4) if np.isfinite(t_flip) else None,
            "am0": int(am[0]), "am1": int(am[-1]),
            "d0": round(r["d0"], 6), "d1": round(r["d1"], 6),
            "ep_err": round(max(r["endpoint_err"].values()), 6),
            "prefix_err": round(r["prefix_err"], 8)}


def sweep(model, pairs, seqs, block, ts, device, raw, tag, store_raw=True):
    """Run every pair at one interpolation block; returns list of per-pair stat dicts."""
    out = []
    t0 = time.time()
    for n, (i, j) in enumerate(pairs):
        r = run_pair(model, seqs[i], seqs[j], block, ts, device)
        s = pair_stats(ts, r)
        s["i"], s["j"] = int(i), int(j)
        out.append(s)
        if store_raw:
            raw[f"{tag}|d|{i}_{j}"] = r["d_logit"].astype(np.float32)
            raw[f"{tag}|am|{i}_{j}"] = r["argmax"].astype(np.int16)
        if (n + 1) % 500 == 0:
            print(f"  {tag}: {n+1}/{len(pairs)} pairs ({time.time()-t0:.0f}s)", flush=True)
    ws = [s["w"] for s in out if s["w"] is not None]
    print(f"{tag}: n={len(out)} median w={np.median(ws):.3f} "
          f"strict {sum(s['plateau'] for s in out)}/{len(out)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    assert V == 65, V
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    pairs = list(itertools.combinations(range(V), 2))          # A = lower vocab index
    print(f"{V} characters, {len(pairs)} unordered pairs", flush=True)

    raw = {"ts": ts}
    summary = {"context": CONTEXT, "n_t": N_T, "chars": chars, "vocab_size": V,
               "final_step": FINAL_STEP, "init_step": INIT_STEP, "n_pairs": len(pairs)}

    # (1) main sweep: final checkpoint, interpolation block 0
    model = load_model(FINAL_STEP, device)
    summary["final_block0"] = sweep(model, pairs, seqs, 0, ts, device, raw, "final|L0")

    # (4) swap-symmetry check on 100 random pairs (d(t) is not symmetric by construction)
    rng = np.random.default_rng(SEED)
    swap_idx = rng.choice(len(pairs), size=N_SWAP, replace=False)
    swapped = [(pairs[k][1], pairs[k][0]) for k in swap_idx]
    summary["swap"] = sweep(model, swapped, seqs, 0, ts, device, raw, "final|L0|swap", store_raw=False)
    summary["swap_of"] = [[int(pairs[k][0]), int(pairs[k][1])] for k in swap_idx]

    # (3) depth control: same 200 random pairs at interpolation blocks 4, 8, 11
    dep_idx = rng.choice(len(pairs), size=N_DEPTH, replace=False)
    dep_pairs = [pairs[k] for k in dep_idx]
    summary["depth_pairs"] = [[int(a), int(b)] for a, b in dep_pairs]
    summary["depth"] = {}
    for L in DEPTH_BLOCKS:
        summary["depth"][str(L)] = sweep(model, dep_pairs, seqs, L, ts, device, raw,
                                         f"final|L{L}", store_raw=False)

    # per-character plausibility and endpoint separation at the final checkpoint
    with torch.no_grad():
        ctx = torch.tensor([[stoi[c] for c in CONTEXT]], device=device)
        p_next = torch.softmax(model(ctx)[0][0, -1], dim=-1).cpu().numpy()
        X = torch.from_numpy(np.stack(seqs)).to(device)
        lg = model(X)[0][:, -1, :]                              # [V, V] endpoint logit vectors
        sep = torch.cdist(lg, lg).cpu().numpy()
    summary["p_next"] = [float(p_next[stoi[c]]) for c in chars]
    summary["logit_sep"] = [float(sep[i, j]) for i, j in pairs]
    del model
    torch.cuda.empty_cache()

    # (2) learned-vs-init control: the identical 2080-pair sweep at step 0
    model = load_model(INIT_STEP, device)
    summary["init_block0"] = sweep(model, pairs, seqs, 0, ts, device, raw, "init|L0")
    del model
    torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(RES, "allpairs_raw.npz"), **raw)
    json.dump(summary, open(os.path.join(RES, "allpairs_summary.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
