"""What do the trainable blocks COMPUTE? -- chord-linearization of individual MLP neurons.

Where the series stands: the sharpness is made by the MLPs of blocks 1-4 (gain intervention:
deleting them returns w 0.351 -> 0.796, the untrained value), it is not the next-character decision
and not endpoint plausibility (per-block scan), and freezing experiments only ever relocated it.
Nothing so far says WHAT those MLPs compute along the path -- only where the computation may live.

This probe asks the smallest mechanistic question that can be answered with forward passes:

    is the bend produced by a FEW neurons that switch along the path, or by thousands of small
    contributions?

The intervention is deliberately gentler than deleting an MLP. For a chosen set S of MLP hidden
units in blocks 1-4, we replace each unit's post-GeLU activation at the patched (final) position by
the CHORD of its own endpoint values:

    a_j(t)  ->  (1 - t) * a_j(0) + t * a_j(1)      for j in S, at every t

i.e. we keep the unit, keep its endpoint values, and delete only the part of its response that is
NONLINEAR in the interpolation parameter t. Two consequences make this a clean test:
  * at t = 0 and t = 1 the chord equals the true activation, so BOTH endpoints are reproduced
    exactly under any S (checked per pair: d(0) < 1e-3, d(1) > 1 - 1e-3), and the endpoints -- the
    things d(t) is measured against -- are untouched;
  * a straight-line d(t) after linearizing S means the whole bend was carried by S.

Selection of S, three modes, all at the same sizes k so they are directly comparable:
  pair    per-pair top-k by importance  imp_j = ||W_proj[:,j]|| * max_t |a_j(t) - chord_j(t)|
          (how much of the residual stream that unit moves off the chord);
  global  ONE fixed set of k units, ranked by importance averaged over the 150 pairs -- tests
          whether a single shared circuit accounts for every pair;
  random  k units drawn uniformly per pair -- the control that says the ranking, not the count of
          edited units, is what matters.

Same 150-pair subsample, shared context, block-0 interpolation and step-30000 character checkpoint
as mlp_gain_probe.py / mlp_block_scan.py, so every width is comparable to the rest of the report.
The untrained (step-0) width on the same pairs is measured in-run as the "straight line" reference.

Raw -> results/neuron_path_raw.npz, stats -> results/neuron_path_summary.json.
"""
import os, sys, json, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from matthew_assay import run_pair, is_plateau, pava_isotonic, iso_crossing, self_test
from allpairs_sweep import load_vocab, CONTEXT, N_T
from frozen_assay import load_ckpt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT_DIR = "/tmp/dir13_frozen/checkpoints_ref_pos"   # reference recipe, seed 1337 (see pos_assay.py)
BLOCK = 0             # interpolation block, as everywhere else in the report
EARLY = [1, 2, 3, 4]  # the blocks the gain intervention implicated
N_PAIRS = 150
SEED = 0
K_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # "all units" is appended at runtime


class ChordMLP(nn.Module):
    """Wraps a block's MLP. mode='record' stores the final-position post-GeLU activations;
    mode='ablate' overwrites the units in `idx` with the stored chord values."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.mode = "off"
        self.rec = None      # [K,H] recorded activations at the final position
        self.idx = None      # LongTensor of unit indices to linearize
        self.chord = None    # [K,H] chord of the recorded endpoints

    def forward(self, x):
        a = F.gelu(self.mlp.c_fc(x))
        if self.mode == "record":
            self.rec = a[:, -1, :].detach().clone()
        elif (self.mode == "ablate" and self.idx is not None and self.idx.numel()
              and a.shape[0] == self.chord.shape[0]):
            # run_pair first runs the two endpoint prompts (batch 2) to define x_A, x_B; that pass
            # must stay unablated, and the batch size distinguishes it from the K-step path batch.
            a = a.clone()
            a[:, -1, self.idx] = self.chord[:, self.idx].to(a.dtype)
        return self.mlp.drop(self.mlp.c_proj(a))


def install(model):
    wraps = {}
    for l in EARLY:
        w = ChordMLP(model.blocks[l].mlp)
        model.blocks[l].mlp = w
        wraps[l] = w
    return wraps


def uninstall(model, wraps):
    for l, w in wraps.items():
        model.blocks[l].mlp = w.mlp


def set_mode(wraps, mode):
    for w in wraps.values():
        w.mode = mode


def width(ts, d):
    ok, w, t_lo, t_hi, dev = is_plateau(ts, d)
    return w, bool(ok)


def record_pair(model, wraps, seqs, a, b, ts, device):
    """One unablated pass: baseline curve, per-unit chords, per-unit importance."""
    set_mode(wraps, "record")
    r = run_pair(model, seqs[a], seqs[b], BLOCK, ts, device, batch_k=len(ts))
    set_mode(wraps, "off")
    tt = torch.tensor(ts, dtype=torch.float32, device=device)[:, None]
    imp = []
    for l in EARLY:
        w = wraps[l]
        rec = w.rec.float()                                   # [K,H]
        chord = (1 - tt) * rec[0][None] + tt * rec[-1][None]   # [K,H]
        w.chord = chord
        with torch.no_grad():
            wnorm = torch.linalg.norm(w.mlp.c_proj.weight.float(), dim=0)  # [H], column norms
            imp.append((wnorm * (rec - chord).abs().max(dim=0).values).cpu().numpy())
    return r, np.concatenate(imp)   # imp: [4*H] in EARLY order


def ablate_pair(model, wraps, seqs, a, b, ts, device, sel, H):
    """Run the assay with the units in `sel` (flat indices over EARLY x H) linearized."""
    for i, l in enumerate(EARLY):
        block_sel = sel[(sel >= i * H) & (sel < (i + 1) * H)] - i * H
        wraps[l].idx = torch.as_tensor(block_sel, dtype=torch.long, device=device)
        wraps[l].mode = "ablate"
    r = run_pair(model, seqs[a], seqs[b], BLOCK, ts, device, batch_k=len(ts))
    set_mode(wraps, "off")
    return r


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"
    self_test()

    stoi = load_vocab()
    chars = sorted(stoi, key=lambda c: stoi[c])
    V = len(chars)
    ts = np.linspace(0.0, 1.0, N_T)
    seqs = [np.array([stoi[c] for c in CONTEXT + ch], dtype=np.int64) for ch in chars]
    all_pairs = list(itertools.combinations(range(V), 2))
    rng = np.random.default_rng(SEED)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]

    model, ck = load_ckpt(os.path.join(CKPT_DIR, "ckpt_030000.pt"), device)
    H = model.blocks[1].mlp.c_fc.out_features
    n_units = len(EARLY) * H
    global K_LIST
    K_LIST = [k for k in K_LIST if k < n_units] + [n_units]
    wraps = install(model)
    t0 = time.time()

    # ---- pass 1: baselines + importance --------------------------------------------------
    imps = np.zeros((N_PAIRS, n_units), dtype=np.float32)
    w_base = np.zeros(N_PAIRS)
    strict_base = np.zeros(N_PAIRS, dtype=bool)
    ep_worst = 0.0
    for i, (a, b) in enumerate(pairs):
        r, imp = record_pair(model, wraps, seqs, a, b, ts, device)
        imps[i] = imp
        w_base[i], strict_base[i] = width(ts, r["d_logit"])
        ep_worst = max(ep_worst, abs(r["d0"]), abs(1 - r["d1"]))
    print(f"baseline median_w={np.nanmedian(w_base):.4f} strict={strict_base.mean():.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # global ranking: per-pair importance normalized to its own max, then averaged
    norm_imp = imps / np.maximum(imps.max(axis=1, keepdims=True), 1e-12)
    global_rank = np.argsort(-norm_imp.mean(axis=0))

    # ---- pass 2: ablations ----------------------------------------------------------------
    modes = ["pair", "global", "random"]
    W = {m: np.full((len(K_LIST), N_PAIRS), np.nan) for m in modes}
    S = {m: np.zeros((len(K_LIST), N_PAIRS), dtype=bool) for m in modes}
    ep_ab = 0.0
    rng2 = np.random.default_rng(1234)
    top32 = np.zeros((N_PAIRS, 32), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        r, imp = record_pair(model, wraps, seqs, a, b, ts, device)   # refreshes the chords
        order = np.argsort(-imp)
        top32[i] = order[:32]
        for ki, k in enumerate(K_LIST):
            for m in modes:
                if m == "pair":
                    sel = order[:k]
                elif m == "global":
                    sel = global_rank[:k]
                else:
                    sel = rng2.choice(n_units, k, replace=False)
                ra = ablate_pair(model, wraps, seqs, a, b, ts, device, sel, H)
                W[m][ki, i], S[m][ki, i] = width(ts, ra["d_logit"])
                ep_ab = max(ep_ab, abs(ra["d0"]), abs(1 - ra["d1"]))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1}/{N_PAIRS} ({time.time()-t0:.0f}s)", flush=True)

    # ---- untrained reference on the same pairs ---------------------------------------------
    uninstall(model, wraps)
    del model
    torch.cuda.empty_cache()
    m0, _ = load_ckpt(os.path.join(CKPT_DIR, "ckpt_000000.pt"), device)
    w_init = np.array([width(ts, run_pair(m0, seqs[a], seqs[b], BLOCK, ts, device,
                                          batch_k=len(ts))["d_logit"])[0] for a, b in pairs])
    del m0
    torch.cuda.empty_cache()

    wb, wi = float(np.nanmedian(w_base)), float(np.nanmedian(w_init))
    gap = wi - wb

    def rec_frac(x):   # fraction of the trained->untrained width gap recovered
        return round(float((np.nanmedian(x) - wb) / gap), 4)

    # per-pair smallest k (pair mode) recovering >=50% of that pair's own gap
    k50 = []
    for i in range(N_PAIRS):
        target = w_base[i] + 0.5 * (w_init[i] - w_base[i])
        hit = [K_LIST[ki] for ki in range(len(K_LIST)) if W["pair"][ki, i] >= target]
        k50.append(hit[0] if hit else np.nan)
    k50 = np.array(k50, dtype=float)

    # reuse of the per-pair top-32 units across pairs, and which block they sit in
    counts = np.bincount(top32.ravel(), minlength=n_units)
    blocks_of_top32 = [int(((top32 // H) == i).sum()) for i in range(len(EARLY))]

    summary = {
        "context": CONTEXT, "step": 30000, "block": BLOCK, "n_t": N_T, "n_pairs": N_PAIRS,
        "ckpt_dir": CKPT_DIR, "early_blocks": EARLY, "hidden_per_block": H, "n_units": n_units,
        "k_list": K_LIST,
        "baseline": {"median_w": round(wb, 4), "strict_frac": round(float(strict_base.mean()), 4)},
        "untrained": {"median_w": round(wi, 4)},
        "endpoint_check": {"worst_baseline": round(ep_worst, 6), "worst_ablated": round(ep_ab, 6)},
        "curves": {m: {"median_w": [round(float(np.nanmedian(W[m][ki])), 4) for ki in range(len(K_LIST))],
                       "iqr_lo": [round(float(np.nanpercentile(W[m][ki], 25)), 4) for ki in range(len(K_LIST))],
                       "iqr_hi": [round(float(np.nanpercentile(W[m][ki], 75)), 4) for ki in range(len(K_LIST))],
                       "recovered_frac": [rec_frac(W[m][ki]) for ki in range(len(K_LIST))],
                       "strict_frac": [round(float(S[m][ki].mean()), 4) for ki in range(len(K_LIST))]}
                   for m in modes},
        "k50": {"median": float(np.nanmedian(k50)), "q25": float(np.nanpercentile(k50, 25)),
                "q75": float(np.nanpercentile(k50, 75)),
                "frac_defined": round(float(np.isfinite(k50).mean()), 4)},
        "reuse_top32": {
            "n_distinct_units": int((counts > 0).sum()),
            "max_pairs_one_unit": int(counts.max()),
            "frac_units_in_1_pair": round(float((counts == 1).sum() / max((counts > 0).sum(), 1)), 4),
            "median_overlap_with_global32": round(float(np.median(
                [len(set(top32[i]) & set(global_rank[:32])) for i in range(N_PAIRS)])), 2),
            "blocks_of_top32": blocks_of_top32},
    }
    with open(os.path.join(RES, "neuron_path_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RES, "neuron_path_raw.npz"),
                        pairs=np.array(pairs), k_list=np.array(K_LIST), w_base=w_base,
                        w_init=w_init, top32=top32, counts=counts,
                        global_rank32=global_rank[:32], imp_mean=norm_imp.mean(axis=0),
                        k50=k50, **{f"w_{m}": W[m] for m in modes},
                        **{f"strict_{m}": S[m] for m in modes})
    print(json.dumps(summary["curves"]["pair"], indent=2))
    print(json.dumps(summary["k50"], indent=2))
    print(json.dumps(summary["reuse_top32"], indent=2))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
