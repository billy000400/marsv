"""Operator feedback #5, item 2 — raw d(t) curves run in BOTH directions for well-trained pairs.

The all-pairs sweep stored curves for one endpoint order only (A = lower vocab index) and checked
symmetry with a scalar (|w(A,B) - w(B,A)|). This re-runs a handful of pairs whose BOTH endpoints are
well-trained (>= 1000 occurrences in the training split) in both orders through the identical frozen
code path (matthew_assay.run_pair, context "The house was ", 50 t, block-0 patch, final position) and
stores both raw curves so the asymmetry can simply be looked at.

Writes results/followup_asym.json (curves + widths).
"""
import os, sys, json, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from matthew_assay import run_pair, is_plateau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(RES, "checkpoints_grok_char", "ckpt_030000.pt")
CORPUS = "/tmp/tinyshakespeare.txt"
CONTEXT = "The house was "
N_T = 50
FREQ_MIN = 1000
PAIRS = [("e", "o"), ("t", "s"), (" ", "e"), (".", ","), ("T", "A"), ("a", ".")]


def main():
    torch.cuda.set_per_process_memory_fraction(0.225)
    torch.set_num_threads(2)
    device = "cuda"

    text = open(CORPUS, "rb").read().decode("utf-8")
    freq = collections.Counter(text[:int(0.9 * len(text))])
    stoi = torch.load(os.path.join(RES, "checkpoints", "ckpt_00000.pt"),
                      map_location="cpu", weights_only=False)["stoi"]

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ts = np.linspace(0.0, 1.0, N_T)
    seq = lambda c: np.array([stoi[x] for x in CONTEXT + c], dtype=np.int64)
    out = {"context": CONTEXT, "n_t": N_T, "freq_min": FREQ_MIN, "step": 30000, "pairs": []}
    for a, b in PAIRS:
        assert freq[a] >= FREQ_MIN and freq[b] >= FREQ_MIN, (a, b, freq[a], freq[b])
        fwd = run_pair(model, seq(a), seq(b), 0, ts, device)["d_logit"]
        bwd = run_pair(model, seq(b), seq(a), 0, ts, device)["d_logit"]
        w_f = is_plateau(ts, fwd)[1]
        w_b = is_plateau(ts, bwd)[1]
        out["pairs"].append({"A": a, "B": b, "freq_A": freq[a], "freq_B": freq[b],
                             "d_fwd": [round(float(v), 5) for v in fwd],
                             "d_bwd": [round(float(v), 5) for v in bwd],
                             "w_fwd": round(float(w_f), 4), "w_bwd": round(float(w_b), 4)})
        print(f"{a!r}->{b!r}: w_fwd={w_f:.3f} w_bwd={w_b:.3f}", flush=True)

    out["ts"] = [round(float(t), 5) for t in ts]
    json.dump(out, open(os.path.join(RES, "followup_asym.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
