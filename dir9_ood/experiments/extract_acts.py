"""Extract GPT-2 last-token activations for the cupbearer evaluation.

Runs in the BASE env (working transformers + gpt2 cache). Produces per-split, per-point
.npy arrays under results/acts/ that the isolated cupenv then feeds to the REAL cupbearer
detectors (cup_eval.py). This keeps the heavy GPT-2 forward pass in the env that already
has it, and the cupbearer package isolated — no transformers needed in cupenv.

Splits: idtrain (fit/trusted, 1000 seqs), idtest (200), random/shuffled/code (200 each).
Points: input, resid3, resid6, resid9 (last-token activation). Also saves MSP per split as a
reference baseline. seq_len=64, matching plateau_v2.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import plateau_score as ps

torch.set_num_threads(2)
torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45)
ps.DEVICE = DEVICE   # plateau_score hardcodes "cpu"; run GPT-2 on the GPU like plateau_v2

POINTS = ["input", "resid3", "resid6", "resid9"]
SEQ = 64
N_OOD = 200
N_IDTEST = 200
N_FIT = 1000
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "acts")
os.makedirs(OUT, exist_ok=True)


def _hs_index(point):
    return 0 if point == "input" else int(point[5:]) + 1


@torch.no_grad()
def collect(ids, bs=32):
    """One batched forward -> {point: [N,d] float32}, msp[N]. Last-token activation."""
    model, _ = ps.load_model()
    dev = next(model.parameters()).device
    pos = SEQ - 1
    idx = {p: _hs_index(p) for p in POINTS}
    acts = {p: [] for p in POINTS}
    msp = []
    for i in range(0, ids.shape[0], bs):
        b = ids[i:i + bs].to(dev)
        out = model(b, output_hidden_states=True)
        for p in POINTS:
            acts[p].append(out.hidden_states[idx[p]][:, pos].float().cpu())
        msp.append((1.0 - torch.softmax(out.logits[:, pos], -1).max(-1).values).cpu())
        del out
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {p: torch.cat(acts[p]).numpy().astype(np.float32) for p in POINTS}, torch.cat(msp).numpy()


def save_split(name, ids):
    acts, msp = collect(ids)
    for p in POINTS:
        np.save(os.path.join(OUT, f"{name}__{p}.npy"), acts[p])
    np.save(os.path.join(OUT, f"{name}__msp.npy"), msp.astype(np.float32))
    print(f"  {name}: {ids.shape[0]} seqs, dims " + ", ".join(f"{p}={acts[p].shape[1]}" for p in POINTS), flush=True)


def main():
    ps.load_model()
    texts = [l.rstrip("\n") for l in open(os.path.join(os.path.dirname(__file__), "..", "data", "fineweb_sample.txt"))]
    id_all = ps.encode_fixed(texts, seq_len=SEQ)
    print(f"FineWeb encoded: {id_all.shape[0]} seqs of {SEQ} tokens", flush=True)
    need = N_FIT + N_IDTEST
    assert id_all.shape[0] >= need, f"need {need} ID seqs, have {id_all.shape[0]}"
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(id_all.shape[0], generator=g)
    id_fit = id_all[perm[:N_FIT]]
    id_test = id_all[perm[N_FIT:N_FIT + N_IDTEST]]
    # OOD built from the held-out ID test ids so shapes/positions match.
    print("Collecting activations...", flush=True)
    save_split("idtrain", id_fit)
    save_split("idtest", id_test)
    for kind in ["random", "shuffled", "code"]:
        ood = ps.make_ood(id_test, kind, seed=0)
        save_split(kind, ood)
    print("DONE extract", flush=True)


if __name__ == "__main__":
    main()
