"""Operator ask 2026-07-08 #2: the layer-6 AE reconstructs POOLED (all-token)
activations, which may carry "too much information" and blur any elbow. Collect
LAST-token GPT-2 layer-6 activations instead, so the AE sweep can be re-run on the
per-sequence final-token representation (matching the Qwen study's last-token setup).

resid_post for layer 6 = GPT2Model hidden_states[7]. We reuse the cached FineWeb
docs (data/fineweb_texts.json), tokenize each, split into non-overlapping windows of
SEQ_LEN tokens, keep only full windows, and store the hidden state at the FINAL
position of each window (one point per window). Raw fp16, mean-centered at analysis
time — identical storage convention to the pooled collection.
"""
import os, json, time
import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast

torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "1")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.18)
LAYER = 6
SEQ_LEN = 64
TARGET = 30_000
BATCH_SEQS = 64
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data")
TEXT_CACHE = os.path.join(OUT, "fineweb_texts.json")


def main():
    t0 = time.time()
    texts = json.load(open(TEXT_CACHE))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(DEV).eval()
    print(f"[{time.time()-t0:.0f}s] {len(texts)} texts, seq_len={SEQ_LEN}, dev={DEV}", flush=True)

    # tokenize + chunk into full SEQ_LEN windows
    windows = []
    for t in texts:
        ids = tok(t, truncation=False)["input_ids"]
        for s in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            windows.append(ids[s:s + SEQ_LEN])
            if len(windows) >= TARGET:
                break
        if len(windows) >= TARGET:
            break
    print(f"[{time.time()-t0:.0f}s] {len(windows)} full windows", flush=True)

    buf = []
    for i in range(0, len(windows), BATCH_SEQS):
        batch = windows[i:i + BATCH_SEQS]
        ids = torch.tensor(batch).to(DEV)              # (b, SEQ_LEN), all full
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        last = out.hidden_states[LAYER + 1][:, -1, :]  # (b, 768) final position
        buf.append(last.float().cpu().numpy().astype(np.float16))
        if (i // BATCH_SEQS) % 40 == 0:
            print(f"[{time.time()-t0:.0f}s] {i+len(batch)}/{len(windows)}", flush=True)
    arr = np.concatenate(buf, 0)
    np.save(os.path.join(OUT, f"acts_layer{LAYER}_lasttoken.npy"), arr)
    meta = {"model": "gpt2", "layer": LAYER, "seq_len": SEQ_LEN,
            "n_points": int(arr.shape[0]), "dtype": "float16",
            "pooling": "LAST token of each non-overlapping window",
            "resid_def": "hidden_states[7] final position"}
    json.dump(meta, open(os.path.join(OUT, "lasttoken_meta.json"), "w"), indent=2)
    print(f"[{time.time()-t0:.0f}s] saved {arr.shape} {arr.dtype}", flush=True)


if __name__ == "__main__":
    main()
