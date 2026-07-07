"""
Collect Qwen3-1.7B last-token resid_post activations at layers 2 and 10,
reproducing the colleague's AE setup (seq_len=10, last-token-only).

Data: the already-cached FineWeb documents (data/fineweb_texts.json), chunked
into non-overlapping seq_len=10 windows; each window is fed as its own length-10
sequence (RoPE positions 0..9) and we keep the LAST token's residual (position 9),
exactly matching colleague `train_autoencoder.py` (`hidden[:, -1, :]`).

Colleague uses HuggingFaceFW/fineweb-edu; we use the cached HuggingFaceFW/fineweb
docs (corpus is factor #4 — noted, controlled separately if needed).

Saves fp16 [N,2048] arrays to ae_study/cache/qwen_l{2,10}.npy.
"""
import os, json, time
import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.180)
torch.set_num_threads(1)

DIR = "/mars-vol/marsv/dir3_manifold"
OUT = os.path.join(DIR, "ae_study/cache")
os.makedirs(OUT, exist_ok=True)

SEQ_LEN = 10
TARGET = 150_000          # last-token activations to collect
LAYERS = {2: 3, 10: 11}   # layer L -> hidden_states index (L+1)
KEEP_UP_TO = 10           # truncate model to first 11 blocks (need layer 10)

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
# truncate to the first KEEP_UP_TO+1 transformer blocks (saves compute + memory)
model.model.layers = model.model.layers[:KEEP_UP_TO + 1]
model.eval().cuda()

texts = json.load(open(os.path.join(DIR, "data/fineweb_texts.json")))
print(f"{len(texts)} source docs")

# Build seq_len=10 windows from tokenized docs
windows = []
for t in texts:
    ids = tok(t, add_special_tokens=True)["input_ids"]
    for s in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
        windows.append(ids[s:s + SEQ_LEN])
        if len(windows) >= TARGET:
            break
    if len(windows) >= TARGET:
        break
print(f"{len(windows)} windows of len {SEQ_LEN}")

acts = {L: [] for L in LAYERS}
BATCH = 128
t0 = time.time()
for i in range(0, len(windows), BATCH):
    batch = torch.tensor(windows[i:i + BATCH], device="cuda")
    with torch.no_grad():
        out = model(batch, output_hidden_states=True)
    for L, idx in LAYERS.items():
        h = out.hidden_states[idx][:, -1, :].float().cpu().numpy().astype(np.float16)
        acts[L].append(h)
    if (i // BATCH) % 50 == 0:
        done = i + BATCH
        rate = done / (time.time() - t0 + 1e-9)
        print(f"  {done}/{len(windows)} ({rate:.0f}/s)", flush=True)
    del out
print()

for L in LAYERS:
    arr = np.concatenate(acts[L], axis=0)
    p = os.path.join(OUT, f"qwen_l{L}.npy")
    np.save(p, arr)
    n = arr.shape[0]
    # per-doc mean norm sanity
    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    print(f"layer {L}: {arr.shape} saved {p}  mean_norm={norms.mean():.2f}")

meta = {"seq_len": SEQ_LEN, "n": int(arr.shape[0]), "layers": list(LAYERS.keys()),
        "source": "HuggingFaceFW/fineweb (cached docs) chunked into seq_len=10 windows, last token",
        "model": "Qwen/Qwen3-1.7B", "d_model": 2048}
json.dump(meta, open(os.path.join(OUT, "collect_meta.json"), "w"), indent=2)
print("done", time.time() - t0, "s")
