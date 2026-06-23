"""S1 — collect GPT-2 residual-stream activations on FineWeb text.

resid_post for layer L = GPT2Model hidden_states[L+1]
(hidden_states[0] = embeddings; hidden_states[i] = output of block i-1).

Text is pulled from the HF datasets-server REST API (no `datasets` lib needed)
and cached to data/fineweb_texts.json. Activations stored RAW (no centering,
no unit-normalization) as float16; mean-centering happens at analysis time.
"""
import os, json, time, urllib.request
import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast

torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
# This torch build (cu130, sm_75+) has no kernels for the box's V100 (sm_70),
# so CUDA is present but unusable -> run on CPU. GPT-2 small is fine on CPU.
USE_CUDA = False
if USE_CUDA and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.45)

LAYERS = [0, 3, 6, 9, 11]
SEQ_LEN = 256
TARGET_PER_LAYER = 200_000
BATCH_SEQS = 16
N_TEXTS = 1500          # FineWeb docs to pull (enough for 200k tokens/layer)
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)
TEXT_CACHE = os.path.join(OUT, "fineweb_texts.json")

DS = "HuggingFaceFW%2Ffineweb"
CONFIG = "CC-MAIN-2013-20"


def fetch_texts(n):
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE) as f:
            texts = json.load(f)
        if len(texts) >= n:
            print(f"loaded {len(texts)} cached texts", flush=True)
            return texts[:n]
    texts = []
    offset = 0
    while len(texts) < n:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DS}"
               f"&config={CONFIG}&split=train&offset={offset}&length=100")
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=30))
                break
            except Exception as e:
                print(f"retry offset={offset} ({e})", flush=True)
                time.sleep(3)
        else:
            break
        for r in d["rows"]:
            t = r["row"].get("text", "")
            if t and len(t) >= 200:
                texts.append(t)
        offset += 100
        print(f"fetched {len(texts)} texts (offset {offset})", flush=True)
    with open(TEXT_CACHE, "w") as f:
        json.dump(texts, f)
    return texts[:n]


def main():
    dev = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
    t0 = time.time()
    texts = fetch_texts(N_TEXTS)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained("gpt2").to(dev).eval()
    if dev == "cuda":
        model = model.half()
    print(f"[{time.time()-t0:.0f}s] model on {dev}, {len(texts)} texts", flush=True)

    buffers = {L: [] for L in LAYERS}
    counts = {L: 0 for L in LAYERS}
    n_seqs = 0
    bs = BATCH_SEQS

    i = 0
    while i < len(texts) and min(counts.values()) < TARGET_PER_LAYER:
        batch = texts[i:i + bs]
        i += bs
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN, padding=True)
        ids = enc["input_ids"].to(dev)
        mask = enc["attention_mask"].to(dev)
        try:
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            i -= len(batch)
            print(f"OOM -> bs={bs}", flush=True)
            continue
        hs = out.hidden_states
        m = mask.bool()
        for L in LAYERS:
            sel = hs[L + 1][m]
            buffers[L].append(sel.float().cpu().numpy().astype(np.float16))
            counts[L] += sel.shape[0]
        n_seqs += ids.shape[0]
        if n_seqs % 160 == 0:
            print(f"[{time.time()-t0:.0f}s] seqs={n_seqs} min_count={min(counts.values())}", flush=True)
        del out, hs
    if dev == "cuda":
        torch.cuda.empty_cache()

    for L in LAYERS:
        arr = np.concatenate(buffers[L], axis=0)[:TARGET_PER_LAYER]
        np.save(os.path.join(OUT, f"acts_layer{L}.npy"), arr)
        print(f"saved layer{L}: {arr.shape} {arr.dtype}", flush=True)

    meta = {
        "model": "gpt2", "d_model": 768, "layers": LAYERS,
        "seq_len": SEQ_LEN, "n_seqs": n_seqs,
        "per_layer": {str(L): int(min(counts[L], TARGET_PER_LAYER)) for L in LAYERS},
        "dtype": "float16",
        "centering": "raw stored; mean-centered at analysis time",
        "normalized": False,
        "dataset": f"HuggingFaceFW/fineweb {CONFIG} via datasets-server REST (first {N_TEXTS} docs, len>=200)",
        "resid_def": "hidden_states[L+1] = block L output (resid_post); all token positions pooled",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "collection_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
