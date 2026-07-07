"""Collect Qwen3-1.7B last-token residual-stream activations on FineWeb-Edu.

Replicates the colleague's AE setup (autoencoder_share.tar.gz):
  - model Qwen3-1.7B (d_model=2048, 28 layers)
  - layers 2 and 10  -> hidden_states[3] and hidden_states[11]
  - LAST TOKEN ONLY of each seq_len=10 sequence
  - data: HuggingFaceFW/fineweb-edu, sample-10BT

Data is pulled via the HF datasets-server REST API (no `datasets` lib needed),
cached to cache/fineweb_edu_texts.json. Each document is tokenized with the Qwen
tokenizer and split into non-overlapping 10-token chunks; each chunk is run as an
independent seq_len=10 sequence (positions 0..9) and we keep the position-9
(last-token) activation. (Deviation from the colleague's one-sequence-per-doc:
chunks start at arbitrary document offsets, not always doc-initial — far more
data-efficient; the "seq_len=10, last-token, RoPE positions 0..9" character is
identical.) Activations saved RAW as float16; centering happens at train time.
"""
import os, json, time, urllib.request
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.180)
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
TEXT_CACHE = os.path.join(CACHE, "fineweb_edu_texts.json")

SEQ_LEN = 10
LAYERS = {2: 3, 10: 11}          # layer -> hidden_states index
TARGET = 160_000                 # activations per layer
N_DOCS = 2500                    # fineweb-edu docs to pull
BATCH = 64
DS = "HuggingFaceFW%2Ffineweb-edu"
CONFIG = "sample-10BT"


def fetch_texts(n):
    if os.path.exists(TEXT_CACHE):
        texts = json.load(open(TEXT_CACHE))
        if len(texts) >= n:
            print(f"loaded {len(texts)} cached texts", flush=True)
            return texts[:n]
    texts, offset = [], 0
    while len(texts) < n:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DS}"
               f"&config={CONFIG}&split=train&offset={offset}&length=100")
        for _ in range(4):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=30)); break
            except Exception as e:
                print("retry", offset, e, flush=True); time.sleep(3)
        else:
            break
        for r in d["rows"]:
            t = r["row"].get("text", "")
            if t and len(t) >= 200:
                texts.append(t)
        offset += 100
        if offset % 500 == 0:
            print(f"fetched {len(texts)} texts (offset {offset})", flush=True)
    json.dump(texts, open(TEXT_CACHE, "w"))
    return texts[:n]


def main():
    t0 = time.time()
    texts = fetch_texts(N_DOCS)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("cuda").eval()
    print(f"[{time.time()-t0:.0f}s] model loaded", flush=True)

    # tokenize + chunk into seq_len=10 sequences
    chunks = []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        for s in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            chunks.append(ids[s:s + SEQ_LEN])
            if len(chunks) >= TARGET:
                break
        if len(chunks) >= TARGET:
            break
    print(f"[{time.time()-t0:.0f}s] {len(chunks)} seq_len={SEQ_LEN} chunks", flush=True)

    buf = {L: [] for L in LAYERS}
    for i in range(0, len(chunks), BATCH):
        batch = torch.tensor(chunks[i:i + BATCH], device="cuda")
        with torch.no_grad():
            o = model(batch, output_hidden_states=True)
        for L, hsidx in LAYERS.items():
            buf[L].append(o.hidden_states[hsidx][:, -1, :].float().cpu().half())
        if (i // BATCH) % 100 == 0:
            print(f"  [{time.time()-t0:.0f}s] {i}/{len(chunks)}", flush=True)

    meta = {"model": "Qwen/Qwen3-1.7B", "d_model": 2048, "seq_len": SEQ_LEN,
            "layers": list(LAYERS), "n": len(chunks), "dataset": "fineweb-edu sample-10BT",
            "token": "last (position 9)"}
    for L in LAYERS:
        arr = torch.cat(buf[L]).numpy()
        np.save(os.path.join(CACHE, f"acts_qwen_L{L}.npy"), arr)
        print(f"saved acts_qwen_L{L}.npy {arr.shape}", flush=True)
    json.dump(meta, open(os.path.join(CACHE, "collect_meta.json"), "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] done. VRAM peak MB {torch.cuda.max_memory_allocated()/1e6:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
