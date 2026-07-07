"""Controlled-experiment collector: Qwen3-1.7B layer-2 activations, ALL token
positions pooled (vs last-token in collect_qwen.py). Everything else identical
(same model, layer, fineweb-edu text, seq_len=10). This isolates the effect of
the "last-token only" choice (factor #3) on the massive-activation dimension and
the AE reconstruction curve. Saves acts_qwen_L2_pooled.npy."""
import os, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.180)
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TEXT_CACHE = os.path.join(CACHE, "fineweb_edu_texts.json")
SEQ_LEN = 10
HSIDX = 3           # layer 2 -> hidden_states[3]
TARGET = 160_000    # pooled vectors
BATCH = 64


def main():
    t0 = time.time()
    texts = json.load(open(TEXT_CACHE))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("cuda").eval()
    chunks = []
    n_seq = TARGET // SEQ_LEN + BATCH
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        for s in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            chunks.append(ids[s:s + SEQ_LEN])
            if len(chunks) >= n_seq:
                break
        if len(chunks) >= n_seq:
            break
    print(f"[{time.time()-t0:.0f}s] {len(chunks)} seqs -> up to {len(chunks)*SEQ_LEN} pooled", flush=True)
    buf = []
    for i in range(0, len(chunks), BATCH):
        batch = torch.tensor(chunks[i:i + BATCH], device="cuda")
        with torch.no_grad():
            o = model(batch, output_hidden_states=True)
        buf.append(o.hidden_states[HSIDX].reshape(-1, 2048).float().cpu().half())
        if len(torch.cat(buf)) >= TARGET:  # cheap enough at this size
            break
    arr = torch.cat(buf).numpy()[:TARGET]
    np.save(os.path.join(CACHE, "acts_qwen_L2_pooled.npy"), arr)
    print(f"[{time.time()-t0:.0f}s] saved acts_qwen_L2_pooled.npy {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
