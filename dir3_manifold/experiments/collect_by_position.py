"""Token-position-stratified layer-6 collection + ID (addresses REVIEW overclaim #5).

The main cache pooled ALL token positions, so its ID estimate confounds token
position with residual geometry. Here we re-collect layer-6 resid_post WITH the
absolute token position of each vector, bucket by position, and run TwoNN/MLE per
bucket. CPU (GPT-2 small is light) so it runs alongside the GPU AE job.
Saves data/acts_layer6_pos.npy (+ _posidx.npy) and results/id_by_position.json.
"""
import os, json, time
import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast
from id_estimate import twonn, mle

torch.set_num_threads(int(os.environ.get("CPU_THREADS_PER_AGENT", "2")))
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
LAYER = 6
SEQ_LEN = 256
N_TEXTS = 600
BATCH = 16
TARGET = 80_000

texts = json.load(open(os.path.join(DATA, "fineweb_texts.json")))[:N_TEXTS]
tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
model = GPT2Model.from_pretrained("gpt2").eval()
t0 = time.time()

vecs, poss = [], []
n = 0
for i in range(0, len(texts), BATCH):
    if n >= TARGET:
        break
    enc = tok(texts[i:i + BATCH], return_tensors="pt", truncation=True,
              max_length=SEQ_LEN, padding=True)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    h = out.hidden_states[LAYER + 1]                 # (B,T,768) resid_post layer6
    mask = enc["attention_mask"].bool()
    B, T = mask.shape
    posgrid = torch.arange(T).unsqueeze(0).expand(B, T)
    sel = h[mask].float().numpy().astype(np.float16)
    psel = posgrid[mask].numpy().astype(np.int16)
    vecs.append(sel); poss.append(psel); n += sel.shape[0]
    if (i // BATCH) % 5 == 0:
        print(f"[{time.time()-t0:.0f}s] n={n}", flush=True)

V = np.concatenate(vecs)[:TARGET]
P = np.concatenate(poss)[:TARGET]
np.save(os.path.join(DATA, "acts_layer6_pos.npy"), V)
np.save(os.path.join(DATA, "acts_layer6_posidx.npy"), P)
print(f"collected {V.shape}, pos range {P.min()}-{P.max()}", flush=True)

# Position buckets (absolute token index within sequence)
buckets = {"pos0": (0, 0), "early_1_15": (1, 15), "mid_16_63": (16, 63),
           "late_64_127": (64, 127), "tail_128_255": (128, 255)}
mu = V.astype(np.float32).mean(0, keepdims=True)
results = []
for name, (lo, hi) in buckets.items():
    m = (P >= lo) & (P <= hi)
    Xb = V[m].astype(np.float32)
    nb = Xb.shape[0]
    if nb < 1500:
        print(f"{name}: only {nb} pts, skipping", flush=True)
        results.append({"bucket": name, "pos_lo": lo, "pos_hi": hi, "n": int(nb),
                        "twonn": None, "mle_invavg": None})
        continue
    if nb > 30000:
        idx = np.random.default_rng(0).choice(nb, 30000, replace=False)
        Xb = Xb[idx]; nb = 30000
    Xc = np.ascontiguousarray(Xb - mu, dtype=np.float32)
    tn = twonn(Xc); mi, _ = mle(Xc)
    results.append({"bucket": name, "pos_lo": lo, "pos_hi": hi, "n": int(nb),
                    "twonn": tn, "mle_invavg": mi})
    print(f"[{time.time()-t0:.0f}s] {name:14s} n={nb:6d} TwoNN={tn:.2f} MLE={mi:.2f}",
          flush=True)
    json.dump(results, open(os.path.join(RES, "id_by_position.json"), "w"), indent=2)
print("DONE by_position", flush=True)
