"""PLAN Exp 2.1 gate: are Matthew's completions single GPT-2 BPE tokens after the shared context,
and how often does each completion token occur in the BPE-tokenized training corpus?
Writes ../tokenization_check.txt (both tokenizers) and caches the BPE corpus to /tmp.
"""
import os, sys, json, hashlib
import numpy as np
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tokenization_check.txt")

CONTEXT = "The house was"
WORDS = ["big", "in", "large"]
CHAR_CONTROLS = ["b", "i", "l"]

raw = open("/tmp/tinyshakespeare.txt", "rb").read()
sha = hashlib.sha256(raw).hexdigest()
text = raw.decode("utf-8")
tkz = GPT2TokenizerFast.from_pretrained("gpt2")

cache = "/tmp/tinyshakespeare_gpt2bpe.npy"
if os.path.exists(cache):
    ids = np.load(cache)
else:
    ids = np.array(tkz(text)["input_ids"], dtype=np.uint16)
    np.save(cache, ids)
n_train = int(0.9 * len(ids))
train_ids = ids[:n_train]

lines = [f"corpus sha256={sha} chars={len(text)} gpt2_bpe_tokens={len(ids)} train_tokens={n_train}",
         f"shared_context={CONTEXT!r}", ""]
ctx_ids = tkz(CONTEXT)["input_ids"]
lines.append(f"context token ids={ctx_ids} decoded={[tkz.decode([i]) for i in ctx_ids]}")
gate_pass = True
for w in WORDS:
    full = tkz(CONTEXT + " " + w)["input_ids"]
    comp = full[len(ctx_ids):]
    single = len(comp) == 1
    gate_pass &= single
    tid = comp[0] if single else None
    freq = int((train_ids == tid).sum()) if single else -1
    freq_total = int((ids == tid).sum()) if single else -1
    lines.append(f"completion ' {w}': ids={comp} decoded={[tkz.decode([i]) for i in comp]} "
                 f"single_token={single} train_count={freq} corpus_count={freq_total}")

# char tokenizer control: b, i, l are trivially single char tokens; record their train counts
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
char_arr = np.array([stoi[c] for c in text], dtype=np.uint16)
char_train = char_arr[:int(0.9 * len(char_arr))]
lines.append("")
for c in CHAR_CONTROLS:
    lines.append(f"char control '{c}': id={stoi[c]} train_count={int((char_train == stoi[c]).sum())}")

lines.append("")
lines.append(f"GATE (each completion is one BPE token after context): {'PASS' if gate_pass else 'FAIL'}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
json.dump({"gate_pass": bool(gate_pass)}, open(os.path.join(ROOT, "results", "tokenization_gate.json"), "w"))
