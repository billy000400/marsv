"""Shared setup: budget-respecting device/threads, model loading, token eligibility."""
import os

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/tmp/dir19_data"  # frozen valid-target-ID list copied from dir18
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

MODEL = "EleutherAI/pythia-1.4b-deduped"
FINAL = "step143000"
CONTEXTS = ["The thing was", "They said it was", "I thought it was"]

# Shared-box budget (1 of 4 agents on a 31.9 GB card).
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)


def load(revision=FINAL, model=MODEL):
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    m = GPTNeoXForCausalLM.from_pretrained(model, revision=revision, torch_dtype=torch.float32)
    m.eval().cuda()
    return tok, m


def eligible_word_tokens(tok, vocab_size):
    """Lowercase, alphabetic WORD-START tokens (not necessarily complete words).

    GPT-NeoX BPE marks a word start with 'Ġ'. We require >= 2 alphabetic lowercase characters, so
    single letters are excluded -- but a multi-letter word-start prefix such as ' un' still passes.
    revisions.py reports the sensitivity check that drops the one such pair from the bank.
    """
    ids = []
    for i in range(vocab_size):
        s = tok.convert_ids_to_tokens(i)
        if s is None or not s.startswith("Ġ"):
            continue
        w = s[1:]
        if len(w) >= 2 and w.isalpha() and w.islower() and w.isascii():
            ids.append(i)
    return ids
