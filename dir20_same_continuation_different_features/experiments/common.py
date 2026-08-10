"""Shared setup: budget-respecting device/threads, model loading, prompt pairs, hooks."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

MODELS = {
    "gpt2-medium": dict(name="gpt2-medium", revision=None),
    "pythia-410m": dict(name="EleutherAI/pythia-410m-deduped", revision="step143000"),
    "opt-350m": dict(name="facebook/opt-350m", revision=None),
    # S7: depth-mismatched members of the same family (12 and 36 blocks vs gpt2-medium's 24).
    "gpt2-small": dict(name="gpt2", revision=None),
    "gpt2-large": dict(name="gpt2-large", revision=None),
}

N_BLOCKS = {"gpt2-medium": 24, "pythia-410m": 24, "opt-350m": 24,
            "gpt2-small": 12, "gpt2-large": 36}

# (key, label, prefix, final token A, final token B, is_control)
PAIRS = [
    ("book_to", "gave a book to Mary / her",
     "Mary and John went to the store. John gave a book to", " Mary", " her", False),
    ("two_plus_two", "Two plus two is four / 4",
     "Two plus two is", " four", " 4", False),
    ("answer_case", "The answer is four / Four",
     "The answer is", " four", " Four", False),
    ("element_au", "clue identify? Au / 79",
     "Which chemical element does this clue identify?", " Au", " 79", False),
    # Matthew's own pair: his POSITIVE plateau example (not a negative control).
    ("house_big_in", "The house was big / in (M. plateau case)",
     "The house was", " big", " in", True),
    # Matthew's smooth comparison: two adjectives, no plateau reported.
    ("house_big_large", "The house was big / large (M. smooth case)",
     "The house was", " big", " large", True),
]

# Shared-box budget (1 of 4 agents on a 31.9 GB card).
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(key):
    spec = MODELS[key]
    kw = {} if spec["revision"] is None else dict(revision=spec["revision"])
    tok = AutoTokenizer.from_pretrained(spec["name"], **kw)
    m = AutoModelForCausalLM.from_pretrained(spec["name"], torch_dtype=torch.float32, **kw)
    m.eval().to(DEV)
    return tok, m


def blocks(m):
    """The transformer blocks; each block's output[0] is that block's resid_post."""
    if hasattr(m, "transformer"):          # GPT-2
        return m.transformer.h
    if hasattr(m, "gpt_neox"):             # Pythia / GPT-NeoX
        return m.gpt_neox.layers
    return m.model.decoder.layers          # OPT


def tokenize_pair(tok, prefix, a_str, b_str):
    """Validate: identical tokenized prefix + exactly one differing single final token."""
    pre = tok(prefix)["input_ids"]
    ida = tok(prefix + a_str)["input_ids"]
    idb = tok(prefix + b_str)["input_ids"]
    ok = (ida[:-1] == pre and idb[:-1] == pre and len(ida) == len(pre) + 1
          and len(idb) == len(pre) + 1 and ida[-1] != idb[-1])
    return ok, ida, idb
