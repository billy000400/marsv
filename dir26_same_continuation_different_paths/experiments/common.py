"""Shared setup for dir26: GPT-2 Large, the four prompt pairs, full-sequence block-0 interpolation."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")

MODEL = "gpt2-large"

# (key, label, prompt A, prompt B, intended completion)
PAIRS = [
    ("p1_copy_vs_recall", "Copy available vs factual recall",
     "The capital of France is Paris. The capital of France is",
     "The capital of Spain is Madrid. The capital of France is",
     "Paris"),
    ("p2_succ_vs_pred", "Successor vs predecessor",
     "The number after six is",
     "The number before eight is",
     "seven"),
    ("p3_relation", "Different relation, same answer",
     "The capital of France is",
     "The largest city in France is",
     "Paris"),
    ("p4_entity", "Different entity, same relation and answer",
     "The author of Hamlet is",
     "The author of Macbeth is",
     "Shakespeare"),
]

torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.225)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.eval().to(DEV)
    return tok, m


def blocks(m):
    """Transformer blocks; block i's output[0] is resid_post after block i."""
    return m.transformer.h
