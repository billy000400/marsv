"""Shared helpers for Direction 10 — combined path-smoothness vs. weekday manifold.

Model: Llama 3.1 8B *base* (meta-llama/Llama-3.1-8B), bf16 compute.
Hook point: residual stream at layer 28 (HF hidden_states[28] == output of the
28th transformer block, 1-indexed), last (answer-predicting) token position.

MEMORY NOTE (shared-box budget): 8B in bf16 is ~16 GB > our 7.2 GB VRAM share.
We therefore load with device_map splitting weights across a capped GPU slice
(~6 GiB) and CPU RAM (the box physically has ~230 GB free). This is a forced
resource deviation from the paper's "full bf16 on GPU"; the *precision* is still
bf16, only the *placement* differs, so activations are unchanged.
"""
import os
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HOME"] = ("/network/hf_cache"
                         if os.path.isdir("/network/hf_cache")
                         else "/workspace/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.225)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

_SNAP = ("hub/models--meta-llama--Llama-3.1-8B/"
         "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
MODEL_ID = (f"/network/hf_cache/{_SNAP}"
            if os.path.isdir(f"/network/hf_cache/{_SNAP}")
            else f"/workspace/hf_cache/{_SNAP}")
LAYER = 28  # hidden_states index (1-indexed block output)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
INCREMENTS = ["one", "two", "three", "four", "five", "six", "seven"]

_model = None
_tok = None


def load_model(gpu_gib=6):
    """Load Llama 3.1 8B base, bf16, weights split GPU(cap)/CPU via accelerate."""
    global _model, _tok
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        max_mem = {0: f"{gpu_gib}GiB", "cpu": "60GiB"}
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_mem,
        ).eval()
    return _model, _tok


def build_prompts():
    """The 49 weekday-addition prompts + ground-truth answers.

    Template EXACTLY: 'Q: What day is {k} days after {entity}?\\nA:'
    Ground truth wraps cyclically mod 7.
    Returns list of dicts: prompt, entity, entity_idx, k_word, k, gt_idx, gt.
    """
    rows = []
    for e_idx, entity in enumerate(WEEKDAYS):
        for k, k_word in enumerate(INCREMENTS, start=1):
            gt_idx = (e_idx + k) % 7
            rows.append(dict(
                prompt=f"Q: What day is {k_word} days after {entity}?\nA:",
                entity=entity, entity_idx=e_idx, k_word=k_word, k=k,
                gt_idx=gt_idx, gt=WEEKDAYS[gt_idx],
            ))
    return rows


def weekday_token_ids(tok):
    """Map each weekday -> list of tokenizer token IDs whose decoded surface
    form (stripped, lowercased) equals the weekday name. Captures spelling /
    leading-space variants (Appendix A.2)."""
    vocab = tok.get_vocab()  # token string -> id
    ids = {w: [] for w in WEEKDAYS}
    lower = {w.lower(): w for w in WEEKDAYS}
    for tid in range(len(tok)):
        s = tok.decode([tid]).strip().lower()
        if s in lower:
            ids[lower[s]].append(tid)
    return ids
