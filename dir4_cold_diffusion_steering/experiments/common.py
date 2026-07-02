"""Shared helpers for the cold-diffusion-steering direction.

Model: GPT-2 small (124M) via HF transformers. Hook point: resid_post at a
block output. resid_post for block L == GPT2Model hidden_states[L+1].
All activations are RAW (no centering / no unit-normalization).
"""
import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.18)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FINEWEB = os.path.join(HERE, "..", "..", "dir3_manifold", "data", "fineweb_texts.json")

_model = None
_tok = None


def load_model():
    global _model, _tok
    if _model is None:
        _tok = GPT2TokenizerFast.from_pretrained("gpt2")
        _tok.pad_token = _tok.eos_token
        _model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    return _model, _tok


def fineweb_texts(n):
    import json
    with open(FINEWEB) as f:
        return json.load(f)[:n]


@torch.no_grad()
def resid_post(texts, layer, seq_len=128, batch=16):
    """Return raw resid_post activations at `layer`, flattened over
    (non-pad) token positions: array [N_tokens, 768] float32, plus a
    per-token attention mask flag list is dropped (only real tokens kept)."""
    model, tok = load_model()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        hs = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states
        h = hs[layer + 1]                       # [B, T, 768] resid_post of block `layer`
        m = mask.bool()
        out.append(h[m].float().cpu().numpy())
    return np.concatenate(out, axis=0)


class Patcher:
    """Context manager: add `delta` (broadcast [768]) to resid_post at `layer`
    for every token position during a forward pass."""
    def __init__(self, model, layer, delta):
        self.block = model.transformer.h[layer]
        self.delta = delta
        self.handle = None

    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.delta,) + out[1:]
        return out + self.delta

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


@torch.no_grad()
def lm_loss(texts, layer, delta, seq_len=128, batch=16):
    """Mean next-token cross-entropy (nats) with `delta` added to resid_post
    at `layer`. delta=0 tensor gives the clean baseline."""
    model, tok = load_model()
    total, count = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=seq_len, padding="max_length")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        with Patcher(model, layer, delta):
            logits = model(ids, attention_mask=mask).logits
        # shift for next-token prediction
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        m = mask[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += nll[m].sum().item()
        count += m.sum().item()
    return total / count
