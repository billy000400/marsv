"""Feedback 3: the dir27 ' big' -> token B study on other models (GPT2-XL, Pythia 1.4B, Llama-3.1-8B, Qwen3-8B-Base).

Same prompt, same interpolation, same W definition as common.py (GPT-2 Large). Only the last position is run:
the prefix keys/values are computed once and reused (exact, since the prefix does not depend on B).
8B models stay in their native bf16; blocks that do not fit the GPU budget are copied in from pinned CPU
memory for each forward pass (exact, no quantization).
"""
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from common import PROMPT, ROOT, TS  # noqa: F401  (TS re-exported)

GEN = os.path.join(ROOT, "results", "gen")
GPLOTS = os.path.join(ROOT, "plots", "gen")
os.makedirs(GEN, exist_ok=True)
os.makedirs(GPLOTS, exist_ok=True)

# key -> (HF repo, dtype, number of blocks kept resident on the GPU; None = all)
MODELS = {
    "gpt2-xl": ("openai-community/gpt2-xl", torch.float32, None),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", torch.float32, None),
    "llama-3.1-8b": ("unsloth/Meta-Llama-3.1-8B", torch.bfloat16, 22),
    "qwen3-8b-base": ("Qwen/Qwen3-8B-Base", torch.bfloat16, 24),
}
NAMES = {"gpt2-xl": "GPT2-XL", "pythia-1.4b": "Pythia 1.4B", "llama-3.1-8b": "Llama-3.1-8B",
         "qwen3-8b-base": "Qwen3-8B-Base"}


def blocks_of(base):
    return base.h if hasattr(base, "h") else base.layers


class FrozenPrefix(DynamicLayer):
    """Cache layer holding the prefix keys/values; returns prefix + new keys without storing them."""

    def __init__(self, k, v):
        super().__init__()
        self.keys, self.values, self.is_initialized = k, v, True
        self.dtype, self.device = k.dtype, k.device

    def update(self, key_states, value_states, *args, **kwargs):
        n = key_states.shape[0]
        return (torch.cat([self.keys.expand(n, -1, -1, -1), key_states], dim=-2),
                torch.cat([self.values.expand(n, -1, -1, -1), value_states], dim=-2))


def _stream(block):
    """Keep `block` weights in pinned CPU memory; copy them to the GPU only while the block runs."""
    ps = list(block.parameters())
    host = [p.data.pin_memory() for p in ps]
    for p, h in zip(ps, host):
        p.data = h

    def pre(mod, args, kwargs=None):
        for p, h in zip(ps, host):
            p.data = h.to("cuda", non_blocking=True)

    def post(mod, args, out):
        for p, h in zip(ps, host):
            p.data = h
    block.register_forward_pre_hook(pre)
    block.register_forward_hook(post)


def load(key):
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.45)
    repo, dtype, resident = MODELS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    if resident is None:
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to("cuda").eval()
    else:
        n = AutoConfig.from_pretrained(repo).num_hidden_layers
        dm = {"model.embed_tokens": "cpu", "lm_head": "cpu", "model.norm": 0, "model.rotary_emb": 0}
        dm.update({f"model.layers.{i}": (0 if i < resident else "cpu") for i in range(n)})
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=dm).eval()
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(m)
        for i in range(resident, n):
            _stream(m.model.layers[i])
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


class Runner:
    """Layer 0 = token embedding row. Readouts: last-position output of the middle block (n//2, 0-indexed,
    as block 18 of 36 in GPT-2 Large) and of the final block (before the final norm)."""

    def __init__(self, tok, m):
        self.m, self.base = m, m.base_model
        self.ids = tok(PROMPT)["input_ids"]
        self.id_a = self.ids[-1]
        self.emb = self.base.get_input_embeddings().weight
        blocks = blocks_of(self.base)
        self.n_blocks = len(blocks)
        self.mid, self.final = self.n_blocks // 2, self.n_blocks - 1
        self.cap = {}
        blocks[self.mid].register_forward_hook(self._hook("mid"))
        blocks[self.final].register_forward_hook(self._hook("final"))
        pre = self.rows(self.ids[:-1]).unsqueeze(0)
        with torch.no_grad():
            c = self.fwd(inputs_embeds=pre, use_cache=True).past_key_values
        self.cache = DynamicCache()
        self.cache.layers = [FrozenPrefix(L.keys, L.values) for L in c.layers]
        self.P = pre.shape[1]

    def fwd(self, **kw):
        """bf16 models: bf16 weights and matmuls, but a float32 residual stream (inputs are float32).
        A bf16 residual stream (norm ~1500 at Qwen3's final block) made W depend on batch shape."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.emb.dtype == torch.bfloat16):
            return self.base(**kw)

    def rows(self, ids):
        return self.emb[torch.as_tensor(ids, device=self.emb.device)].to("cuda", torch.float32)

    def _hook(self, name):
        def f(mod, inp, out):
            self.cap[name] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()
        return f

    @torch.no_grad()
    def run(self, last_emb):
        """last_emb: (N, d) float32 layer-0 embeddings for the last position -> (mid, final) (N, d)."""
        n = last_emb.shape[0]
        x = last_emb.unsqueeze(1)
        pos = torch.full((n, 1), self.P, device="cuda", dtype=torch.long)
        self.fwd(inputs_embeds=x, past_key_values=self.cache, position_ids=pos, use_cache=True)
        return self.cap["mid"], self.cap["final"]

    @torch.no_grad()
    def run_full(self, last_emb):
        """Reference path: whole prompt, no cache. Used only to check `run`."""
        n = last_emb.shape[0]
        pre = self.rows(self.ids[:-1]).unsqueeze(0).expand(n, -1, -1)
        x = torch.cat([pre, last_emb.unsqueeze(1)], dim=1)
        self.fwd(inputs_embeds=x, use_cache=False)
        return self.cap["mid"], self.cap["final"]
