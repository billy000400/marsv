"""Shared setup for dir27: ' big' -> token B interpolation at layer 0, widths at blocks 18 and 35."""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

MODEL = "gpt2-large"
PROMPT = "The house was big"
MID_BLOCK = 18     # output of transformer.h[18] (0-indexed) = hidden_states[19]
FINAL_BLOCK = 35   # output of transformer.h[35], before ln_f

N_T = 101
TS = np.linspace(0.0, 1.0, N_T)

# green-free CVD palette (CLAUDE.md rule 13)
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def load():
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.225)
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m, dev


class Runner:
    """Runs interpolated last-token embeddings; returns last-position outputs of MID and FINAL blocks.

    Layer 0 = the embedding layer. wpe at the last position is shared by A and B, so linear
    interpolation of wte[A], wte[B] equals linear interpolation of the layer-0 hidden states.
    """

    def __init__(self, tok, m, dev):
        self.m, self.dev = m, dev
        self.ids = tok.encode(PROMPT)
        self.id_a = self.ids[-1]
        self.wte = m.transformer.wte.weight
        self.prefix = self.wte[torch.tensor(self.ids[:-1], device=dev)]
        self.cap = {}
        m.transformer.h[MID_BLOCK].register_forward_hook(self._hook("mid"))
        m.transformer.h[FINAL_BLOCK].register_forward_hook(self._hook("final"))

    def _hook(self, name):
        def f(mod, inp, out):
            self.cap[name] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()
        return f

    @torch.no_grad()
    def run(self, last_emb):
        """last_emb: (N, d) layer-0 token embeddings for the last position."""
        n = last_emb.shape[0]
        emb = torch.cat([self.prefix.unsqueeze(0).expand(n, -1, -1), last_emb.unsqueeze(1)], dim=1)
        self.m.transformer(inputs_embeds=emb, use_cache=False)
        return self.cap["mid"], self.cap["final"]


def y_curve(h):
    """h: (..., N_T, d) readout states along t. y(t) = |h(t)-h(0)| / |h(1)-h(0)|."""
    num = (h - h[..., :1, :]).norm(dim=-1)
    return num / num[..., -1:]


def crossings(ts, y, level):
    """All linearly-interpolated crossings of `level` (upward or downward)."""
    out = []
    for i in range(len(y) - 1):
        lo, hi = y[i], y[i + 1]
        if lo == level:
            out.append(float(ts[i]))
        elif (lo - level) * (hi - level) < 0:
            out.append(float(ts[i] + (level - lo) / (hi - lo) * (ts[i + 1] - ts[i])))
    if y[-1] == level:
        out.append(float(ts[-1]))
    return out


def width(ts, y):
    """W = t_0.9 - t_0.1 using the first crossing of each level; flag if any level is crossed >1 time."""
    c1, c9 = crossings(ts, y, 0.1), crossings(ts, y, 0.9)
    t1, t9 = c1[0], c9[0]
    backstep = float(max(0.0, -np.diff(y).min()))
    return t1, t9, t9 - t1, len(c1), len(c9), backstep
