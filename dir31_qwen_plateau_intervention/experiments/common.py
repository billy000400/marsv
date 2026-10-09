"""Shared setup for the plateau-intervention study. Reuses dir30's pinned Qwen3-0.6B / transcoder loader."""
import contextlib
import importlib.util
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dir30 loader (pinned revisions, float32, VRAM fraction 0.45, 4 threads); loaded by path because it is also named common.py
_spec = importlib.util.spec_from_file_location(
    "dir30_common", os.path.join(os.path.dirname(ROOT), "dir30_qwen_city_country_features", "experiments", "common.py"))
d30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d30)

OUT = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
SLICES = os.path.join(OUT, "slices")
DEV = d30.DEV
N_LAYERS = d30.N_LAYERS
SEED = 0
N_T = 41
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]

COUNTRIES = ["Japan", "Germany"]
CITY = {"Japan": "Tokyo", "Germany": "Berlin"}
# Capital templates that Qwen answers correctly for BOTH countries (dir30 samples.jsonl). prefix ends at the
# country token; bridge is teacher-forced after it. "tune" selects the edit; the other two are reserved.
TEMPLATES = {
    "tune": ("The capital city of {}", " is"),
    "original": ("The capital of {}", " is"),
    "city_of": ("The capital of {}", " is the city of"),
}
# Explicit-country, non-capital prefixes that END at the country token (always " Japan"/" Germany", never position 0).
PROBE_FAMILIES = {
    "currency": ["The official currency of {}", "The money used in {}", "Prices in {}", "The central bank of {}"],
    "language": ["People in {}", "The language used in {}", "Most people living in {}", "Schools in {}"],
    "identity": ["I was born in {}", "My friend is from {}", "She moved to {}", "We are talking about {}"],
    "general": ["The population of {}", "The flag of {}", "The weather in {}", "The history of {}"],
}


def setup():
    d30.setup_torch()
    return d30.load_model()


def ids_of(tok, text):
    return tok(text, return_tensors="pt").input_ids.to(DEV)


def slerp_lerp_norm(ea, eb, ts):
    """Project-standard interpolation (dir22/dir23): shortest-arc SLERP of direction, linear L2 norm."""
    na, nb = ea.norm(), eb.norm()
    u, v = ea / na, eb / nb
    om = torch.arccos(torch.clamp((u * v).sum(), -1.0, 1.0))
    a = torch.tensor(ts, device=ea.device, dtype=ea.dtype).unsqueeze(1)
    d = (torch.sin((1 - a) * om) * u + torch.sin(a * om) * v) / torch.sin(om)
    return ((1 - a) * na + a * nb) * d, float(om)


class Edit:
    """MLP_new = MLP_native + beta * sum_i (a_donor_i - a_current_i) * decoder_i at one layer and position.
    a_current is recomputed from this run's MLP input. If ctrl_dec is given, the same coefficients are put on
    those (random) decoder directions and the result is rescaled to the real edit's L2 norm."""

    def __init__(self, layer, pos, W_enc, b_enc, W_dec, a_donor, beta, ctrl_dec=None):
        self.layer, self.pos, self.beta = layer, pos, beta
        self.W_enc, self.b_enc, self.W_dec, self.a_donor, self.ctrl_dec = W_enc, b_enc, W_dec, a_donor, ctrl_dec
        self.norm = None

    def delta(self, x):
        coef = self.beta * (self.a_donor - torch.relu(x @ self.W_enc.T + self.b_enc))
        real = coef @ self.W_dec
        self.norm = real.norm(dim=-1).detach()
        if self.ctrl_dec is None:
            return real
        c = coef @ self.ctrl_dec
        return c * (real.norm(dim=-1, keepdim=True) / c.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def _mlp_hook(e):
    def fn(module, args, output):
        output = output.clone()
        output[:, e.pos] += e.delta(args[0][:, e.pos])
        return output
    return fn


@contextlib.contextmanager
def edited(model, edits):
    """Keep the edits active for every forward pass inside the block (used for greedy generation without KV cache)."""
    handles = [model.model.layers[e.layer].mlp.register_forward_hook(_mlp_hook(e)) for e in edits]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run(model, ids, pos, patch0=None, edits=(), read_pos=None):
    """One forward pass of the native model (no KV cache). ids: [B, T]. pos: country position.
    patch0: [B, d] replacement for the layer-0 block output at pos. edits: Edit objects.
    Returns dict(resid=[B, 28, d] block outputs at read_pos (default pos), logits_pos=[B, V], logits_last=[B, V])."""
    read_pos = pos if read_pos is None else read_pos
    resid, handles = [None] * N_LAYERS, []

    def block_hook(i):
        def fn(module, args, output):
            if i == 0 and patch0 is not None:
                output = output.clone()
                output[:, pos] = patch0
            resid[i] = output[:, read_pos].detach()
            return output
        return fn

    for i in range(N_LAYERS):
        handles.append(model.model.layers[i].register_forward_hook(block_hook(i)))
    try:
        with edited(model, edits):
            logits = model(ids).logits
    finally:
        for h in handles:
            h.remove()
    return dict(resid=torch.stack(resid, 1), logits_pos=logits[:, pos], logits_last=logits[:, -1])


def jsd_bits(p, q):
    """Jensen-Shannon divergence in bits between probability rows."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log2(a.clamp_min(1e-30)) - torch.log2(b.clamp_min(1e-30)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
