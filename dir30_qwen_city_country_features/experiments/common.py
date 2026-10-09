"""Shared constants and loaders for the Qwen3-0.6B city/country transcoder-feature study."""
import os
import shutil

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
LRE_DIR = os.path.join(ROOT, "data", "lre")

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
TC_REPO = "mwhanna/qwen3-0.6b-transcoders-lowl0"
TC_REV = "28aefe686c09e9bc1a862195b51e145f10868d87"
LRE_REV = "1b9ec3cf2b8368e42bde7e80fcef384312a6ec07"
N_LAYERS = 28
SEED = 0
VRAM_FRACTION = 0.45
# The shared HF cache volume hit its disk quota after three transcoder layers, so the
# remaining layers are streamed through local scratch one at a time and deleted after use.
SCRATCH = "/tmp/hf_dir30"
DEV = "cuda"


def setup_torch():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, torch_dtype=torch.float32)
    return tok, model.to(DEV).eval()


def load_transcoder(layer):
    """Return (W_enc, b_enc, W_dec, b_dec) as float32 on GPU. W_enc and W_dec are [n_features, d_model]."""
    name = f"layer_{layer}.safetensors"
    path = try_to_load_from_cache(TC_REPO, name, revision=TC_REV)
    scratch = not isinstance(path, str)
    if scratch:
        path = hf_hub_download(TC_REPO, name, revision=TC_REV, cache_dir=SCRATCH)
    sd = load_file(path)
    if scratch:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    assert sd["W_enc"].shape == sd["W_dec"].shape == (163840, 1024), {k: v.shape for k, v in sd.items()}
    return tuple(sd[k].to(DEV, torch.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec"))


def encode(x, W_enc, b_enc):
    """ReLU feature activations, following circuit-tracer's SingleLayerTranscoder.encode."""
    return torch.relu(x @ W_enc.T + b_enc)


class MLPTap:
    """Records the input and output of every layer's MLP during a forward pass."""

    def __init__(self, model):
        self.x, self.y = {}, {}
        self.handles = []
        for i, block in enumerate(model.model.layers):
            self.handles.append(block.mlp.register_forward_hook(self._hook(i)))

    def _hook(self, i):
        def fn(module, args, output):
            self.x[i], self.y[i] = args[0].detach(), output.detach()
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()
